import copy
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, Generic, List, Literal, Optional, Tuple, TypeVar, Union

from utils.ab_mcts_a.prob_state import NodeProbState, PriorConfig
from utils.base import Algorithm
from utils.tree import Node, Tree
from utils.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")

logger = getLogger(__name__)


@dataclass
class ABMCTSAStateManager(Generic[StateT]):
    """Manager for ABMCTSAState instances associated with expand_idx values."""

    states: Dict[int, NodeProbState] = field(default_factory=dict)
    default_prior_config: Optional["PriorConfig"] = None
    default_reward_average_priors: Optional[Union[float, Dict[str, float]]] = None
    default_model_selection_strategy: str = "multiarm_bandit_thompson"

    def __contains__(self, expand_idx: int) -> bool:
        """Check if a thompson state exists for the given expand_idx."""
        return expand_idx in self.states

    def __len__(self) -> int:
        return len(self.states)

    def get(self, node: Node) -> Optional[NodeProbState]:
        """Get thompson state for the given expand_idx if it exists."""
        return self.states.get(node.expand_idx)

    def create(
        self,
        node: Node,
        actions: List[str],
        prior_config: Optional["PriorConfig"] = None,
        reward_average_priors: Optional[Union[float, Dict[str, float]]] = None,
        model_selection_strategy: Optional[str] = None,
    ) -> NodeProbState:
        """Create a new thompson state for the given expand_idx with optional prior configuration."""
        # Use provided configs or fall back to defaults
        prior_config = prior_config or self.default_prior_config
        reward_average_priors = (
            reward_average_priors or self.default_reward_average_priors
        )
        model_selection_strategy = (
            model_selection_strategy or self.default_model_selection_strategy
        )

        state = NodeProbState(
            actions=actions,
            prior_config=prior_config,
            reward_average_priors=reward_average_priors,
            model_selection_strategy=model_selection_strategy,
        )
        self.states[node.expand_idx] = state
        return state

    def get_or_create(
        self,
        node: Node,
        actions: List[str],
        prior_config: Optional["PriorConfig"] = None,
        reward_average_priors: Optional[Union[float, Dict[str, float]]] = None,
        model_selection_strategy: Optional[str] = None,
    ) -> NodeProbState:
        """Get existing thompson state or create a new one if it doesn't exist."""
        if node.expand_idx in self.states:
            return self.states[node.expand_idx]
        return self.create(
            node,
            actions,
            prior_config=prior_config,
            reward_average_priors=reward_average_priors,
            model_selection_strategy=model_selection_strategy,
        )


def build_default_dict_of_list() -> Dict[str, List[float]]:
    """
    For pickle we define it at the top-module level as named function.
    Ref: https://docs.python.org/3.13/library/pickle.html#what-can-be-pickled-and-unpickled
    """
    return defaultdict(list)


@dataclass
class ABMCTSAAlgoState(Generic[StateT]):
    """State for ABMCTSA algorithm."""

    tree: Tree[StateT]
    thompson_states: ABMCTSAStateManager = field(default_factory=ABMCTSAStateManager)
    all_rewards_store: Dict[str, List[float]] = field(
        default_factory=build_default_dict_of_list
    )


class ABMCTSA(Algorithm[StateT, ABMCTSAAlgoState[StateT]]):
    """
    Adaptive Monte Carlo Tree Search algorithm with Node Aggregation.

    This algorithm uses Thompson Sampling to decide whether to generate a new node (GEN)
    or continue exploring from an existing one (CONT).
    """

    def __init__(
        self,
        dist_type: Literal["gaussian", "beta"] = "gaussian",
        reward_average_priors: Optional[Union[float, Dict[str, float]]] = None,
        prior_config: Optional[PriorConfig] = None,
        model_selection_strategy: str = "multiarm_bandit_thompson",
    ):
        """
        Initialize the AB-MCTS-A algorithm.

        Args:
            dist_type: Type of Probability distribution. Either "gaussian" or "beta".
            reward_average_priors: Optional prior reward average values (global or per-action)
            prior_config: Optional prior configuration for Thompson sampling distributions. If specified, dist_type will not be used.
            model_selection_strategy: Strategy for model selection. One of:
                - "stack": Perform separate fits for each model (traditional approach)
                - "multiarm_bandit_thompson": Use Thompson Sampling for joint selection
                - "multiarm_bandit_ucb": Use UCB for joint selection
        """
        if prior_config is None:
            prior_config = PriorConfig(dist_type=dist_type)
        else:
            if dist_type != prior_config.dist_type:
                logger.warning(
                    f"dist_type argument {dist_type} is different from the one specified by prior_config {prior_config.dist_type}. {dist_type} is ignored, and {prior_config.dist_type} will be used."
                )

        self.prior_config = prior_config
        self.reward_average_priors = reward_average_priors

        # Strategy for model selection:
        # "stack": Perform separate fits for each model (traditional approach)
        # "multiarm_bandit_thompson": Use Thompson Sampling for joint selection
        # "multiarm_bandit_ucb": Use UCB for joint selection
        if model_selection_strategy not in [
            "stack",
            "multiarm_bandit_thompson",
            "multiarm_bandit_ucb",
        ]:
            raise ValueError(
                f"Invalid model_selection_strategy: {model_selection_strategy}. "
                f"Must be one of: 'stack', 'multiarm_bandit_thompson', 'multiarm_bandit_ucb'"
            )
        self.model_selection_strategy = model_selection_strategy

    def init_tree(self) -> ABMCTSAAlgoState:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        state_manager = ABMCTSAStateManager[StateT](
            default_prior_config=self.prior_config,
            default_reward_average_priors=self.reward_average_priors,
            default_model_selection_strategy=self.model_selection_strategy,
        )
        return ABMCTSAAlgoState(tree=tree, thompson_states=state_manager)

    def step(
        self,
        state: ABMCTSAAlgoState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> ABMCTSAAlgoState:
        """
        Perform one step of the Thompson Sampling MCTS algorithm.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        # initialize all_rewards_store
        if len(state.all_rewards_store) == 0:
            for action in generate_fn:
                state.all_rewards_store[action] = []

        # If the tree is empty (only root), expand the root
        if not state.tree.root.children:
            self._expand_node(state, state.tree.root, generate_fn)
            return state

        # Run one simulation step
        node = state.tree.root

        # Selection phase: traverse tree until we reach a leaf node or need to create a new node
        while node.children:
            node, action_used = self._select_child(state, node, generate_fn)

            # If action is not None, it means we've generated a new node
            if action_used is not None:
                return state

        # Expansion phase: expand leaf node
        self._expand_node(state, node, generate_fn)

        return state

    def _select_child(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, Optional[str]]:
        """
        Select a child node using Thompson Sampling.

        Args:
            state: Current algorithm state
            node: Node to select child from
            generate_fn: Mapping of action names to generation functions

        Returns:
            Tuple of (selected node, action if new node was generated)
        """
        # Get or create thompson state for this node
        thompson_state = state.thompson_states.get_or_create(
            node,
            list(generate_fn.keys()),
        )

        # Ask for next node or action using Thompson Sampling
        selection = thompson_state.select_next(state.all_rewards_store)

        # If string returned, we need to generate a new node with that action
        if isinstance(selection, str):
            new_node = self._generate_new_child(state, node, generate_fn, selection)
            return new_node, selection
        else:
            # Otherwise, we return the existing child with that index
            if selection >= len(node.children):
                raise RuntimeError(
                    f"Something went wrong in ABMCTSA algorithm: selected index {selection} is out of bounds."
                )
            return node.children[selection], None

    def _expand_node(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node, str]:
        """
        Expand a leaf node by generating a new child.

        Args:
            state: Current algorithm state
            node: Node to expand
            generate_fn: Mapping of action names to generation functions

        Returns:
            Tuple of (new node, action used)
        """
        # Create thompson state for this node if it doesn't exist
        thompson_state = state.thompson_states.get_or_create(
            node,
            list(generate_fn.keys()),
        )

        # Get action to use for generating child
        selection = thompson_state.select_next(state.all_rewards_store)

        # Ensure we get a string action name, not an index
        if not isinstance(selection, str):
            raise RuntimeError(
                f"Something went wrong in ABMCTSA algorithm: selection should always be str when the expansion is from the leaf node, whle got {selection}"
            )

        new_node = self._generate_new_child(state, node, generate_fn, selection)
        return new_node, selection

    def _generate_new_child(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        action: str,
    ) -> Node:
        """
        Generate a new child node using the specified action.

        Args:
            state: Current algorithm state
            node: Parent node
            generate_fn: Mapping of action names to generation functions
            action: Name of action to use for generation

        Returns:
            Newly created node
        """
        # Generate new state and score using the selected action
        node_state = None if node.is_root() else node.state
        new_state, new_score = generate_fn[action](node_state)

        # Add new node to the tree
        new_node = state.tree.add_node((new_state, new_score), node)

        # Update Thompson state with the new node
        thompson_state = state.thompson_states.get(node)
        if thompson_state:
            thompson_state.register_new_child_node(
                action, new_node, self.model_selection_strategy
            )
        else:
            raise RuntimeError(
                f"Internal Error in ABMCTSA: thompson_state should not be None for node {node}"
            )

        # Backpropagate the score through the parents
        self._backpropagate(state, new_node, new_score, action)

        return new_node

    def _backpropagate(
        self, state: ABMCTSAAlgoState, node: Node, score: float, action: str
    ) -> None:
        """
        Update Thompson Sampling statistics for all nodes in the path from node to root.

        Args:
            state: Current algorithm state
            node: Leaf node to start backpropagation from
            score: Score to backpropagate
            action: The action which have generated node
        """
        # Update all_rewards_score for multiarm bandit use
        state.all_rewards_store[action].append(score)

        # NOTE: For the newly created node, we always update the score for GEN node
        assert node.parent is not None
        thompson_state = state.thompson_states.get(node.parent)
        if thompson_state is None:
            raise RuntimeError(
                "Internal Error in ABMCTSA: ThompsonState should have been already initialized"
            )
        thompson_state.update_action_reward(action=action, reward=score)

        current: Optional[Node] = node.parent
        while current is not None and current.parent is not None:
            thompson_state = state.thompson_states.get(current.parent)
            if thompson_state is None:
                raise RuntimeError(
                    "Internal Error in ABMCTSA: ThompsonState should have been already initialized"
                )

            # Update the Thompson state with the score
            thompson_state.update_node_reward(current, score)

            # Move up to the parent
            current = current.parent

    def get_state_score_pairs(
        self, state: ABMCTSAAlgoState
    ) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()
