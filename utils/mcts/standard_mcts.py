import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, log, sqrt
from random import shuffle
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

from utils.base import Algorithm
from utils.tree import Node, Tree
from utils.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")


def softmax(values: List[float]) -> List[float]:
    """
    Compute softmax values for a list of scores.

    Args:
        values: List of scores

    Returns:
        List of softmax probabilities
    """
    # Shift values for numerical stability (prevent overflow)
    shifted = [x - max(values) for x in values]
    exp_values = [exp(x) for x in shifted]
    sum_exp = sum(exp_values)
    return [x / sum_exp for x in exp_values]


@dataclass
class MCTSState(Generic[StateT]):
    """State for Monte Carlo Tree Search algorithm."""

    tree: Tree[StateT]
    visit_counts: Dict[int, int] = field(default_factory=dict)
    value_sums: Dict[int, float] = field(default_factory=dict)
    priors: Dict[int, float] = field(default_factory=dict)
    next_nodes: List[Tuple[Node[StateT], str]] = field(default_factory=list)


class StandardMCTS(Algorithm[StateT, MCTSState[StateT]]):
    """
    Standard Monte Carlo Tree Search (MCTS) algorithm with UCT scoring.

    This implementation uses the Upper Confidence Bound for Trees (UCT)
    formula to balance exploration and exploitation.
    """

    def __init__(
        self, *, samples_per_action: int = 2, exploration_weight: float = sqrt(2)
    ):
        """
        Initialize the MCTS algorithm.

        Args:
            samples_per_action: Number of samples to generate for each action
            exploration_weight: Weight for the exploration term in UCT formula
        """
        self.samples_per_action = samples_per_action
        self.exploration_weight = exploration_weight

    def step(
        self,
        state: MCTSState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> MCTSState:
        """
        Perform one step of the MCTS algorithm.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        # If no nodes are queued for expansion, select nodes to expand
        if not state.next_nodes:
            # Selection: Find the most promising node to expand
            node = self._select(state)

            # Create pairs of (node, action) for all actions
            pairs_to_add = []
            actions = list(generate_fn.keys())

            # For each sample, add all actions
            for _ in range(self.samples_per_action):
                for action in actions:
                    pairs_to_add.append((node, action))

            # Shuffle the pairs to add randomness to expansion order
            shuffle(pairs_to_add)
            state.next_nodes.extend(pairs_to_add)

        # Get the next node and action to expand
        node, action = state.next_nodes.pop(0)

        # Simulation: Generate a new state using the selected action
        new_state, new_score = generate_fn[action](node.state)

        # Add the new node to the tree
        new_node = state.tree.add_node((new_state, new_score), node)

        # Update statistics for the new node
        node_id = new_node.expand_idx
        state.visit_counts[node_id] = 1
        state.value_sums[node_id] = new_score

        # Backpropagation: Update statistics for all nodes in the path
        self._backpropagate(state, new_node, new_score)

        # Update priors if this node has siblings
        parent = new_node.parent
        if parent and len(parent.children) > 1:
            self._update_priors(state, parent)

        return state

    def _update_priors(self, state: MCTSState, parent: Node) -> None:
        """
        Update prior probabilities for all children of a node using softmax.

        Args:
            state: Current algorithm state
            parent: Parent node whose children's priors will be updated
        """
        children = parent.children
        scores = [child.score for child in children]
        priors = softmax(scores)

        for child, prior in zip(children, priors):
            state.priors[child.expand_idx] = prior

    def _select(self, state: MCTSState) -> Node:
        """
        Select a node to expand using UCT.

        Starts from the root and selects child nodes with highest UCT score
        until reaching a leaf node or a node with unexplored actions.

        Args:
            state: Current algorithm state

        Returns:
            Selected node
        """
        node = state.tree.root

        # If the tree is empty, return the root
        if not node.children:
            return node

        # Traverse down the tree selecting best child according to UCT
        while node.children:
            # We're selecting based on the UCT score, which balances exploration and exploitation.
            best_child = max(
                node.children, key=lambda child: self._uct_score(state, child, node)
            )
            node = best_child

        return node

    def _uct_score(self, state: MCTSState, node: Node, parent: Node) -> float:
        """
        Calculate the UCT score for a node.

        UCT = prior * average_value + exploration_weight * sqrt(log(parent_visits) / node_visits)

        Args:
            state: Current algorithm state
            node: Node to calculate score for
            parent: Parent node

        Returns:
            UCT score
        """
        # Get visit counts
        parent_visits = state.visit_counts.get(parent.expand_idx, 1)
        node_visits = state.visit_counts.get(node.expand_idx, 1)

        # Get value sum
        value_sum = state.value_sums.get(node.expand_idx, 0)

        # Calculate exploitation term
        exploitation = value_sum / node_visits

        # Get prior (default to 1.0 if not set)
        prior = state.priors.get(node.expand_idx, 1.0)

        # Calculate exploration term (weighted by prior)
        exploration = (
            self.exploration_weight * prior * sqrt(log(parent_visits) / node_visits)
        )

        return exploitation + exploration

    def _backpropagate(self, state: MCTSState, node: Node, score: float) -> None:
        """
        Update statistics for all nodes in the path from node to root.

        Args:
            state: Current algorithm state
            node: Leaf node to start backpropagation from
            score: Score to backpropagate
        """
        current: Optional[Node] = node
        while current is not None:
            node_id = current.expand_idx
            state.visit_counts[node_id] = state.visit_counts.get(node_id, 0) + 1
            state.value_sums[node_id] = state.value_sums.get(node_id, 0) + score
            current = current.parent

    def init_tree(self) -> MCTSState:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return MCTSState(tree=tree)

    def get_state_score_pairs(self, state: MCTSState) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()
