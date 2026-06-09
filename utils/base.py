import abc
from collections.abc import Mapping
from typing import Generic, List, TypeVar

from .types import GenerateFnType, StateScoreType

# Type variables for node state and algorithm state
NodeStateT = TypeVar("NodeStateT")
AlgoStateT = TypeVar("AlgoStateT")


class Algorithm(Generic[NodeStateT, AlgoStateT], abc.ABC):
    """
    Algorithm base class for tree search.

    The Algorithm object itself should be stateless, other than the algorithm configuration which should be specified at object instantiation time.
    The state should be maintained and saved by the caller of `step` function.
    """

    @abc.abstractmethod
    def step(
        self,
        state: AlgoStateT,
        generate_fn: Mapping[str, GenerateFnType[NodeStateT]],
        inplace: bool = False,
    ) -> AlgoStateT:
        """
        Generate one additional node and add that to a given state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def init_tree(self) -> AlgoStateT:
        """
        Initialize the AlgoState, e.g. creating the root-only tree etc.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_score_pairs(
        self, state: AlgoStateT
    ) -> List[StateScoreType[NodeStateT]]:
        """
        Get all the state-score pairs of the tree.
        """
        raise NotImplementedError()
