from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from utils.apis import call_llm
from utils.utils import task_name2extract_func, task_name2eval_func
from utils.mcts.standard_mcts import StandardMCTS
from utils.types import GenerateFnType

from utils.prompt_utils import (
    build_prompt,
    issues_from_eval,
    add_one_reflection,
    scale_score_fixed
)

from planner.vanilla_planner import Planner


@dataclass
class _PlanState:
    """
    Information stored in each tree node.

    prev_response : the full reflection text so far, used as `prev_response`
                when building the next prompt.
    index     : how many reflections have already been appended (root = 0).
    raw       : the LLM’s raw answer generated at this node.
    score     : the original evaluator score (before scaling).
    """
    prev_response: str
    index: int
    raw: str
    score: float


class MCTSPlanner(Planner):

    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        samples_per_action: int = 4,
        depth: int = 20,
        simulations: int = 64,
        exploration_weight: float = math.sqrt(2),
        multiple_reflection: bool = True,
    ):
        super().__init__(task_name, prev_response, llm_config)

        self.samples_per_action = samples_per_action
        self.simulations = simulations
        self.multiple_reflection = multiple_reflection
        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = task_name2eval_func(self.task_name)
        self.usage = {'input_token': 0, 'output_token': 0}

        # Instantiate the treequest algorithm.  `samples_per_action`
        # controls how many children are generated per expansion.
        self._algo = StandardMCTS(
            samples_per_action=samples_per_action,
            exploration_weight=exploration_weight,
        )

    def run(self, query: Dict[str, Any], reference_info: Optional[str] = None) -> str:
        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = ast.literal_eval(query)
            if isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = ast.literal_eval(query["local_constraint"])

        # Initialise an empty search tree
        tree_state = self._algo.init_tree()

        best_raw: Optional[str] = None
        best_score: float = -500.0

        def _generate(parent_state: Optional[_PlanState]) -> Tuple[_PlanState, float]:
            # Determine the reflection text so far
            if parent_state is None:          # root
                prev_response_txt = self.prev_response
                idx = 0
            else:                             # non-root
                prev_response_txt = parent_state.prev_response
                idx = parent_state.index

            # Build the prompt and query the LLM once
            prompt = build_prompt(
                task_name=self.task_name,
                query=query,
                prev_response=prev_response_txt
            )
            resp, single_usage = call_llm(**{**self.llm_config, "prompt": prompt, "num_responses": 1})
            self.usage['input_token'] += single_usage['input_token']
            self.usage['output_token'] += single_usage['output_token']

            raw_answer = resp if isinstance(resp, str) else resp[0]

            # Evaluate the answer
            obj = self.extract_obj_from_raw(raw_answer)
            eva = self.eval_obj(query, obj)
            score_raw = eva["score"]
            score_scaled = scale_score_fixed(task_name=self.task_name, score=score_raw)

            # Track the best answer encountered so far
            nonlocal best_raw, best_score
            if score_raw > best_score:
                best_raw = raw_answer
                best_score = score_raw

            # Build reflection block for this answer
            issues = issues_from_eval(self.task_name, eva)
            if self.multiple_reflection:
                reflection_block = add_one_reflection(
                    index=idx + 1,
                    task_name=self.task_name,
                    obj=obj,
                    issues=issues,
                )
                new_prev_response = prev_response_txt + reflection_block
            else:
                new_prev_response = add_one_reflection(
                    index=1,
                    task_name=self.task_name,
                    obj=obj,
                    issues=issues,
                )
            # print(new_prev_response)
            # Return new state plus the scaled score
            new_state = _PlanState(
                prev_response=new_prev_response,
                index=idx + 1,
                raw=raw_answer,
                score=score_raw,
            )
            return new_state, score_scaled

        # Provide a single action called "GEN"
        generate_map: Mapping[str, GenerateFnType[_PlanState]] = {"GEN": _generate}

        # ------------------- search loop -------------------
        for _ in range(self.simulations):
            # print(f"current depth {_}")
            tree_state = self._algo.step(tree_state, generate_map, inplace=True)

            # Early exit if a perfect plan is found
            if best_score == 0.0:
                return best_raw, self.usage, _ + 1                            # type: ignore[arg-type]

            # Enforce depth limit manually (StandardMCTS has no built-in depth cap)
        # Return the best plan encountered (empty string if none)
        return best_raw, self.usage, self.simulations
