from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from utils.apis import call_llm
from utils.utils import task_name2extract_func, task_name2eval_func


from utils.ab_mcts_a.algo import ABMCTSA
from utils.ab_mcts_a.prob_state import PriorConfig  # optional, for type hints
from utils.types import GenerateFnType

from utils.prompt_utils import (
    build_prompt,
    issues_from_eval,
    add_one_reflection,
    scale_score_fixed
)

from planner.vanilla_planner import Planner


# ---------- Per-node payload -------------------------------------------------
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


class AB_MCTS_APlanner(Planner):
    """
    Wrapper planner that uses ABMCTSA (AB-MCTS-A) internally.

    Semantics match your MCTSPlanner/AB_MCTS_MPlanner:
    - Accumulate reflection text along the path (prev_response grows each step).
    - Evaluate each answer; keep the best raw answer seen so far.
    - Use scaled score in [0,1] for the search algorithm.
    - Stop early if evaluator score == 0.0.
    """

    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        simulations: int = 64,
        # ABMCTSA knobs (mirror algp.py constructor)
        dist_type: str = "gaussian",   # "gaussian" | "beta"
        reward_average_priors: Optional[float | Dict[str, float]] = None,
        multiple_reflection: bool = True,
        prior_config: Optional[PriorConfig] = None,
        model_selection_strategy: str = "multiarm_bandit_thompson",  # "stack" | "multiarm_bandit_thompson" | "multiarm_bandit_ucb"
    ):
        super().__init__(task_name, prev_response, llm_config)

        self.simulations = simulations
        self.multiple_reflection = multiple_reflection
        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = task_name2eval_func(self.task_name)
        self.usage = {'input_token': 0, 'output_token': 0}


        # Instantiate ABMCTSA with your settings
        self._algo = ABMCTSA(
            dist_type=dist_type,
            reward_average_priors=reward_average_priors,
            prior_config=prior_config,
            model_selection_strategy=model_selection_strategy,
        )


    def run(self, query: Dict[str, Any], reference_info: Optional[str] = None) -> str:
        # Accept query also as a string
        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = ast.literal_eval(query)
            if isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = ast.literal_eval(query["local_constraint"])

        # Initialise an empty search tree
        tree_state = self._algo.init_tree()

        best_raw: Optional[str] = None
        best_score: float = -500.0

        # -------- generator callback: parent_state -> (child_state, scaled_score) --------
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

            # Track the best answer encountered so far (by raw score)
            nonlocal best_raw, best_score
            if score_raw > best_score:
                best_raw = raw_answer
                best_score = score_raw

            # Build reflection block for this answer (append to history)
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

        # Provide action(s) for ABMCTSA.
        # Single action works; ABMCTSA will still choose between existing children vs. GEN (new child),
        # and (if multiple actions are provided) between different actions via Thompson/UCB.
        generate_map: Mapping[str, GenerateFnType[_PlanState]] = {"GEN": _generate}

        # ------------------- search loop -------------------
        for _ in range(self.simulations):
            # print(f"current depth:{_}")
            tree_state = self._algo.step(tree_state, generate_map, inplace=True)

            # Early exit if a perfect plan is found
            if best_score == 0.0:
                return best_raw or "", self.usage, _ + 1

        # Return the best plan encountered
        return best_raw or "", self.usage, self.simulations
