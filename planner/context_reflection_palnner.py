import sys
import os
import json, ast
from typing import Dict, Any, Optional, List
import re
from utils.apis import call_llm

from utils.utils import task_name2extract_func, task_name2eval_func
from utils.prompt_utils import build_prompt, issues_from_eval, add_one_reflection
from planner.vanilla_planner import Planner


class ContextReflectionPlanner(Planner):
    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        context_window: int = -1,  # -1 means include all history
        max_rounds: int = 5,
    ):
        """
        Args:
            context_window: Reflection context window size. -1 means include all history; positive integer k means include only the most recent k segments.
        """
        super().__init__(task_name=task_name, prev_response=prev_response, llm_config=llm_config)
        self.max_rounds = max_rounds
        # Only keep context_window; -1 means include all, otherwise clamp to >= 1
        self.context_window = -1 if int(context_window) == -1 else max(1, int(context_window))

        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = task_name2eval_func(self.task_name)
        self.usage = {'input_token': 0, 'output_token': 0}

        # History of reflection blocks, each element is a concatenable string segment
        self.history: List[str] = []
        if prev_response:
            # Treat the initial prev_response as the first history segment
            self.history.append(prev_response)
        self.prev_response = "".join(self._window_slice()) if self.history else ""

    # ===== Utility methods =====
    def _window_slice(self) -> List[str]:
        """Return the list of historical reflection blocks that should be included in the prompt."""
        return self.history if self.context_window == -1 else self.history[-self.context_window:]

    def _windowed_prev_response(self) -> str:
        """Concatenate the historical reflection blocks within the window into a string."""
        return "".join(self._window_slice()) if self.history else self.prev_response

    # ===== Logic methods =====
    def update_history(self, i: int, obj: Any, res: Dict[str, Any]) -> None:
        """
        Generate a new reflection based on this round's evaluation results, and update the history and windowed string.
        """
        issues = issues_from_eval(self.task_name, res)

        # Use round number i+1
        reflection = add_one_reflection(
            index=i + 1,
            task_name=self.task_name,
            obj=obj,
            issues=issues,
        )

        # Always append new reflection to history; windowing only takes effect when constructing the prompt
        self.history.append(reflection)

        # Synchronize a “post-window” string for backward compatibility with old logic
        self.prev_response = "".join(self._window_slice())

    def run(self, query: Dict, reference_info: Optional[str] = None):
        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = eval(query)
            if isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = eval(query["local_constraint"])

        best_raw = None
        best_score = -500.0

        for i in range(self.max_rounds):
            prompt = build_prompt(
                task_name=self.task_name,
                prev_response=self._windowed_prev_response(),  # only include the last k segments or all
                query=query
            )

            raw, single_usage = call_llm(**{**self.llm_config, "prompt": prompt})
            self.usage['input_token'] += single_usage.get('input_token', 0)
            self.usage['output_token'] += single_usage.get('output_token', 0)

            obj = self.extract_obj_from_raw(raw)
            res = self.eval_obj(query, obj)

            if res.get("score", 0.0) == 0.0:
                return raw, self.usage, i + 1

            if res.get("score", -500.0) > best_score:
                best_raw, best_score = raw, res["score"]

            self.update_history(i, obj, res)

        return best_raw, self.usage, self.max_rounds


if __name__ == "__main__":
    from datasets import load_dataset
    import os, sys

    data = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    first = data[-1]
    reference_information = first['reference_information']
    llm_cfg = {
        "model_type": "dsv3",
        "model_name": "deepseek-v3",
        "temperature": 0.0,
        "max_tokens": 4096
    }
    # Example: only include the most recent 2 history segments; to include all, set context_window=-1
    planner = ReflectionPlanner(task_name='TravelPlanner', llm_config=llm_cfg, max_rounds=3, context_window=2)
    result = planner.run(query=first, reference_info=reference_information)
    # print("Generated Plan:", result)
