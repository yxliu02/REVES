import sys
import os
import json, ast
from typing import Dict, Any, Optional, List
import re
from utils.apis import call_llm

from utils.utils import task_name2extract_func, task_name2eval_func
from utils.prompt_utils import build_prompt, issues_from_eval, add_one_reflection
from planner.vanilla_planner import Planner


class ReflectionPlanner(Planner):
    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        max_rounds: int = 5,
        multiple_reflection: bool = True
    ):
        super().__init__(task_name=task_name, prev_response=prev_response, llm_config=llm_config)
        self.max_rounds = max_rounds
        self.multiple_reflection = multiple_reflection
        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = task_name2eval_func(self.task_name)
        self.usage = {'input_token': 0, 'output_token': 0}


    def update_history(self, i: int, obj: Any, res: Dict[str, Any]) -> None:
        """
        Update self.prev_response by appending a new reflection block.
        """
        issues = issues_from_eval(self.task_name, res)
        
        if self.multiple_reflection:
            reflection = add_one_reflection(
                index=i + 1,
                task_name=self.task_name,
                obj=obj,
                issues=issues,
            )
            self.prev_response += reflection
        else:
            reflection = add_one_reflection(
                index=1,
                task_name=self.task_name,
                obj=obj,
                issues=issues,
            )
            self.prev_response = reflection

    def run(self, query: Dict, reference_info: Optional[str] = None) -> str:        

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
                prev_response=self.prev_response,
                query=query
            )
            
            raw, single_usage = call_llm(**{**self.llm_config, "prompt": prompt})
            self.usage['input_token'] += single_usage['input_token']
            self.usage['output_token'] += single_usage['output_token']
            obj = self.extract_obj_from_raw(raw)
            # with open("log_NEW.txt", "a") as f:
            #     f.write(f"prompt: {prompt}\n")
            #     f.write("-" * 40 + "\n")
            #     f.write(f"obj: {obj}\n")
            #     f.write("-" * 40 + "\n")
            res = self.eval_obj(query, obj)
            
            if res["score"] == 0.0:
                return raw, self.usage, i+1
            if res["score"] > best_score:
                best_raw, best_score = raw, res["score"]

            self.update_history(i, obj, res)

        return best_raw, self.usage, self.max_rounds

if __name__ == "__main__":
    from datasets import load_dataset
    import os, sys

    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # sys.path.insert(0, base_dir)

    data = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    first = data[-1]
    reference_information = first['reference_information']
    llm_cfg = {
        "model_type": "dsv3",
        "model_name": "deepseek-v3",
        "temperature": 0.0,
        "max_tokens": 4096
    }
    planner = ReflectionPlanner(task_name='TravelPlanner', llm_config=llm_cfg, max_rounds=3)
    result = planner.run(query=first, reference_info=reference_information)
    # print("Generated Plan:", result)