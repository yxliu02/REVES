import sys
import os
import json, ast
from typing import Dict, Any, Optional, List
import re
from utils.apis import call_llm
from utils.prompt_utils import build_prompt
from utils.utils import task_name2extract_func, task_name2eval_func
from planner.vanilla_planner import Planner
from math import ceil

class BoNPlanner(Planner):
    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        n: int = 5,
        num_per_generation: int = 5
    ):
        llm_config = {**(llm_config or {}), "num_responses": num_per_generation}
        super().__init__(task_name=task_name, prev_response=prev_response, llm_config=llm_config)
        self.n = n
        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = task_name2eval_func(self.task_name)
        self.usage = {'input_token': 0, 'output_token': 0}

    def run(self, query: Dict) -> str:

        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = eval(query)
            if isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = eval(query["local_constraint"])
        
        num_batches = ceil(self.n / self.llm_config["num_responses"])

        best_raw = None
        best_score = -500.0

        for i in range(num_batches):
            curr_num_responses = min(self.llm_config["num_responses"], self.n - i * self.llm_config["num_responses"])
            llm_temp_config = {**self.llm_config, "num_responses": curr_num_responses}

            assert self.prev_response == ""

            prompt = build_prompt(
                task_name=self.task_name,
                prev_response=self.prev_response,
                query=query
            )
            raws, single_usage = call_llm(**{**llm_temp_config, "prompt": prompt})
            self.usage['input_token'] += single_usage['input_token']
            self.usage['output_token'] += single_usage['output_token']
            if curr_num_responses > 1:
                assert isinstance(raws, list)
                for raw in raws:
                    obj = self.extract_obj_from_raw(raw)
                    res = self.eval_obj(query, obj)
                    if res["score"] == 0.0:
                        return raw, self.usage, i+1
                    if res["score"] > best_score:
                        best_raw, best_score = raw, res["score"]

            elif curr_num_responses == 1:
                assert isinstance(raws, str)
                obj = self.extract_obj_from_raw(raws)
                res = self.eval_obj(query, obj)
                if res["score"] == 0.0:
                    return raws, self.usage, i+1
                if res["score"] > best_score:
                    best_raw, best_score = raws, res["score"]
            
            else:
                raise ValueError("Error in processing data")

        return best_raw, self.usage, self.n

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
    planner = BoNPlanner(task_name='TravelPlanner', llm_config=llm_cfg, n=2, num_per_generation=2)
    result = planner.run(query=first, reference_info=reference_information)
    print("Generated Plan:", result)