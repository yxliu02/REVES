import sys
import os
import json, ast
from typing import Dict, Any, Optional, List
import re
import math
from utils.apis import call_llm_with_prob

from utils.utils import task_name2extract_func, task_name2eval_func
from utils.prompt_utils import build_prompt, issues_from_eval, add_one_reflection
from planner.vanilla_planner import Planner
from evaluator.confidence import compute_tail_confidence
from math_verify import parse, verify

def eval_obj_hack(query: Dict, obj: Any) -> Dict[str, Any]:
    return {
        "score": -1,
    }

class ReflectionPlanner(Planner):
    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        max_rounds: int = 5,
        multiple_reflection: bool = True,
        answer_key_stop_threshold: Optional[float] = 0.5,
        answer_key_stop_min_rounds: int = 3,
    ):
        if llm_config is not None:
            llm_config["logprobs_k"] = 3
        super().__init__(task_name=task_name, prev_response=prev_response, llm_config=llm_config)
        self.max_rounds = max_rounds
        self.multiple_reflection = multiple_reflection
        # Early-stop when the most common answer_key proportion reaches this threshold.
        # - None: disable early-stop (always sample max_rounds)
        # - 1.0: only stop when all parsed answers agree so far
        self.answer_key_stop_threshold = answer_key_stop_threshold
        self.answer_key_stop_min_rounds = max(1, int(answer_key_stop_min_rounds))
        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = eval_obj_hack
        self.usage = {'input_token': 0, 'output_token': 0}
        # Useful debugging info (doesn't change the return signature)
        self.selected_answer_key: Optional[str] = None
        self.selected_agg_conf: Optional[float] = None


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

        raw_response_list = []
        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = eval(query)
            if isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = eval(query["local_constraint"])
        
        best_raw = None
        best_score = -500.0

        def _answer_key(parsed_obj: Any) -> Optional[str]:
            """
            Build a stable grouping key for "same answer".
            For math_verify.parse output, parsed[1] is typically the extracted final answer.
            """
            if parsed_obj is None:
                return None
            if isinstance(parsed_obj, (list, tuple)):
                if len(parsed_obj) >= 2:
                    k = str(parsed_obj[1]).strip()
                    return k if k else None
                if len(parsed_obj) == 1:
                    k = str(parsed_obj[0]).strip()
                    return k if k else None
                return None
            if isinstance(parsed_obj, dict):
                for kk in ("answer", "final_answer", "final", "boxed", "result", "value"):
                    if kk in parsed_obj and parsed_obj[kk] is not None:
                        k = str(parsed_obj[kk]).strip()
                        return k if k else None
                try:
                    return json.dumps(parsed_obj, sort_keys=True, ensure_ascii=False, default=str)
                except Exception:
                    return str(parsed_obj).strip() or None
            k = str(parsed_obj).strip()
            return k if k else None

        for i in range(self.max_rounds):

            prompt = build_prompt(
                task_name=self.task_name,
                prev_response=self.prev_response,
                query=query
            )

            raw, single_usage = call_llm_with_prob(**{**self.llm_config, "prompt": prompt})
            self.usage['input_token'] += single_usage['input_token']
            self.usage['output_token'] += single_usage['output_token']
            obj = self.extract_obj_from_raw(raw)

            res = self.eval_obj(query, obj)
            
            conf_tail = compute_tail_confidence(single_usage, K=200, k_top=3)
            parsed_obj = parse(raw)
            raw_response_list.append((raw, conf_tail, parsed_obj))

            self.update_history(i, obj, res)

            # Early-stop: if the dominant answer_key reaches a proportion threshold, stop sampling.
            # We only count parse-able keys; un-parseable outputs shouldn't trigger early stopping.
            if i == 0:
                answer_key_counts: Dict[str, int] = {}
                parsed_key_total = 0
            key_now = _answer_key(parsed_obj)
            if key_now is not None:
                answer_key_counts[key_now] = answer_key_counts.get(key_now, 0) + 1
                parsed_key_total += 1
                if (
                    self.answer_key_stop_threshold is not None
                    and parsed_key_total >= self.answer_key_stop_min_rounds
                ):
                    top_key, top_cnt = max(answer_key_counts.items(), key=lambda kv: kv[1])
                    top_prop = top_cnt / parsed_key_total if parsed_key_total > 0 else 0.0
                    if top_prop >= float(self.answer_key_stop_threshold):
                        with open("log.txt", "a") as f:
                            f.write(
                                "early_stop(answer_key_prop): "
                                f"round={i+1}\tthreshold={self.answer_key_stop_threshold}\t"
                                f"top_key={top_key}\ttop_cnt={top_cnt}\tparsed_total={parsed_key_total}\t"
                                f"top_prop={top_prop}\n"
                            )
                        break

        def _conf_for_compare(c: Any) -> float:
            try:
                v = float(c)
            except Exception:
                return float("-inf")
            if math.isnan(v):
                return float("-inf")
            return v

        # Group by answer and aggregate confidence (sum over same answer)
        groups: Dict[str, Dict[str, Any]] = {}
        # For logging/debugging: (idx, key, conf_tail)
        candidate_summaries: List[tuple[int, str, Any]] = []
        for idx, (raw, conf_tail, parsed_obj) in enumerate(raw_response_list):
            key = _answer_key(parsed_obj)
            # Un-parseable answers should not accidentally aggregate together
            if key is None:
                key = f"__UNPARSED__:{idx}"

            candidate_summaries.append((idx, key, conf_tail))

            conf_cmp = _conf_for_compare(conf_tail)
            conf_sum = 0.0 if conf_cmp == float("-inf") else conf_cmp

            g = groups.get(key)
            if g is None:
                groups[key] = {
                    "sum_conf": conf_sum,
                    "best_conf": conf_cmp,
                    "best_raw": raw,
                    "count": 1,
                }
            else:
                g["sum_conf"] += conf_sum
                g["count"] += 1
                if conf_cmp > g["best_conf"]:
                    g["best_conf"] = conf_cmp
                    g["best_raw"] = raw

        # Choose by highest aggregated confidence; tie-break by best single confidence
        best_key = max(groups.keys(), key=lambda k: (groups[k]["sum_conf"], groups[k]["best_conf"]))
        best_raw = groups[best_key]["best_raw"]
        self.selected_answer_key = best_key
        self.selected_agg_conf = float(groups[best_key]["sum_conf"])

        with open("log.txt", "a") as f:
            f.write("candidates(answer_key, conf_tail):\n")
            for idx, key, conf_tail in candidate_summaries:
                f.write(f"  idx={idx}\tanswer_key={key}\tconf_tail={conf_tail}\n")
            f.write("aggregated_by_answer_key(sum_conf, count, best_conf):\n")
            for key, g in sorted(groups.items(), key=lambda kv: kv[1]["sum_conf"], reverse=True):
                f.write(
                    f"  answer_key={key}\tsum_conf={g['sum_conf']}\tcount={g['count']}\tbest_conf={g['best_conf']}\n"
                )
            f.write(
                f"chosen:\tanswer_key={best_key}\tsum_conf={groups[best_key]['sum_conf']}\tbest_conf={groups[best_key]['best_conf']}\n"
            )
            f.write("-" * 40 + "\n")



        rounds_used = len(raw_response_list)
        return best_raw, self.usage, rounds_used

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