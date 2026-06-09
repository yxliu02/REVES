import sys
import os
import json
import random
from typing import Dict, Any, Optional, List, Tuple

from utils.apis import call_llm
from utils.prompt_utils import (
    build_prompt,
    issues_from_eval,
    add_one_reflection,
)
from utils.utils import task_name2extract_func, task_name2eval_func
from planner.vanilla_planner import Planner


class Global_planner(Planner):
    """
    Budget-aware global planner with *batch width first*, then *reflection*, plus early-stop.

    - Two Beta posteriors (k=0 width/no-history, k=1 reflection):
        p0 ~ Beta(a0,b0), p1 ~ Beta(a1,b1)
    - Start of run: Thompson-sample p0, p1 → compute W via exp = p0/(p0+p1), W=round(B**exp) in [1,B].
      Reflection budget = B - W (number of reflection samples allowed in this run).
    - Stage 1 (Width): generate **W** initial candidates in one call, then **evaluate in order**.
        * Early stop as soon as a perfect (score==0.0) appears while iterating.
        * Beta update on success:
            - If this is the **very first overall sample** (index==1 of width list and no prior samples), do **no update**.
            - Otherwise if success is in width stage → reinforce p0 only: a0 += 1.
    - Stage 2 (Reflection): if not solved and we still have budget, do **round-robin** reflection over unsolved
      width candidates, one reflection per step, evaluating immediately after each reflection.
        * Early stop on the first perfect during reflection.
        * Beta update on reflection success: a1 += 1 (no failure updates).
    - Returns (final_raw, usage, attempt_count) where attempt_count counts *samples generated* this run:
      W (all width samples were generated at once) + number_of_reflection_samples actually produced.
    """

    def __init__(
        self,
        task_name: str,
        prev_response: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        budget: int = 64,
        multiple_reflection: bool = False,
        beta_prior: Optional[Dict[int, Tuple[float, float]]] = None,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(task_name=task_name, prev_response=prev_response, llm_config=llm_config or {})
        self.budget = max(1, int(budget))
        self.multiple_reflection = multiple_reflection
        self.extract_obj_from_raw = task_name2extract_func(self.task_name)
        self.eval_obj = task_name2eval_func(self.task_name)
        self.usage = {"input_token": 0, "output_token": 0}
        if beta_prior is None:
            self.beta = {0: [1.0, 1.0], 1: [1.0, 1.0]}
        else:
            self.beta = {
                0: [float(beta_prior.get(0, (1.0, 1.0))[0]), float(beta_prior.get(0, (1.0, 1.0))[1])],
                1: [float(beta_prior.get(1, (1.0, 1.0))[0]), float(beta_prior.get(1, (1.0, 1.0))[1])],
            }
        self._rng = random.Random(rng_seed)

    # ------------------------- helpers -------------------------

    def _sample_W(self, B: int) -> Tuple[int, float, float, float]:
        print(self.beta[0], self.beta[1])
        a0, b0 = self.beta[0]
        a1, b1 = self.beta[1]
        a0 = max(a0, 1e-12); b0 = max(b0, 1e-12)
        a1 = max(a1, 1e-12); b1 = max(b1, 1e-12)
        p0 = self._rng.betavariate(a0, b0)
        p1 = self._rng.betavariate(a1, b1)
        denom = p0 + p1
        exp = 0.5 if denom <= 1e-12 else max(0.0, min(1.0, p0 / denom))
        W = int(round(B ** exp))
        W = max(1, min(B, W))
        return W, p0, p1, exp

    def _eval_raw(self, query: Dict[str, Any], raw: str) -> Tuple[float, Any, Dict[str, Any]]:
        obj = self.extract_obj_from_raw(raw)
        res = self.eval_obj(query, obj)
        sc = float(res.get("score", 0.0))
        return sc, obj, res

    def _reflect_once(
        self,
        query: Dict[str, Any],
        curr_obj: Any,
        curr_res: Dict[str, Any],
        history: str,
        round_idx: int,
    ) -> Tuple[str, float, Any, Dict[str, Any], str]:
        issues = issues_from_eval(self.task_name, curr_res)
        if self.multiple_reflection:
            reflection = add_one_reflection(
                index=round_idx + 1,
                task_name=self.task_name,
                obj=curr_obj,
                issues=issues,
            )
            history += reflection
            history_for_prompt = history
        else:
            reflection = add_one_reflection(
                index=1,
                task_name=self.task_name,
                obj=curr_obj,
                issues=issues,
            )
            history_for_prompt = reflection

        prompt = build_prompt(task_name=self.task_name, prev_response=history_for_prompt, query=query)
        raw_new, single_usage = call_llm(**{**self.llm_config, "num_responses": 1, "prompt": prompt})
        self.usage["input_token"] += single_usage.get("input_token", 0)
        self.usage["output_token"] += single_usage.get("output_token", 0)
        assert isinstance(raw_new, str)
        sc_new, obj_new, res_new = self._eval_raw(query, raw_new)
        return raw_new, sc_new, obj_new, res_new, history

    # --------------------------- main ---------------------------

    def run(self, query: Dict, reference_info: Optional[str] = None):
        # Normalize TravelPlanner structure
        if self.task_name == "TravelPlanner":
            if isinstance(query, str):
                query = eval(query)
            if isinstance(query.get("local_constraint"), str):
                query["local_constraint"] = eval(query["local_constraint"])

        B = int(self.budget)
        W, p0_s, p1_s, exp = self._sample_W(B)
        reflection_budget = max(0, B - W)

        best_raw_global: Optional[str] = None
        best_score_global = -500.0
        attempt_count = 0  # count *samples* generated in this run

        # ---------- Stage 1: width batch ----------
        prompt0 = build_prompt(task_name=self.task_name, prev_response="", query=query)
        raws, single_usage = call_llm(**{**self.llm_config, "num_responses": W, "prompt": prompt0})
        self.usage["input_token"] += single_usage.get("input_token", 0)
        self.usage["output_token"] += single_usage.get("output_token", 0)
        assert isinstance(raws, list) and len(raws) == W

        queue: List[Dict[str, Any]] = []  # for reflection: unfinished candidates
        solved_raw = None
        earliest_width_hit_idx = None

        for i, raw in enumerate(raws, start=1):
            sc, obj, res = self._eval_raw(query, raw)
            attempt_count += 1  # even though generated in one call, count sample-wise
            if sc > best_score_global:
                best_score_global = sc
                best_raw_global = raw
            if sc == 0.0:
                solved_raw = raw
                earliest_width_hit_idx = i
                break
            # not solved → enqueue for reflection
            queue.append({
                "raw": raw,
                "score": sc,
                "obj": obj,
                "res": res,
                "history": "",
                "done": False,
            })

        if solved_raw is not None:
            # Beta update logic for width success
            if earliest_width_hit_idx != 1:  # first-ever sample hit → no update
                a0, b0 = self.beta[0]
                self.beta[0] = [a0 + 1.0, b0]
            return solved_raw, self.usage, attempt_count

        # ---------- Stage 2: reflection (round-robin until budget exhausted) ----------
        q_ptr = 0
        while reflection_budget > 0 and queue:
            # find next unfinished candidate
            n = len(queue)
            found = -1
            for _ in range(n):
                idx = q_ptr % n
                if not queue[idx]["done"]:
                    found = idx
                    break
                q_ptr += 1
            if found == -1:
                break  # all done (no unfinished)

            st = queue[found]
            raw_new, sc_new, obj_new, res_new, hist_new = self._reflect_once(
                query=query,
                curr_obj=st["obj"],
                curr_res=st["res"],
                history=st["history"],
                round_idx=0,
            )
            attempt_count += 1
            reflection_budget -= 1

            st.update({
                "raw": raw_new,
                "score": sc_new,
                "obj": obj_new,
                "res": res_new,
                "history": hist_new,
                "done": (sc_new == 0.0),
            })
            q_ptr = (found + 1) % max(1, len(queue))

            if sc_new > best_score_global:
                best_score_global = sc_new
                best_raw_global = raw_new

            if sc_new == 0.0:
                # reflection success → reinforce p1
                a1, b1 = self.beta[1]
                self.beta[1] = [a1 + 1.0, b1]
                return raw_new, self.usage, attempt_count

        # No perfect found → return best seen
        return best_raw_global, self.usage, attempt_count
