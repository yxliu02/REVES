# planner/math_reflection_planner.py
# -*- coding: utf-8 -*-
import re
import json
from typing import Any, Dict, List, Optional, Tuple

from utils.apis import call_llm  # must return (text, usage)
from prompt.math_pmt import (
    REPAIR_SYSTEM, REPAIR_USER_TEMPLATE,
    VERIFICATION_SYSTEM_2, VERIFICATION_USER_TEMPLATE_2,
    INITIAL_SYSTEM, INITIAL_USER_TEMPLATE,
    BINARY_FEEDBACK_INCORRECT,
)

# Local formal verifier
from math_verify import parse, verify


# -----------------------------
# Basic extraction & verification
# -----------------------------
def extract_problem(item: Dict[str, Any]) -> Optional[str]:
    """Try to extract the problem statement text from the current sample."""
    pr = item.get("prompt")
    if isinstance(pr, list):
        for m in pr:
            if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
                return m["content"].strip()
        for m in pr:
            if isinstance(m.get("content"), str) and m["content"].strip():
                return m["content"].strip()

    for k in ["user_input", "instruction", "question", "problem", "query", "input", "prompt"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    rm = item.get("reward_model", {})
    if isinstance(rm, dict):
        v = rm.get("user_input")
        if isinstance(v, str) and v.strip():
            return v.strip()

    msgs = item.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
                return m["content"].strip()
        for m in msgs:
            if m.get("role") == "system" and isinstance(m.get("content"), str) and m["content"].strip():
                return m["content"].strip()

    ei = item.get("extra_info", {})
    if isinstance(ei, dict):
        for k in ["question", "problem", "user_input"]:
            v = ei.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


def extract_gold_list(item: Dict[str, Any]) -> Optional[List[str]]:
    """Extract math ground truth list from various common locations/keys. Returns list[str]; returns None on failure."""
    try:
        import numpy as np
        NP_TYPES = (np.ndarray, np.generic)
    except Exception:
        NP_TYPES = tuple()

    def _to_list(v) -> List[str]:
        if v is None:
            return []
        if NP_TYPES and isinstance(v, NP_TYPES):
            try:
                if hasattr(v, "tolist"):
                    v = v.tolist()
            except Exception:
                pass
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                if NP_TYPES and isinstance(x, NP_TYPES):
                    try:
                        x = x.item()
                    except Exception:
                        x = str(x)
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
        if NP_TYPES and isinstance(v, NP_TYPES):
            try:
                v = v.item()
            except Exception:
                v = str(v)
        s = str(v).strip()
        return [s] if s else []

    rm = item.get("reward_model", {})
    if isinstance(rm, dict) and "ground_truth" in rm:
        gt = rm["ground_truth"]
        if isinstance(gt, str) and gt.strip().startswith("[") and gt.strip().endswith("]"):
            try:
                arr = json.loads(gt)
                lst = _to_list(arr)
                if lst:
                    return lst
            except Exception:
                pass
        vals = _to_list(gt)
        if vals:
            return vals
        if isinstance(gt, dict):
            for key in ["answer", "final_answer", "result", "gold", "target", "output", "label", "gt"]:
                if key in gt and gt[key] is not None:
                    vals = _to_list(gt[key])
                    if vals:
                        return vals
            bag = []
            for _, v in gt.items():
                bag += _to_list(v)
            if bag:
                return bag

    if isinstance(rm, dict):
        for key in ["answer", "final_answer", "result", "gold", "target", "output", "label", "gt"]:
            if key in rm and rm[key] is not None:
                vals = _to_list(rm[key])
                if vals:
                    return vals

    ei = item.get("extra_info", {})
    if isinstance(ei, dict):
        for key in ["answer", "final_answer", "result", "gold", "target", "output", "label", "gt"]:
            if key in ei and ei[key] is not None:
                vals = _to_list(ei[key])
                if vals:
                    return vals

    for key in ["answer", "final_answer", "result", "gold", "target", "output", "label", "gt"]:
        if key in item and item[key] is not None:
            vals = _to_list(item[key])
            if vals:
                return vals

    return None


def math_verify_reward_function(solution_str, ground_truth):
    """Quick determination using parse+verify"""
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    try:
        parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    if len(parsed) < 2:
        return 0.0
    if parsed[1] in ground_truth:
        return 1.0
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    return 0.0


def _is_correct_math(model_output: str, gold_list: List[str]) -> bool:
    try:
        score = math_verify_reward_function(model_output, gold_list)
        return bool(score >= 0.5)
    except Exception:
        return False


# -----------------------------
# Planner
# -----------------------------
class MathReflectionPlanner:
    """
    Workflow:
      - Initial answer (no data written)
      - If initial answer is wrong: loop up to max_retries
          (a) Repair: based on the previous round's wrong answer + binary negative feedback "incorrect" -> new answer (write datapoint)
          (b) Verification: local judgment, **only record the prompt** (write datapoint), ground truth is ["True"] / ["False"]
        If any round is judged correct, return accumulated datapoints for this problem; if never correct, return None.

    Output datapoint structure:
      {
        "prompt": [ {"content": "<single User content>", "role": "user"} ],
        "reward_model": {"ground_truth": "[\"...\"]", "style": "rule"},
        "meta": {"type": "repair|verification", "round": r}
      }
    """

    def __init__(self, llm_config: Dict[str, Any], max_retries: int = 8):
        self.llm_config = dict(llm_config)
        self.max_retries = max_retries

    # ---------- LLM interaction (only used for initial / repair) ----------
    def _chat(self, system: str, user: str) -> Tuple[str, Dict[str, int]]:
        prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        out_text, usage = call_llm(**{**self.llm_config, "prompt": prompt})
        return out_text, usage

    def _initial_answer(self, problem: str) -> Tuple[str, Dict[str, int]]:
        user = INITIAL_USER_TEMPLATE.format(problem=problem)
        return self._chat(INITIAL_SYSTEM, user)

    def _repair(self, problem: str, previous_answer: str) -> Tuple[str, Dict[str, int]]:
        user = REPAIR_USER_TEMPLATE.format(
            problem=problem,
            previous_answer=previous_answer,
            binary_feedback=BINARY_FEEDBACK_INCORRECT,
        )
        return self._chat(REPAIR_SYSTEM, user)

    # ---------- verification: only construct and return prompt, do not call LLM ----------
    @staticmethod
    def _escape_verification_template(tmpl: str) -> str:
        """
        Prevent {True/False} from being treated as placeholders by str.format.
        Only keep {problem} / {answer} as placeholders; double-escape all other curly braces.
        """
        tmpl = tmpl.replace("{True/False}", "{{True/False}}")
        tmpl = tmpl.replace("{true/false}", "{{true/false}}")
        tmpl = tmpl.replace("\\boxed{True/False}", "\\boxed{{True/False}}")

        def _repl(m):
            key = m.group(1).strip()
            if key in ("problem", "answer"):
                return "{" + key + "}"
            return "{{" + key + "}}"

        import re as _re
        tmpl = _re.sub(r"\{([^{}]+)\}", _repl, tmpl)
        return tmpl

    def _verification_prompt_only(self, problem: str, answer_fulltext: str) -> str:
        try:
            safe_tmpl = self._escape_verification_template(VERIFICATION_USER_TEMPLATE_2)
            user = safe_tmpl.format(problem=problem, answer=answer_fulltext)
        except Exception:
            user = (
                "Problem:\n{p}\n\nAnswer:\n{a}\n\n"
                "Judge whether the answer is correct. "
                "Output the final line exactly as: VERDICT: \\boxed{True/False}"
            ).format(p=problem, a=answer_fulltext)
        _ = f"[SYSTEM]\n{VERIFICATION_SYSTEM_2}\n\n[USER]\n{user}"  # only construct, do not send
        return user

    # ---------- main workflow ----------
    def run(self, item: Dict[str, Any], skip_if_initial_correct: bool = True) -> Optional[List[Dict[str, Any]]]:
        """
        Returns: list of training samples produced for this problem; returns None if no correct answer appears within max_retries.
        """
        if not isinstance(item, dict) or item.get("ability", "").lower() != "math":
            return None

        problem = extract_problem(item)
        gold_list = extract_gold_list(item)
        if not problem or not gold_list:
            return None

        # 0) Initial answer
        init_out, _ = self._initial_answer(problem)
        init_ok = _is_correct_math(init_out, gold_list)
        if init_ok and skip_if_initial_correct:
            return None

        records: List[Dict[str, Any]] = []
        prev_answer_full = init_out

        # 1) Error-correction and verification loop
        for r in range(1, self.max_retries + 1):
            # (a) Repair -> new answer
            repair_out, _ = self._repair(problem, prev_answer_full)

            # Write repair sample (problem ground truth)
            records.append({
                "prompt": [
                    {
                        "content": REPAIR_USER_TEMPLATE.format(
                            problem=problem,
                            previous_answer=prev_answer_full,
                            binary_feedback=BINARY_FEEDBACK_INCORRECT,
                        ),
                        "role": "user",
                    }
                ],
                "reward_model": {
                    "ground_truth": json.dumps(gold_list, ensure_ascii=False),
                    "style": "rule",
                },
                "meta": {
                    "type": "repair",
                    "round": r,
                },
            })

            # (b) Local judgment
            ok = _is_correct_math(repair_out, gold_list)

            # Record verification prompt (do not call LLM); **ground truth is True/False**
            _ = self._verification_prompt_only(problem, repair_out)
            verif_truth = json.dumps(["True"] if ok else ["False"], ensure_ascii=False)

            records.append({
                "prompt": [
                    {
                        "content": self._escape_verification_template(VERIFICATION_USER_TEMPLATE_2).format(
                            problem=problem,
                            answer=repair_out,
                        ),
                        "role": "user",
                    }
                ],
                "reward_model": {
                    "ground_truth": verif_truth,   # ["True"] or ["False"] (as string)
                    "style": "rule",
                },
                "meta": {
                    "type": "verification",
                    "round": r,
                },
            })

            if ok:
                return records

            prev_answer_full = repair_out

        return None
