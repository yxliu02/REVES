# -*- coding: utf-8 -*-
import json
from typing import Any, Dict, List, Optional, Tuple

from utils.apis import call_llm  # must return (text, usage)

# It is recommended to place these three templates in prompt/code_pmt.py, with the same interface as the math version
from prompt.code_pmt import (
    INITIAL_SYSTEM, INITIAL_USER_TEMPLATE,
    REPAIR_SYSTEM, REPAIR_USER_TEMPLATE,
    VERIFICATION_SYSTEM, VERIFICATION_USER_TEMPLATE,
    BINARY_FEEDBACK_INCORRECT,  # Usually a fixed string, e.g. "The previous solution is incorrect."
)

# Scorer: uses the evaluator.livecodebench.compute_score function
from evaluator.livecodebench.compute_score import compute_score


# -----------------------------
# Basic extraction
# -----------------------------
def extract_problem(item: Dict[str, Any]) -> Optional[str]:
    “””
    Extract the “user problem statement” (i.e., the code task's problem specification) from the sample as best as possible.
    Prioritize extracting the user's content from the OR1 prompt list.
    “””
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


def extract_testcases_json_str(item: Dict[str, Any]) -> Optional[str]:
    """
    Extract the ground_truth for a code task -- note: the OR1 format requires it to be a **string** (JSON string).
    We only extract this string here; during scoring, it is parsed via json.loads and passed to compute_score.
    """
    rm = item.get("reward_model", {})
    if isinstance(rm, dict) and "ground_truth" in rm:
        gt = rm["ground_truth"]
        if isinstance(gt, str) and gt.strip():
            return gt.strip()
        # Handle rare cases where dict/list is provided: convert back to string
        if isinstance(gt, (dict, list)):
            try:
                return json.dumps(gt, ensure_ascii=False)
            except Exception:
                return None

    # Fallback key names (rarely used)
    for key in ["answer", "final_answer", "result", "gold", "target", "output", "label", "gt"]:
        v = rm.get(key) if isinstance(rm, dict) else item.get(key)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                pass
    return None


# -----------------------------
# Correctness check: call compute_score
# -----------------------------
def _is_correct_code(model_output: str, testcase_json_str: str) -> bool:
    """
    Perform binary correctness judgment using compute_score.
    testcase_json_str is a string (keeping the same format as OR1 ground_truth);
    it is internally parsed into a dict and passed to compute_score.
    """
    try:
        test_cases = json.loads(testcase_json_str)
    except Exception:
        return False

    try:
        ok, _ = compute_score(
            completion=model_output,
            test_cases=test_cases,
            timeout=6,
            is_binary_reward=True,
            is_power4_reward=False,
        )
        return bool(ok)
    except Exception:
        return False


# -----------------------------
# Planner main class
# -----------------------------
class CodeReflectionPlanner:
    """
    Same as the math version:
      - Initial answer (no data written)
      - If initial answer is wrong: loop up to max_retries
          (a) Repair: based on the previous round's wrong answer + binary negative feedback "incorrect" -> new answer (write datapoint)
          (b) Verification: local judgment, **only record the prompt** (write datapoint), ground truth is ["True"] / ["False"]
        If any round is judged correct, return accumulated datapoints for this problem; if never correct, return None.

    Output datapoint structure (OR1):
      {
        "prompt": [ {"content": "<single User content>", "role": "user"} ],
        "reward_model": {"ground_truth": "<string>", "style": "rule"},
        "meta": {"type": "repair|verification", "round": r}
      }
    """

    def __init__(self, llm_config: Dict[str, Any], max_retries: int = 8):
        self.llm_config = dict(llm_config)
        self.max_retries = max_retries

    # ---------- LLM (only used for initial / repair) ----------
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

    # ---------- verification: only construct prompt, do not call LLM ----------
    @staticmethod
    def _escape_verification_template(tmpl: str) -> str:
        # Same as the math version: prevent {True/False} from being consumed by .format
        tmpl = tmpl.replace("{True/False}", "{{True/False}}")
        tmpl = tmpl.replace("{true/false}", "{{true/false}}")
        tmpl = tmpl.replace("\\boxed{True/False}", "\\boxed{{True/False}}")

        import re as _re
        def _repl(m):
            key = m.group(1).strip()
            if key in ("problem", "answer"):
                return "{" + key + "}"
            return "{{" + key + "}}"
        tmpl = _re.sub(r"\{([^{}]+)\}", _repl, tmpl)
        return tmpl

    def _verification_prompt_only(self, problem: str, answer_fulltext: str) -> str:
        try:
            safe_tmpl = self._escape_verification_template(VERIFICATION_USER_TEMPLATE)
            user = safe_tmpl.format(problem=problem, answer=answer_fulltext)
        except Exception:
            user = (
                "Problem:\n{p}\n\nAnswer:\n{a}\n\n"
                "Judge whether the answer (a Python program) passes all tests. "
                "Output the final line exactly as: VERDICT: \\boxed{True/False}"
            ).format(p=problem, a=answer_fulltext)
        _ = f"[SYSTEM]\n{VERIFICATION_SYSTEM}\n\n[USER]\n{user}"  # only construct, do not send
        return user

    # ---------- main workflow ----------
    def run(self, item: Dict[str, Any], skip_if_initial_correct: bool = True) -> Optional[List[Dict[str, Any]]]:
        """
        Returns: list of training samples produced for this problem; returns None if no correct answer appears within max_retries.
        Only processes samples with ability == "code".
        """
        if not isinstance(item, dict) or item.get("ability", "").lower() != "code":
            return None

        problem = extract_problem(item)
        testcase_json_str = extract_testcases_json_str(item)
        if not problem or not testcase_json_str:
            return None

        # 0) Initial answer
        init_out, _ = self._initial_answer(problem)
        init_ok = _is_correct_code(init_out, testcase_json_str)
        if init_ok and skip_if_initial_correct:
            return None

        records: List[Dict[str, Any]] = []
        prev_answer_full = init_out

        # 1) Error-correction and verification loop
        for r in range(1, self.max_retries + 1):
            # (a) Repair -> new answer
            repair_out, _ = self._repair(problem, prev_answer_full)

            # Write repair sample (problem ground truth: still the original ground_truth string!)
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
                    "ground_truth": testcase_json_str,  # keep as string
                    "style": "rule",
                },
                "meta": {
                    "type": "repair",
                    "round": r,
                },
            })

            # (b) Local judgment
            ok = _is_correct_code(repair_out, testcase_json_str)

            # Record verification prompt (do not call LLM); **ground truth is ["True"]/["False"] (still as string)**
            _ = self._verification_prompt_only(problem, repair_out)
            verif_truth = json.dumps(["True"] if ok else ["False"], ensure_ascii=False)

            records.append({
                "prompt": [
                    {
                        "content": self._escape_verification_template(VERIFICATION_USER_TEMPLATE).format(
                            problem=problem,
                            answer=repair_out,
                        ),
                        "role": "user",
                    }
                ],
                "reward_model": {
                    "ground_truth": verif_truth,
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
