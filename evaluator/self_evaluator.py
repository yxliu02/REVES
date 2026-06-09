import os
import re
from typing import Any, Dict, Union
from utils.apis import call_llm

# Hardcoded LLM config for self-evaluation
# Prefer environment variables set by bash; fall back to sane defaults
_env = os.getenv
def _get_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default

def _get_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default

# Allow both SELF_EVAL_* and common names (MODEL_NAME, TEMPERATURE, MAX_TOKENS, MODEL_TYPE)
_model_type = _env("SELF_EVAL_MODEL_TYPE", _env("MODEL_TYPE", "vllm"))
_model_name = _env("SELF_EVAL_MODEL_NAME", _env("MODEL_NAME", "qwen-7b"))
_temperature = _get_float("SELF_EVAL_TEMPERATURE", _get_float("TEMPERATURE", 0.0))
_max_tokens = _get_int("SELF_EVAL_MAX_TOKENS", _get_int("MAX_TOKENS", 8192))

LLM_CONFIG: Dict[str, Any] = {
    "model_type": _model_type,
    "model_name": _model_name,
    "temperature": _temperature,
    "max_tokens": _max_tokens,
}

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


# def _extract_problem_text(query: Union[Dict[str, Any], str, Any]) -> tuple[str, Any]:
#     """Robustly extract a human-readable problem text from various query shapes.
#     Returns (problem, ability).
#     - Supports: raw string, dict-like payload, and object inputs (e.g., CodeGenerationProblem).
#     - Never touches test cases; only composes a textual statement.
#     """
#     # Case 1: raw string already the problem
#     if isinstance(query, str):
#         return query, None

#     # Case 2: mapping/dict input
#     if isinstance(query, dict):
#         # prefer explicit problem field
#         problem = query.get("problem")
#         if not isinstance(problem, str) or not problem.strip():
#             # fallbacks commonly seen
#             for key in ("prompt", "question", "description", "text", "content", "statement", "title"):
#                 val = query.get(key)
#                 if isinstance(val, str) and val.strip():
#                     problem = val
#                     break
#         if not isinstance(problem, str):
#             problem = ""

#         ability = query.get("ability")
#         return problem, ability

#     # Case 3: object input (e.g., CodeGenerationProblem)
#     # If it looks like CodeGenerationProblem, compose a descriptive statement
#     if all(
#         _safe_getattr(query, name) is not None for name in (
#             "question_title", "question_content", "platform", "difficulty"
#         )
#     ):
#         # Build text using the same LiveCodeBench prompt concatenation logic
#         import os
#         change_prompt_flag = os.environ.get("change_prompt", "").lower() == "true"

#         try:
#             question_content = _safe_getattr(query, "question_content") or ""
#         except Exception:
#             question_content = ""
#         try:
#             starter_code = _safe_getattr(query, "starter_code") or ""
#         except Exception:
#             starter_code = ""

#         if change_prompt_flag:
#             try:
#                 from prompt.LiveCodeBench_pmt import LIVECODEBENCH_PROMPT  # type: ignore
#                 problem = LIVECODEBENCH_PROMPT.format(
#                     question_content=question_content,
#                     starter_code=starter_code,
#                     previous_plan="",
#                 )
#             except Exception:
#                 # Fallback to original concatenation if template import fails
#                 change_prompt_flag = False

#         if not change_prompt_flag:
#             system_header = (
#                 "You are an expert Python programmer. You will be given a problem specification "
#                 "and must produce a correct Python program that passes the tests."
#             )
#             prompt_lines = [system_header, "", f"### Question:\n{question_content}", ""]
#             if starter_code:
#                 prompt_lines.append("")
#                 prompt_lines.append(
#                     "### Format: You will use the following starter code and enclose your final code in triple backticks."
#                 )
#                 prompt_lines.append(f"```python\n{starter_code}\n```\n")
#             else:
#                 prompt_lines.append("")
#                 prompt_lines.append(
#                     "### Format: Read inputs from stdin and write the answer to stdout. Enclose your final code in triple backticks."
#                 )
#                 prompt_lines.append("```python\n# YOUR CODE HERE\n```\n")
#             prompt_lines.append("### Answer: Provide ONLY the final Python code, enclosed in triple backticks.\n")
#             problem = "\n".join(prompt_lines)

#         meta = _safe_getattr(query, "metadata") or {}
#         ability = meta.get("ability") if isinstance(meta, dict) else None
#         return problem, ability

#     # Try common attribute names without assuming a specific class
#     attr_candidates = (
#         "problem",
#         "prompt",
#         "description",
#         "question",
#         "text",
#         "content",
#         "statement",
#         "doc",
#     )
#     problem = None
#     for name in attr_candidates:
#         val = _safe_getattr(query, name)
#         if isinstance(val, str) and val.strip():
#             problem = val
#             break

#     # Optionally try metadata-like containers
#     if not problem:
#         meta = _safe_getattr(query, "metadata") or {}
#         if isinstance(meta, dict):
#             for key in ("problem", "prompt", "description", "question", "title"):
#                 val = meta.get(key)
#                 if isinstance(val, str) and val.strip():
#                     problem = val
#                     break

#     if not isinstance(problem, str):
#         problem = ""

#     # Try to extract optional ability/expected from metadata if present
#     ability = None
#     meta = _safe_getattr(query, "metadata") or {}
#     if isinstance(meta, dict):
#         ability = meta.get("ability", ability)
#     return problem, ability


def eval_self(query: Union[Dict[str, Any], str], answer: str) -> Dict[str, int]:
    # Accept inputs shaped like eval_public_only (e.g., CodeGenerationProblem),
    # but only build a textual problem statement; do not run test cases.
    # problem, _ability = _extract_problem_text(query)

    problem=query['prompt'][0]['content']

    # prompt = (
    #     "You are a strict verifier. Read the problem and the candidate answer, and judge whether the candidate answer is correct.\n"
    #     "Do not provide extra explanation.\n"
    #     "At the end, output exactly one final line in the format: `VERDICT: \\boxed{True}` or `VERDICT: \\boxed{False}`.\n\n"
    #     f"Problem:\n```\n{problem}\n```\n\n"
    #     f"Candidate Answer:\n```\n{answer}\n```\n\n"
    #     "Final line (no extra text): VERDICT: \\boxed{True/False}"
    # )

    prompt = (
        "You are a strict verifier. Read the problem and the candidate answer, and carefully verify whether the candidate answer is correct.\n"
        "You should provide a clear and logical verification process:\n"
        "- Explain why the answer is correct, OR\n"
        "- If it is incorrect, explain where the mistake occurs and why it is wrong.\n\n"
        "After your reasoning, output exactly one final line in the format:\n"
        "VERDICT: \\boxed{True}  (if the answer is correct)\n"
        "VERDICT: \\boxed{False} (if the answer is incorrect)\n\n"
        f"Problem:\n```\n{problem}\n```\n\n"
        f"Candidate Answer:\n```\n{answer}\n```\n\n"
        "First output the logical verification process, and THEN output the final line: VERDICT: \\boxed{True/False}"
    )
    raw, _usage = call_llm(**{**LLM_CONFIG, "prompt": prompt})

    with open("self_eval_log.txt", "a") as f:
        f.write("=== New Eval ===\n")
        f.write("Prompt Sent to Judge:\n")
        f.write(prompt + "\n\n")
        f.write("LLM Raw Output:\n")
        f.write(raw + "\n")
        f.write("=================\n\n")

    m = re.search(r"VERDICT:\s*\\boxed\{(True|False)\}", raw)
    if not m:
        return {"score": -1}
    return {"score": 0 if m.group(1) == "True" else -1}


def llm_as_judge(query: Union[Dict[str, Any], str], answer: str) -> Dict[str, int]:
    """
    Use Gemini as a judge/verifier to evaluate answers.
    Input and output format same as eval_self, but uses Gemini with hardcoded parameters.
    
    Args:
        query: The problem query (same format as eval_self)
        answer: The candidate answer to evaluate
    
    Returns:
        Dict[str, int]: {"score": 0} if correct, {"score": -1} if incorrect or verdict not found
    """
    # Extract problem text using the same logic as eval_self
    # problem, _ability = _extract_problem_text(query)
    problem=query['prompt'][0]['content']
    
    # prompt = (
    #     "You are a strict verifier. Read the problem and the candidate answer, and judge whether the candidate answer is correct.\n"
    #     "Do not provide extra explanation.\n"
    #     "At the end, output exactly one final line in the format: `VERDICT: \\boxed{True}` or `VERDICT: \\boxed{False}`.\n\n"
    #     f"Problem:\n```\n{problem}\n```\n\n"
    #     f"Candidate Answer:\n```\n{answer}\n```\n\n"
    #     "Final line (no extra text): VERDICT: \\boxed{True/False}"
    # )
    prompt = (
    "You are a strict verifier.\n\n"
    f"Problem:\n```\n{problem}\n```\n\n"
    f"Candidate Answer:\n```\n{answer}\n```\n\n"
    "Output exactly ONE line and nothing else:\n"
    "VERDICT: \\boxed{True} or VERDICT: \\boxed{False}"
    )
# gemini-2.5-flash-lite
# gemini-3-flash-preview
    # Hardcoded Gemini parameters
    raw, _usage = call_llm(
        prompt=prompt,
        model_type='gemini',
        model_name='gemini-3-flash-preview',
        temperature=0.6,
        max_tokens=500,
        num_responses=1
    )
    # ---- log prompt + raw only ----
    with open("judge_log.txt", "a") as f:
        f.write("=== New Eval ===\n")
        f.write("Prompt Sent to Judge:\n")
        f.write(prompt + "\n\n")
        f.write("LLM Raw Output:\n")
        f.write(raw + "\n")
        f.write("=================\n\n")

    m = re.search(r"VERDICT:\s*\\boxed\{(True|False)\}", raw)
    if not m:
        return {"score": -1}
    return {"score": 0 if m.group(1) == "True" else -1}


VERIFY_USER_TEMPLATE = (
    "Check the math solution step-by-step. "
    "If you find a mistake: state the wrong step, explain why it's wrong, "
    "and end your response with 'The answer is wrong'. "
    "If all steps are correct, end your response with 'The answer is correct'.\n\n"
    "Problem: {problem}\n\n"
    "Answer: {answer}\n"
)

def eval_self_test(query: Union[Dict[str, Any], str], answer: str) -> Dict[str, int]:
    """
    Self-verification using natural-language reasoning + final binary verdict.

    Returns:
        {"score": 0}   if verifier says "The answer is correct"
        {"score": -1}  if verifier says "The answer is wrong" OR parsing fails
    """

    # Extract problem text
    problem = query["prompt"][0]["content"]

    # Build prompt
    prompt = VERIFY_USER_TEMPLATE.format(
        problem=problem,
        answer=answer
    )

    # Call LLM
    raw, _usage = call_llm(**{**LLM_CONFIG, "prompt": prompt})

    # Parse final verdict (ONLY trust the explicit sentence)
    if re.search(r"The answer is correct\.?$", raw.strip()):
        return {"score": 0}

    if re.search(r"The answer is wrong\.?$", raw.strip()):
        return {"score": -1}

    # Fallback: invalid / not-following-format
    return {"score": -1}