import re, json, os
from typing import List, Union, Literal, Optional, Dict
import ast

def extract_plan_obj(plan_text: str) -> list:
    """
    Extracts the JSON array from a markdown ```json [...]``` block in plan_text.

    - If a valid JSON array is found, returns it as a Python list.
    - If no matching block is found or if parsing fails, returns an empty list.
    """
    pattern = r'```json\s*(\[[\s\S]*?\])\s*```'
    match = re.search(pattern, plan_text)
    if not match:
        # No JSON block found, treat as empty
        return []
    array_text = match.group(1)
    try:
        return json.loads(array_text)
    except json.JSONDecodeError:
        # Invalid JSON content, return empty list
        return []


def extract_code_block(text: str) -> str:
    """Return the first fenced code block content (``` or ```python). If none, return the original text."""
    # Try ```python ...```
    m = re.search(r"```python\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Try any ``` ... ```
    m = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1)
    return text

import re

def extract_fenced_block(text: str) -> str:
    """
    Extract the content inside the last Markdown fenced block:
    ```\n ... \n```
    Accepts optional language tag (e.g., ```txt).
    If none is found, returns the original text stripped.
    """
    s = text.strip()
    # Find all fenced blocks and return the last one.
    # Robust to indentation before fences (common in LLM outputs / markdown lists).
    #
    # Example accepted:
    #   ```txt
    #   Q _ _ _
    #   _ Q _ _
    #   ```
    #
    # Also accepts:
    #     ```            (indented fences)
    #     Q _ _ _
    #     ```
    last_content: Optional[str] = None
    pat = re.compile(r"(?ms)^\s*```[^\n]*\n([\s\S]*?)^\s*```[ \t]*$")
    for m in pat.finditer(s):
        last_content = m.group(1)
    if last_content is not None:
        return last_content.strip("\n")
    return s


def extract_n_queens_board(text: str) -> str:
    """
    For n_queens: return the board inside a fenced block, e.g.
    ```
    Q _ _ _
    _ Q _ _
    _ _ Q _
    _ _ _ Q
    ```
    """
    s = text.strip()
    # Prefer the *last* fenced block that looks like a board (has multiple lines and contains Q/_).
    pat = re.compile(r"(?ms)^\s*```[^\n]*\n([\s\S]*?)^\s*```[ \t]*$")
    candidates = []
    for m in pat.finditer(s):
        body = m.group(1).strip("\n")
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if len(lines) >= 2 and any(("Q" in ln or "_" in ln) for ln in lines):
            candidates.append(body)
    if candidates:
        return candidates[-1]
    # Fallback: just return the last fenced block (if any), otherwise whole text.
    return extract_fenced_block(text)

def extract_mini_sudoku_board(text: str) -> str:
    """
    Extract the 4x4 grid from a code block and return it as a string
    in the same format as 'answer':
    '4 2 1 3\\n1 3 4 2\\n2 1 3 4\\n3 4 2 1'
    Returns an empty string if extraction fails.
    """
    s = text.strip()

    # Find the last ```...``` block
    blocks = re.findall(r"```(.*?)```", s, flags=re.DOTALL)
    if not blocks:
        return ""
    block = blocks[-1].strip()

    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 4:
        return ""

    # Take only the first 4 lines and normalize whitespace
    norm_lines = []
    for ln in lines[:4]:
        tokens = ln.split()
        norm_lines.append(" ".join(tokens))

    return "\n".join(norm_lines)
# def extract_countdown_answer(text: str) -> str:
#     """
#     Extract the final arithmetic expression for countdown tasks.

#     Priority:
#     1) If \\boxed{ ... } exists, return its content.
#     2) Otherwise, take the last non-empty line.
#     3) Strip quotes / code fences / 'Answer:' prefix.
#     """

#     s = text.strip()

#     # --- Case 1: extract from \boxed{ ... } ---
#     m = re.search(r"\\boxed\{([^}]*)\}", s)
#     if m:
#         expr = m.group(1).strip()
#     else:
#         # remove ``` code fences if present
#         if s.startswith("```"):
#             s = re.sub(r"^```.*?\n", "", s, flags=re.DOTALL)
#             s = s.replace("```", "").strip()

#         # drop 'Answer:' prefix if exists
#         s = re.sub(r"(?i)^answer\s*[:：]\s*", "", s).strip()

#         # take last non-empty line
#         lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
#         expr = lines[-1] if lines else ""

#     # strip outer quotes
#     expr = expr.strip(" '\"")

#     return expr

def extract_countdown_answer(text: str) -> str:
    """
    Extract the final arithmetic expression for countdown tasks.

    Priority:
    1) If \\boxed{ ... } exists, return its content.
    2) Otherwise, take the last non-empty line.
    3) Strip quotes / code fences / 'Answer:' prefix.

    If nothing can be extracted, return:
    "No answer extracted."
    """

    def _extract_braced_content(s: str, start_idx: int) -> Optional[tuple[str, int]]:
        """
        Given s[start_idx] == '{', return (content, end_idx_exclusive_after_closing_brace).
        Supports nested braces. Returns None if unbalanced.
        """
        if start_idx >= len(s) or s[start_idx] != "{":
            return None
        depth = 0
        i = start_idx
        content_start = start_idx + 1
        while i < len(s):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (s[content_start:i], i + 1)
            i += 1
        return None

    def _extract_boxed_content(s: str) -> Optional[str]:
        """
        Extract content from the first occurrence of \\boxed{...}, supporting nested braces.
        Also tolerates double backslashes in the raw output (e.g., '\\\\boxed{...}').
        """
        needles = [r"\boxed{", r"\\boxed{"]
        starts = [(s.find(n), n) for n in needles if s.find(n) != -1]
        if not starts:
            return None
        start_pos, needle = min(starts, key=lambda x: x[0])
        brace_pos = start_pos + len(needle) - 1  # points to '{'
        parsed = _extract_braced_content(s, brace_pos)
        if not parsed:
            return None
        content, _end = parsed
        return content

    s = text.strip()

    # --- Case 1: extract from \boxed{ ... } ---
    boxed = _extract_boxed_content(s)
    if boxed is not None:
        expr = boxed.strip()
    else:
        # remove ``` code fences if present
        if s.startswith("```"):
            s = re.sub(r"^```.*?\n", "", s, flags=re.DOTALL)
            s = s.replace("```", "").strip()

        # drop 'Answer:' prefix if exists
        s = re.sub(r"(?i)^answer\s*[:：]\s*", "", s).strip()

        # take last non-empty line
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        expr = lines[-1] if lines else ""

    # Strip outer quotes and remove all whitespace.
    # NOTE: Do NOT convert LaTeX operators here; evaluator handles LaTeX-to-expression normalization.
    # expr = expr.strip(" '\"")
    expr = re.sub(r"\s+", "", expr)

    # --- fallback if nothing extracted ---
    if not expr:
        return "No answer extracted."

    return expr

def load_json(path: str) -> list:
    """
    Load records from a JSON or JSONL file, and coerce embedded strings
    for 'plan' and 'local_constraint' into Python structures.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        first_char = ''
        while True:
            c = f.read(1)
            if not c or not c.isspace():
                first_char = c
                break
        f.seek(0)
        if first_char == '[':
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    # Coerce stringified fields
    for rec in records:
        # plan may be a JSON string or Python literal string
        plan = rec.get('plan')
        if isinstance(plan, str):
            try:
                rec['plan'] = json.loads(plan)
            except json.JSONDecodeError:
                rec['plan'] = ast.literal_eval(plan)
        # local_constraint may be embedded
        lc = rec.get('local_constraint')
        if isinstance(lc, str):
            try:
                rec['local_constraint'] = json.loads(lc)
            except json.JSONDecodeError:
                rec['local_constraint'] = ast.literal_eval(lc)
    return records

def save_json(data: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def task_name2extract_func(task_name: str):
    if task_name == "TravelPlanner":
        return extract_plan_obj
    # IMPORTANT: avoid `if task_name == "LiveCodeBench" or "ARC-AGI":` which is always truthy
    if task_name in ("LiveCodeBench", "ARC-AGI", "CodeContest"):
        return extract_code_block
    if task_name == "Countdown":
        return extract_countdown_answer
    if task_name == "n_queens":
        return extract_n_queens_board
    if task_name == "mini_sudoku":
        return extract_mini_sudoku_board
    if task_name == "AIME25" or task_name == "AIME24" or task_name == "MATH500":
        def f(x):
            return x
        return f
    else:
        raise(ValueError("Unsupported task_name"))


def task_name2eval_func(task_name: str):
    # Allow switching evaluator behavior via env var
    self_eval = os.getenv('SELF_EVAL', '').strip().lower() in ('1', 'true', 'yes', 'y')
    model_as_judge = os.getenv('MODEL_AS_JUDGE', '').strip().lower() in ('1', 'true', 'yes', 'y')
    full_test_cases = os.getenv('FULL_TEST_CASES', '').strip().lower() in ('1', 'true', 'yes', 'y')

    if task_name == "TravelPlanner":
        from evaluator.eval_travelplanner import evaluate_plan
        return evaluate_plan
    if task_name == "Countdown":
        from evaluator.eval_countdown import evaluate_countdown
        return evaluate_countdown
    if task_name == "n_queens":
        from evaluator.eval_n_queens import evaluate_n_queens
        return evaluate_n_queens
    if task_name == "mini_sudoku":
        from evaluator.eval_mini_sudoku import evaluate_mini_sudoku
        return evaluate_mini_sudoku
    if task_name == "LiveCodeBench" or task_name == "CodeContest":
        # Default evaluator
        if not self_eval and not model_as_judge and not full_test_cases:
            # Default behavior: public-only evaluation (score + errors are from public tests only)
            from evaluator.eval_livecodebench import eval_public_only
            return eval_public_only
        if full_test_cases:
            # score: ALL test cases (public+private); errors: PUBLIC only
            from evaluator.eval_livecodebench import eval_all_score_public_errors
            return eval_all_score_public_errors
        # Model as judge path: use Gemini as judge when MODEL_AS_JUDGE=true
        if model_as_judge:
            try:
                from evaluator.self_evaluator import llm_as_judge  # type: ignore
                # print("Using llm_as_judge evaluator")
                return llm_as_judge
            except Exception:
                raise ValueError(f"Evaluator Implementation Error")
        # Self-eval path (optional): try import an alternative evaluator when SELF_EVAL=true
        try:
            from evaluator.self_evaluator import eval_self  # type: ignore
            # print("Using self-eval evaluator")
            return eval_self
        except Exception:
            raise ValueError(f"Evaluator Implementation Error")
    if task_name == "ARC-AGI":
        from evaluator.eval_arcagi import eval_arc_train_only
        return eval_arc_train_only
    if task_name == "AIME25" or task_name == "AIME24":
        if not self_eval and not model_as_judge:
            from math_verify import verify, parse
            def math_equal(query, answer: str):
                final_answer = parse(answer) 
                gold = parse(query['reward_model']['ground_truth'])
                correctness = verify(gold, final_answer)
                score = 0 if correctness else -1
                return {"score": score}
            return math_equal
        # Model as judge path: use Gemini as judge when MODEL_AS_JUDGE=true
        if model_as_judge:
            try:
                from evaluator.self_evaluator import llm_as_judge  # type: ignore
                # print("Using llm_as_judge evaluator")
                return llm_as_judge
            except Exception:
                raise ValueError(f"Evaluator Implementation Error")
        try:
            from evaluator.self_evaluator import eval_self  # type: ignore
            # print("Using self-eval evaluator")
            return eval_self
        except Exception:
            raise ValueError(f"Evaluator Implementation Error")
    if task_name == "MATH500":
        if not self_eval and not model_as_judge:
            # NOTE: do not name local modules `math.py` (shadows stdlib `math` and breaks numpy/pandas)
            from evaluator.math_eval import compute_score
            def math500_equal(query, answer: str):
                ground_truth = query['reward_model']['ground_truth']
                raw_score = compute_score(answer, ground_truth)
                score = 0 if raw_score == 1 else -1
                return {"score": score}
            return math500_equal
        if model_as_judge:
            from evaluator.self_evaluator import llm_as_judge  # type: ignore
            return llm_as_judge
        try:
            from evaluator.self_evaluator import eval_self  # type: ignore
            return eval_self
        except Exception:
            raise ValueError(f"Evaluator Implementation Error")