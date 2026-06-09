# prompt/ARC_AGI_pmt.py
from __future__ import annotations
from typing import Any, Dict, List

_COLOR_MAP = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "pink", 7: "orange", 8: "purple", 9: "brown",
}

def list_format(grid: List[List[int]]) -> str:
    rows = []
    for r in grid:
        rows.append("[" + ", ".join(str(x) for x in r) + "]")
    return "[\n" + "\n".join(rows) + "\n]"

def _task_explanation() -> str:
    return (
        "You will be given some number of paired example inputs and outputs. "
        "The outputs were produced by applying a transformation rule to the inputs. "
        "There are also additional inputs without known outputs. "
        "Your task is to infer the rule and implement it in code.\n\n"
        "Each grid is a matrix of integers in [0,9], representing colors: "
        + ", ".join(f"{_COLOR_MAP[k]}: {k}" for k in range(10)) + ".\n\n"
        "Your transformation must be unambiguous and applicable to all shown examples "
        "and additional inputs. It doesn't have to be universally correct."
    )

def _reasoning_instruction() -> str:
    return (
        "Carefully reason inside <reasoning></reasoning> to infer the rule. "
        "Then write code.\n\n"
        "After your reasoning, output a Python function `transform(grid: list[list[int]]) -> list[list[int]]` "
        "inside triple backticks (```python ... ```). The solution must generalize across the given examples; "
        "do not hardcode outputs."
    )

def _other_instruction() -> str:
    return (
        "Do NOT include tests; only output the `transform` function. "
        "Optionally, you may also include a helper Python function that checks a hypothesis by comparing an input "
        "and an expected output (returns True/False)."
    )

def _problem_from_data(data: Dict[str, Any]) -> str:
    train = data.get("train", []) or []
    tests = data.get("test", []) or []

    segs: List[str] = []
    for i, demo in enumerate(train):
        segs.append(f"\n# Example {i+1}\n\n## Input\n{list_format(demo['input'])}\n")
        segs.append(f"\n## Output\n{list_format(demo['output'])}\n")

    for i, test in enumerate(tests):
        segs.append(f"\n# Additional Input {i+1}\n{list_format(test['input'])}\n")

    return "".join(segs)

# —— Reflection prompt (used by add_one_reflection) ——
REFLECT_PROMPT = (
    "Given the feedback, reflect inside <reflection></reflection> on what went wrong/right, "
    "update your hypothesis, and then write a new generalized `transform` implementation. "
    "Do NOT hardcode per-example outputs.\n"
)

NEW_REFLECT_PROMPT = (
    "\n---\n"
    "### Previous Response {index}\n"
    "{response}\n\n"
    "### Issues to Address\n"
    "{issues}\n"
    "---\n"
)

def ARC_AGI_PROMPT(*, data: Dict[str, Any], previous_response: str = "") -> str:
    “””
    Lightweight ARC-AGI unified template:
    - Always outputs task description + problem statement
    - If previous_response is non-empty, prepends it before the problem as “previous round's reasoning/reflection/code”
    “””
    head = _task_explanation() + "\n\n" + _reasoning_instruction() + "\n\n" + _other_instruction()
    body = _problem_from_data(data)

    if previous_response:
        prev = "\n# Previous Attempts / Notes\n" + previous_response.strip() + "\n"
    else:
        prev = ""

    tail = (
        "\n# Your Task\n"
        "1) Write <reasoning> ... </reasoning> with a clear, step-by-step hypothesis.\n"
        "2) Then output only a single Python code block containing `transform`.\n"
    )
    return head + prev + "\n# Problem\n" + body + tail
