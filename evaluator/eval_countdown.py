import re
from typing import Optional, Tuple

import reasoning_gym


def _extract_braced_content(s: str, start_idx: int) -> Optional[Tuple[str, int]]:
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


def normalize_countdown_answer(answer: str) -> str:
    """
    Convert common LaTeX arithmetic into a plain expression for scoring.
    - Keeps extraction in utils simple; evaluator is responsible for LaTeX tolerance.
    """
    s = answer.strip()

    # If someone passed a boxed answer (or raw output), normalize to inner content.
    boxed = _extract_boxed_content(s)
    if boxed is not None:
        s = boxed

    # Collapse double backslashes that sometimes appear in model outputs (e.g., "\\\\times")
    while "\\\\" in s:
        s = s.replace("\\\\", "\\")

    # Remove common wrappers / spacing
    s = s.replace("$", "")
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace(r"\,", "")

    # Normalize multiplication operators
    s = s.replace(r"\times", "*").replace(r"\cdot", "*")

    # Normalize \frac{a}{b} with brace balancing (supports nesting)
    i = 0
    out = []
    while i < len(s):
        if s.startswith(r"\frac{", i):
            num_start = i + len(r"\frac")
            if num_start < len(s) and s[num_start] == "{":
                num_parsed = _extract_braced_content(s, num_start)
                if num_parsed:
                    num, j = num_parsed
                    if j < len(s) and s[j] == "{":
                        den_parsed = _extract_braced_content(s, j)
                        if den_parsed:
                            den, k = den_parsed
                            out.append(f"({num}/{den})")
                            i = k
                            continue
        out.append(s[i])
        i += 1
    s = "".join(out)

    # Remove all whitespace for scorer robustness
    s = re.sub(r"\s+", "", s)
    return s

def evaluate_countdown(query: dict, answer: str) -> dict:
    dataset = reasoning_gym.create_dataset(
        name="countdown",
        size=1,
        seed=42,
    )
    score = dataset.score_answer(
        answer=normalize_countdown_answer(answer),
        entry=query,
    )
    return {"score": 0} if score == 1 else {"score": -1}

