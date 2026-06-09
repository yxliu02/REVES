from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional


# Assumes these four constants are already defined as in the previous version:
# NQUEENS_INITIAL_SYSTEM
# NQUEENS_INITIAL_USER_TEMPLATE
# NQUEENS_REPAIR_SYSTEM
# NQUEENS_REPAIR_USER_TEMPLATE
from prompt.n_queens_pmt import NQUEENS_INITIAL_SYSTEM, NQUEENS_INITIAL_USER_TEMPLATE, NQUEENS_REPAIR_SYSTEM, NQUEENS_REPAIR_USER_TEMPLATE
def board_list_to_text(puzzle):
    """
    puzzle: list of list, e.g.
      [['_', 'Q', '_', '_'],
       ['_', '_', '_', '_'],
       ['Q', '_', '_', '_'],
       ['_', '_', 'Q', '_']]
    -> "_ Q _ _\n_ _ _ _\nQ _ _ _\n_ _ Q _"
    """
    return "\n".join(" ".join(row) for row in puzzle)

def format_mini_sudoku_puzzle(puzzle):
    """
    Convert puzzle matrix (0 = empty) into printable string form:
    4 _ _ _
    _ 3 _ _
    _ 1 3 _
    _ _ _ _
    """
    lines = []
    for row in puzzle:
        line = " ".join(str(x) if x != 0 else "_" for x in row)
        lines.append(line)
    return "\n".join(lines)
    
def get_nqueens_fields_from_query(query):
    meta = query["metadata"]
    puzzle = meta["puzzle"]
    board_text = board_list_to_text(puzzle)

    # n: prefer the n from difficulty; fall back to puzzle size
    n = meta.get("difficulty", {}).get("n", len(puzzle))

    # Total queens: for standard n-queens, this equals n
    total_queens = n

    # Number to place: prefer metadata["num_removed"], fall back to difficulty field
    num_removed = meta.get("num_removed", None)
    if isinstance(num_removed, int):
        num_to_place = num_removed
    else:
        # Fallback: e.g. if difficulty stores a range like (1,3), take the lower bound
        num_to_place = meta.get("difficulty", {}).get("num_removed", (1,))[0]

    return {
        "n": n,
        "total_queens": total_queens,
        "num_to_place": num_to_place,
        "board_text": board_text,
    }


def build_nqueens_user_prompt(query, prev_response=None, feedback=None):
    """
    Build user prompt using real data + the previously defined NQUEENS_*_TEMPLATE.
    prev_response being None / "" indicates a first response; otherwise it is a repair round.
    feedback can be 'correct' / 'incorrect' or a more detailed string.
    """
    fields = get_nqueens_fields_from_query(query)

    if not prev_response:
        # First response: use INITIAL_USER_TEMPLATE
        user_msg = NQUEENS_INITIAL_SYSTEM + NQUEENS_INITIAL_USER_TEMPLATE.format(**fields)
    else:
        # Repair round: use REPAIR_USER_TEMPLATE
        if feedback is None:
            feedback = "incorrect"
        user_msg = NQUEENS_REPAIR_SYSTEM + NQUEENS_REPAIR_USER_TEMPLATE.format(
            previous_answer=prev_response,
            binary_feedback=feedback,
            **fields,
        )

    return user_msg

def build_prompt(
    *,
    task_name: str,
    query: Dict[str, Any] | Any,
    prev_response: str = "",
) -> str:

    if task_name == "TravelPlanner":
        from prompt.TravelPlanner_pmt import TRAVELPLAN_PROMPT  # local import
        return TRAVELPLAN_PROMPT.format(
            query=query["query"],
            previous_plan=prev_response,
            reference_information=query.get("reference_information")
        )

    if task_name == "Natural-meeting":
        from prompt.NaturalPlan_pmt import NATURALPLAN_MEETING_PROMPT
        return NATURALPLAN_MEETING_PROMPT.format(
            query=query["prompt_0shot"],
            prev_response=prev_response,
        )

    if task_name == "Natural-trip":
        from prompt.NaturalPlan_pmt import NATURALPLAN_TRIP_PROMPT
        return NATURALPLAN_TRIP_PROMPT.format(
            query=query["prompt_0shot"],
            prev_response=prev_response,
        )

    if task_name == "LiveCodeBench" or task_name == "CodeContest":
        # Build a single-string prompt for code generation problems
        import os
        change_prompt_flag = os.environ.get("change_prompt", "").lower() == "true"

        if change_prompt_flag:
            # Use new template
            from prompt.LiveCodeBench_pmt import LIVECODEBENCH_PROMPT
            try:
                question_content = getattr(query, "question_content")
            except Exception:
                question_content = (query.get("question_content", "") if isinstance(query, dict) else "")

            try:
                starter_code = getattr(query, "starter_code")
            except Exception:
                starter_code = (query.get("starter_code", "") if isinstance(query, dict) else "")

            question_content = question_content or ""
            starter_code = starter_code or ""

            return LIVECODEBENCH_PROMPT.format(
                question_content=question_content,
                starter_code=starter_code,
                previous_plan=prev_response or ""
            )

        else:
            # Use the original concatenation logic
            try:
                question_content = getattr(query, "question_content")
                starter_code = getattr(query, "starter_code")
            except Exception:
                question_content = query.get("question_content", "")
                starter_code = query.get("starter_code", "")

            system_header = (
                "You are an expert Python programmer. You will be given a problem specification "
                "and must produce a correct Python program that passes the tests."
            )
            prompt_lines: List[str] = [system_header, "", f"### Question:\n{question_content}", ""]
            if starter_code:
                prompt_lines.append(prev_response)
                prompt_lines.append(
                    "### Format: You will use the following starter code and enclose your final code in triple backticks."
                )
                prompt_lines.append(f"```python\n{starter_code}\n```\n")
            else:
                prompt_lines.append(prev_response)
                prompt_lines.append(
                    "### Format: Read inputs from stdin and write the answer to stdout. Enclose your final code in triple backticks."
                )
                prompt_lines.append("```python\n# YOUR CODE HERE\n```\n")
            prompt_lines.append("### Answer: Provide ONLY the final Python code, enclosed in triple backticks.\n")
            return "\n".join(prompt_lines)
            
    if task_name == "ARC-AGI":
        from prompt.ARC_AGI_pmt import ARC_AGI_PROMPT  # local import

        # Compatible with both objects and dicts
        if isinstance(query, dict):
            data = {"train": query.get("train", []), "test": query.get("test", [])}
        else:
            data = {"train": getattr(query, "train", []) or [],
                    "test": getattr(query, "test", []) or []}

        return ARC_AGI_PROMPT(data=data, previous_response=prev_response or "")
    if task_name == "n_queens":
        # Use the unified function here to assemble real data into the format of those four prompts
        return build_nqueens_user_prompt(query, prev_response=prev_response, feedback="incorrect")
    if task_name == "mini_sudoku":
        from prompt.mini_sudoku_pmt import MINI_SUDOKU_INITIAL_PROMPT, MINI_SUDOKU_REPAIR_PROMPT
        if not prev_response:
            return MINI_SUDOKU_INITIAL_PROMPT.format(puzzle=format_mini_sudoku_puzzle(query["metadata"]["puzzle"]))
        else:
            return MINI_SUDOKU_REPAIR_PROMPT.format(
                puzzle=format_mini_sudoku_puzzle(query["metadata"]["puzzle"]),
                previous_answer=prev_response,
                feedback="The response is incorrect."
            )
    if task_name == "Countdown":
        if not prev_response:
            from prompt.countdown_pmt import COUNTDOWN_INITIAL_USER_TEMPLATE, COUNTDOWN_INITIAL_SYSTEM
            return COUNTDOWN_INITIAL_SYSTEM + "\n" + COUNTDOWN_INITIAL_USER_TEMPLATE.format(
                numbers=query["metadata"]["numbers"], 
                target=query["metadata"]["target"]
            )
        else:
            from prompt.countdown_pmt import COUNTDOWN_REPAIR_USER_TEMPLATE, COUNTDOWN_REPAIR_SYSTEM
            return COUNTDOWN_REPAIR_SYSTEM + "\n" + COUNTDOWN_REPAIR_USER_TEMPLATE.format(
                numbers=query["metadata"]["numbers"], 
                target=query["metadata"]["target"], 
                previous_answer=prev_response, 
                feedback="The response is incorrect. Now please give me a new answer."
            )
    if task_name == "AIME25" or task_name == "AIME24" or task_name == "MATH500":
        if not prev_response:
            # previous_response is ""
            from prompt.math_pmt import INITIAL_USER_TEMPLATE
            return INITIAL_USER_TEMPLATE.format(problem=query['prompt'][0]['content'])
        else:
            from prompt.math_pmt import REPAIR_USER_TEMPLATE
            return REPAIR_USER_TEMPLATE.format(
                problem=query['prompt'][0]['content'],
                previous_answer=prev_response,
                binary_feedback="The response is incorrect."
            )
    raise ValueError(f"Unsupported task: {task_name}")



def issues_from_eval(task_name: str, eva: Dict[str, Any]) -> str:
    """Convert evaluator details dict into a *human readable* issue list."""
    if task_name == "TravelPlanner":
        commonsense = eva.get("commonsense_details")
        hard = eva.get("hard_details")
        score = eva.get("score")

        if commonsense is None and hard is None:
            return "No plan was extracted. Please check the format of the input."

        msgs: List[str] = []

        if isinstance(commonsense, dict):
            msgs.extend(str(m) for ok, m in commonsense.values() if ok is False and m)

        if isinstance(hard, dict):
            msgs.extend(str(m) for ok, m in hard.values() if ok is False and m)

        if not msgs:
            return (
                "Plan is not perfect according to the evaluator, "
                "but no detailed issues were provided." if score != 0.0 else "No issues."
            )

        return "\n".join(msgs)
    elif task_name == "LiveCodeBench" or task_name == "CodeContest":
        if not eva.get("errors"):
            return "The code passed all public available test cases with no issues found."

        descriptions = []
        # print(eva)
        for idx, err in enumerate(eva["errors"], 1):
            inp = err.get("inputs", "")
            exp = err.get("expected", "")
            out = err.get("output", "")
            msg = err.get("error_message", "")

            # Include expected output even if using the provided message
            if msg and msg not in (f"{out} != {exp}",):
                desc = (
                    f"Test case {idx}: For input:"
                    f"`{inp}`, "
                    f"expected output is: "
                    f"`{exp}`, "
                    f"but {msg}"
                )
            else:
                desc = (
                    f"Test case {idx}: For input `{inp}`, expected `{exp}`, "
                    f"but got `{out}`."
                )
            descriptions.append(desc)

        return "\n".join(descriptions)
    elif task_name == "ARC-AGI":
        """
        Expected eva may contain (any one or more of the following):
          - "train_results": [
                {"index": 0, "is_correct": False, "pred": [[...]], "expected": [[...]], "note": "..."},
                ...
            ]
          - "tests_pred": [
                {"index": 0, "valid": True,  "pred": [[...]]} or {"index": 0, "valid": False, "error": "..."}
            ]
          - "parse_error": "...", "runtime_error": "..."
        All are optional; if none are present, return a generic message.
        """
        msgs: List[str] = []

        # Parse errors from the previous round on train examples
        tr = eva.get("train_results")
        if isinstance(tr, list) and tr:
            for item in tr:
                idx = item.get("index")
                ok = item.get("is_correct")
                if ok is True:
                    continue
                note = item.get("note")
                pred = item.get("pred")
                exp  = item.get("expected")
                line = f"Train Example {idx}: wrong output."
                if note:
                    line += f" Hint: {note}"
                if pred is not None and exp is not None:
                    line += " Diff between your output and expected is shown; avoid hard-coding and generalize the rule."
                msgs.append(line)

        # Parse validity/generalization on additional inputs
        tp = eva.get("tests_pred")
        if isinstance(tp, list) and tp:
            for item in tp:
                idx = item.get("index")
                valid = item.get("valid", True)
                if not valid:
                    err = item.get("error", "invalid output")
                    msgs.append(f"Additional Input {idx}: transform() failed or invalid — {err}.")
                # If needed, a “does not seem to generalize” hint can be added on the evaluator side; here we only show failures as a fallback

        # Other generic errors
        if "parse_error" in eva:
            msgs.append(f"Parse error when extracting code: {eva.get('parse_error')}")
        if "runtime_error" in eva:
            msgs.append(f"Runtime error when executing transform(): {eva.get('runtime_error')}")

        if not msgs:
            return "No specific issues provided by the evaluator. Re-check your hypothesized rule and ensure generalization."

        return "\n".join(msgs)
    elif task_name == "Countdown" or task_name == "n_queens" or task_name == "mini_sudoku":
        return "The response is incorrect."
    elif task_name == "AIME25" or task_name == "AIME24" or task_name == "MATH500":
        return "The response is incorrect."
    else:
        raise(ValueError("Error"))


# def issues_from_eval_code(eva: Dict[str, Any]) -> str:
#     """
#     Generate a human-readable description of code issues
#     based on the output of eval_public_only.
#     """
#     if not eva.get("errors"):
#         return "The code passed all test cases with no issues found."

#     descriptions = []
#     for idx, err in enumerate(eva["errors"], 1):
#         inp = err.get("inputs", "").strip()
#         exp = err.get("expected", "").strip()
#         out = err.get("output", "").strip()
#         msg = err.get("error_message", "").strip()

#         # Include expected output even if using the provided message
#         if msg and msg not in (f"{out} != {exp}",):
#             desc = (
#                 f"Test case {idx}: For input `{inp}`, expected `{exp}`, "
#                 f"but {msg}"
#             )
#         else:
#             desc = (
#                 f"Test case {idx}: For input `{inp}`, expected `{exp}`, "
#                 f"but got `{out}`."
#             )
#         descriptions.append(desc)

#     return "\n".join(descriptions)


def add_one_reflection(
    *,
    index: int,
    task_name: str,
    obj: Any,
    issues: str,
) -> str:
    """
    Format a single reflection block given one parent.
    Adds REFLECT_PROMPT if index == 1.
    Currently supports only TravelPlanner.
    """
    if task_name == "TravelPlanner":
        from prompt.TravelPlanner_pmt import REFLECT_PROMPT, NEW_REFLECT_PROMPT
    elif task_name == "LiveCodeBench" or task_name == "CodeContest":
        from prompt.LiveCodeBench_pmt import REFLECT_PROMPT, NEW_REFLECT_PROMPT
    elif task_name == "ARC-AGI":
        from prompt.ARC_AGI_pmt import REFLECT_PROMPT, NEW_REFLECT_PROMPT
    elif task_name == "Countdown" or task_name == "n_queens" or task_name == "mini_sudoku":
        return obj
    elif task_name == "AIME25" or task_name == "AIME24" or task_name == "MATH500":
        return obj
    block = NEW_REFLECT_PROMPT.format(
        index=index,
        response=obj,
        issues=issues,
    )

    if index == 1:
        return REFLECT_PROMPT + block
    else:
        return block


def build_reflection_text(
    *,
    task_name: str,
    parents: List[Dict[str, Any]],
) -> str:
    """
    Build full reflection text by calling `add_one_reflection` on each parent.
    Each parent Dict must include:
        - "obj": str or dict
        - "issues": str
    """
    if not parents:
        return ""

    all_reflections: List[str] = []

    for i, parent in enumerate(parents, start=1):
        reflection_text = add_one_reflection(
            index=i,
            task_name=task_name,
            obj=parent["obj"],
            issues=parent["issues"],
        )
        all_reflections.append(reflection_text)

    return "".join(all_reflections)


def scale_score_fixed(task_name: str, score: float) -> float:
    """
    Map evaluator raw score {-500, -100, -13..0} -> [0,1].
    Keep the same shaping you’ve been using elsewhere.
    """
    if task_name == "TravelPlanner":
        if score == -500.0:
            return 0.0
        if score == -100.0:
            return 0.2
        # linear map [-13, 0] -> (0.2, 1]
        return (score + 13.0) / 13.0
    elif task_name == "LiveCodeBench" or task_name == "CodeContest":
        if score == -100:
            return 0.0
        else:
            return score + 1
    elif task_name == "ARC-AGI":
        if score == -100:
            return 0.0
        else:
            return score + 1
    else:
        raise ValueError("Not implemented yet")