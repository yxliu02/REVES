# prompt/math_pmt.py
# -*- coding: utf-8 -*-

# ========== Initial Answer ==========
INITIAL_SYSTEM = "You are a careful problem solver."
INITIAL_USER_TEMPLATE = (
    "Solve the following math problem. Show brief reasoning and provide the final answer.\n\n"
    "Problem:\n{problem}\n"
)

# ========== Repair (binary feedback only) ==========
REPAIR_SYSTEM = (
    "You are a math solver. You will receive the previous attempt and a verifiable binary feedback.\n"
    "The feedback only tells you whether the previous answer is correct or incorrect.\n"
    "Use it to produce a better, corrected solution. Do NOT invent or assume the correct answer."
)
REPAIR_USER_TEMPLATE = (
    "Problem:\n{problem}\n\n"
    "Previous Attempt:\n{previous_answer}\n\n"
    "Verifiable Feedback:\n{binary_feedback}\n\n"
    "Now produce a corrected solution. Keep reasoning concise.\n"
)

BINARY_FEEDBACK_INCORRECT = "The previous answer is incorrect."

# ========== Verification ==========
VERIFICATION_SYSTEM = (
    "You are a rigorous math verifier. Read the problem and the proposed answer, "
    "then verify concisely. End the last line with `VERDICT: \\boxed{{True/False}}`."
)
VERIFICATION_USER_TEMPLATE = (
    "Problem:\n{problem}\n\n"
    "Proposed Answer (full text):\n{answer}\n\n"
    "Judge correctness. Keep it brief and end with `VERDICT: \\boxed{{True/False}}`."
)

VERIFICATION_SYSTEM_2 = (
    "You are a rigorous math verifier. Read the problem and the proposed answer, "
    "then reason step by step, checking each key claim carefully. "
    "Conclude with a final line exactly in the format: `VERDICT: \\boxed{{True}}` or `VERDICT: \\boxed{{False}}`."
)

VERIFICATION_USER_TEMPLATE_2 = (
    "Problem:\n{problem}\n\n"
    "Proposed Answer (full text):\n{answer}\n\n"
    "Verify the proposed answer by reasoning step by step and checking the logic and computations carefully. "
    "At the end, output exactly one final line in the format: `VERDICT: \\boxed{{True}}` or `VERDICT: \\boxed{{False}}`."
)