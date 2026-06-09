# prompt/code_pmt.py
# -*- coding: utf-8 -*-

# ========== INITIAL ==========
INITIAL_SYSTEM = "You are an expert competitive programmer. Write correct, efficient Python code."
INITIAL_USER_TEMPLATE = """Problem:
{problem}

Please write a correct and efficient Python program. 
Enclose your code within the following delimiters:

```python
# YOUR CODE HERE
```"""

# ========== REPAIR ==========
REPAIR_SYSTEM = "You are an expert competitive programmer. Fix the code to make it correct."
REPAIR_USER_TEMPLATE = """Problem:
{problem}

Previous Answer:
{previous_answer}

Feedback: {binary_feedback}

Please repair the code accordingly. 
Enclose your fixed code within:

```python
# YOUR CODE HERE
```"""

# ========== VERIFICATION ==========
VERIFICATION_SYSTEM = "You are a strict code verifier."
VERIFICATION_USER_TEMPLATE = """Problem:
{problem}

Candidate Answer:
{answer}

Please verify whether the candidate answer is correct. 
At the end, output the final line exactly as:
VERDICT: \\boxed{{True/False}}"""

# ========== FEEDBACK ==========
BINARY_FEEDBACK_INCORRECT = "incorrect"
