LIVECODEBENCH_PROMPT = '''<GENERAL INSTRUCTIONS>
You are the best in the world at competitive programming and code debugging.
Given a problem statement and (optionally) starter code, your mission is to produce a correct, efficient
Python program that passes all tests.

**Rules**
- Read inputs from stdin and write outputs to stdout (unless the starter code specifies I/O wrappers).
- Handle edge cases and respect time/memory constraints implied by the problem.
- Prefer clear, iterative solutions over unnecessarily complex code.
- If starter code is provided, use it and fill in the required logic instead of discarding it.
- Final Output Requirement: Provide both your reasoning/thinking process and the final Python code enclosed in triple backticks like ```python ... ```.

**Reflection & Improvement Mode**
We will simulate a two-role process:
1) **Jane (Critic)**: Analyze the previous attempt, identify logical flaws, missing edge cases, and inefficiencies. Suggest concrete fixes.
2) **John (Coder)**: Using Jane’s analysis, produce the improved, correct, and optimized Python solution.

<END GENERAL INSTRUCTIONS>

## Example
**Problem Statement**
Given an integer n, print the sum of all integers from 1 to n inclusive.

**Starter Code**
```python
# YOUR CODE HERE
```

**Previous Attempt**
```python
n = int(input())
print(sum(range(1, n)))  # BUG: excludes n
```

**Jane's Analysis**
- Off-by-one error: `range(1, n)` excludes n; should be `range(1, n+1)`.
- Otherwise correct and efficient.

**John's Reasoning and Final Solution**
We fix the off-by-one by using `range(1, n+1)`. This preserves correctness and efficiency.
```python
n = int(input())
print(sum(range(1, n + 1)))
```

## Problem Statement
==================================================
{question_content}
==================================================

## Starter Code (if any)
"""
{starter_code}
"""

## Previous Attempt (if any)
"""
{previous_plan}
"""

<CURRENT INSTRUCTION>
Produce a drastically improved version of the solution that fixes all issues from prior attempts and
conforms to the problem’s constraints and I/O format.
Remember: perform your critique silently and output only the final Python code in triple double quotes (""").
<END CURRENT INSTRUCTION>
'''


REFLECT_PROMPT = """

## Previous Codes
Here are some code that were previously proposed and the corresponding issues with these codes:


"""
NEW_REFLECT_PROMPT = """**code_v{index}**
{response}
**End of code_v{index}**

**Issues with code_v{index}**
{issues}


"""