"""
LLM-as-a-Judge: decide whether the system's answer is correct.

Answers are free text ("$1,577 million" vs "$1577.00"), so a simple
string comparison does not work. The LLM compares the MEANING of the
system's answer with the correct answer, like a teacher grading an exam.
"""

import re

from src.generation.llm import ask_llm


JUDGE_PROMPT = """You are grading answers to questions about company financial reports.

Question:
{question}

Correct answer:
{gold_answer}

Answer to grade:
{system_answer}

Grading rules:
- CORRECT if the answer to grade reaches the same conclusion or the same number as the correct answer.
- Small differences in rounding, units or formatting are fine
  (e.g. "$1,577 million" = "$1577.00" = "1.58 billion").
- For yes/no questions, the yes/no conclusion must match.
- INCORRECT if the number or conclusion is different, missing, or the answer says it was not found.
- Do not judge the style or the explanation, only whether the final answer is right.

Reply in exactly this format:
VERDICT: CORRECT or INCORRECT
REASON: one short sentence"""


def judge_answer(question: str, gold_answer: str, system_answer: str) -> dict:
    """
    Ask the LLM to grade one answer.

    Returns {"verdict": "CORRECT" / "INCORRECT" / "UNKNOWN", "reason": str}.
    UNKNOWN means the judge's reply could not be read.
    """

    # 1. Ask the judge
    reply = ask_llm(JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        system_answer=system_answer
    ))

    # 2. Read the verdict and the reason from the reply
    verdict = re.search(r"VERDICT:\s*(CORRECT|INCORRECT)", reply, re.IGNORECASE)
    reason = re.search(r"REASON:\s*(.+)", reply, re.IGNORECASE)

    return {
        "verdict": verdict.group(1).upper() if verdict else "UNKNOWN",
        "reason": reason.group(1).strip() if reason else reply[:200]
    }
