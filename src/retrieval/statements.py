"""
Recognize financial statements in pages and in questions.

Many FinanceBench questions say where the answer is, e.g.
"...based on the cash flow statement". Plain vector search does not
use that hint. Here we:

    1. label each page with the statement it shows (from its title)
    2. find which statements a question mentions

so the search can move those pages to the top.
"""

import re


# ==========================================
# 1. STATEMENT NAMES IN PAGE TITLES
# ==========================================

# How each statement is titled in a report, e.g.
# "Consolidated Statements of Cash Flows", "Consolidated Balance Sheets"
PAGE_TITLE_PATTERNS = {
    "cash_flow": r"statements?\s+of\s+(?:consolidated\s+)?cash\s*flows?",
    "balance_sheet": r"balance\s*sheets?"
                     r"|statements?\s+of\s+(?:consolidated\s+)?financial\s+position",
    "income": r"statements?\s+of\s+(?:consolidated\s+)?(?:income|operations|earnings)"
              r"|income\s+statements?"
}

# The title is at the top of the page, so we only look at the start.
# (Lower down, normal text often says "see the Statement of Cash Flows".)
TITLE_REGION_CHARS = 400


def detect_page_statement(page_text: str):
    """
    Return the statement a page shows ("cash_flow", "balance_sheet",
    "income") or None.
    """

    # 1. Take the top of the page and clean up the spacing.
    #    PDF extraction sometimes breaks words, e.g. "Cash Flow s".
    head = re.sub(r"\s+", " ", page_text[:TITLE_REGION_CHARS])
    head = re.sub(r"(?i)\bflow s\b", "flows", head)

    # 2. Check which statement titles appear
    found = [
        name
        for name, pattern in PAGE_TITLE_PATTERNS.items()
        if re.search(pattern, head, re.IGNORECASE)
    ]

    # 3. Exactly one title = a statement page.
    #    Several titles = probably a table of contents, so no label.
    return found[0] if len(found) == 1 else None


# ==========================================
# 2. STATEMENT NAMES IN QUESTIONS
# ==========================================

# How people refer to each statement in a question
QUESTION_PATTERNS = {
    "cash_flow": r"cash\s*flows?\s+statement|statements?\s+of\s+cash\s*flows?",
    "balance_sheet": r"balance\s*sheets?|statements?\s+of\s+financial\s+position",
    "income": r"income\s+statements?|\bp\s*&\s*l\b|profit\s+and\s+loss"
              r"|statements?\s+of\s+(?:consolidated\s+)?(?:income|operations|earnings)"
}


def detect_question_statements(question: str) -> set[str]:
    """
    Return the statements a question mentions (can be empty).
    """

    return {
        name
        for name, pattern in QUESTION_PATTERNS.items()
        if re.search(pattern, question, re.IGNORECASE)
    }


# ==========================================
# 3. MOVE MATCHING PAGES TO THE TOP
# ==========================================

def boost_statement_chunks(
    ranked_chunks: list[tuple[int, float]],
    chunks: list[dict],
    page_statements: dict,
    wanted: set[str]
) -> list[tuple[int, float]]:
    """
    Put chunks from pages of the wanted statements first.

    Both groups keep their original order, so the vector search
    still decides the order inside each group.
    """

    if not wanted:
        return ranked_chunks

    boosted = []
    others = []

    for chunk_id, score in ranked_chunks:

        page = chunks[chunk_id]["metadata"]["page"]

        if page_statements.get(page) in wanted:
            boosted.append((chunk_id, score))
        else:
            others.append((chunk_id, score))

    return boosted + others
