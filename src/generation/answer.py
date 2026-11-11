"""
Answer a question about a PDF: search for the right pages, then let
the LLM write an answer using only those pages.

    question -> search (FAISS+Statement) -> top pages -> LLM -> answer

Usage:
    python -m src.generation.answer
    python -m src.generation.answer --pdf data/raw/APPLE_2022_10K.pdf --question "..."
"""

import argparse
from pathlib import Path

from src.config import CONTEXT_PAGES, RAW_PDF_DIR
from src.generation.llm import ask_llm
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.pipeline import BEST_METHOD, build_document_index, retrieve


DEFAULT_PDF = RAW_PDF_DIR / "3M_2018_10K.pdf"

DEFAULT_QUESTION = (
    "What is the FY2018 capital expenditure amount "
    "(in USD millions) for 3M? Give a response to the "
    "question by relying on the details shown in the "
    "cash flow statement."
)

PROMPT_TEMPLATE = """You are a financial analyst. Answer the question using ONLY the pages below.

Rules:
- Use only numbers and facts written in the pages. Do not use outside knowledge.
- If you calculate something, show the numbers you used.
- Say which page(s) you used, like this: (page 60).
- If the pages do not contain the answer, reply exactly: "Not found in the provided pages."

{context}

Question: {question}

Answer:"""


# ==========================================
# STEP 1: PICK THE PAGES TO READ
# ==========================================

def top_pages(ranked_chunks: list, chunks: list[dict], n_pages: int) -> list[int]:
    """
    Return the first `n_pages` different page numbers in the ranking.
    """

    pages = []

    for chunk_id, _ in ranked_chunks:

        page = chunks[chunk_id]["metadata"]["page"]

        if page not in pages:
            pages.append(page)

        if len(pages) == n_pages:
            break

    return pages


# ==========================================
# STEP 2: BUILD THE PROMPT
# ==========================================

def build_context(page_numbers: list[int], document_index: dict) -> str:
    """
    Join the FULL text of the chosen pages, each with its page number.

    We give whole pages (not chunks) so tables are not cut in half.
    """

    # Look up each page's text by its page number
    texts = {
        page["metadata"]["page"]: page["text"]
        for page in document_index["pages"]
    }

    return "\n\n".join(
        f"===== Page {number} =====\n{texts[number]}"
        for number in page_numbers
    )


# ==========================================
# STEP 3: ANSWER
# ==========================================

def answer_question(
    question: str,
    document_index: dict,
    model,
    n_pages: int = CONTEXT_PAGES
) -> dict:
    """
    Search the document and let the LLM answer from the top pages.

    Returns {"answer": str, "pages": [page numbers the LLM read]}.
    """

    # 1. Search with the best method
    results = retrieve(question, document_index, model)

    # 2. Pick the top pages
    pages = top_pages(results[BEST_METHOD], document_index["chunks"], n_pages)

    # 3. Build the prompt and ask the LLM
    prompt = PROMPT_TEMPLATE.format(
        context=build_context(pages, document_index),
        question=question
    )

    return {
        "answer": ask_llm(prompt),
        "pages": pages
    }


def parse_args():

    parser = argparse.ArgumentParser(description="Answer a question about one PDF.")

    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--pages", type=int, default=CONTEXT_PAGES,
                        help="how many top pages the LLM reads")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # 1. Build the search indexes for the PDF
    embedding_model = load_embedding_model()
    document_index = build_document_index(args.pdf, embedding_model)

    # 2. Answer
    result = answer_question(args.question, document_index, embedding_model, args.pages)

    print("\n========== QUESTION ==========\n")
    print(args.question)
    print("\nPages read:", result["pages"])
    print("\n========== ANSWER ==========\n")
    print(result["answer"])
