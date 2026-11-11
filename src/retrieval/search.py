"""
Ask one question about one PDF and print the pages that answer it.

By default it uses the best method (FAISS+Statement).

Usage:
    python -m src.retrieval.search
    python -m src.retrieval.search --pdf data/raw/APPLE_2022_10K.pdf --question "..."
    python -m src.retrieval.search --all-methods
"""

import argparse
from pathlib import Path

from src.config import RAW_PDF_DIR, TOP_K
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.pipeline import BEST_METHOD, build_document_index, retrieve
from src.retrieval.statements import detect_question_statements


DEFAULT_PDF = RAW_PDF_DIR / "3M_2018_10K.pdf"

DEFAULT_QUESTION = (
    "What is the FY2018 capital expenditure amount "
    "(in USD millions) for 3M? Give a response to the "
    "question by relying on the details shown in the "
    "cash flow statement."
)

PREVIEW_CHARS = 500


def best_chunk_per_page(ranked_chunks: list, chunks: list[dict], top_k: int) -> list:
    """
    Keep only the best chunk of each page, for the first `top_k` pages.

    Several chunks can come from the same page; showing the page
    once keeps the results easy to read.
    """

    seen_pages = set()
    results = []

    for chunk_id, score in ranked_chunks:

        page = chunks[chunk_id]["metadata"]["page"]

        if page in seen_pages:
            continue

        seen_pages.add(page)
        results.append((chunk_id, score))

        if len(results) == top_k:
            break

    return results


def print_results(method: str, results: list, document_index: dict):
    """
    Print one method's ranked pages.
    """

    chunks = document_index["chunks"]
    page_statements = document_index["page_statements"]

    print(f"\n========== {method} RESULTS ==========\n")

    for rank, (chunk_id, score) in enumerate(results, start=1):

        chunk = chunks[chunk_id]
        page = chunk["metadata"]["page"]

        # Show the statement label if this page is a financial statement
        label = page_statements.get(page)
        label_text = f"  [{label}]" if label else ""

        print(f"Rank: {rank}")
        print(f"Page: {page}{label_text}")
        print(f"Score: {score:.4f}")
        print("Text:")
        print(chunk["text"][:PREVIEW_CHARS])
        print("-" * 60)


def run_search(pdf_path: Path, question: str, top_k: int, all_methods: bool):

    # 1. Check the PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 2. Load the embedding model
    model = load_embedding_model()

    # 3. Build the indexes for this PDF
    document_index = build_document_index(pdf_path, model)

    print("Total pages:", len(document_index["pages"]))
    print("Total chunks:", len(document_index["chunks"]))

    # 4. Show the question and any statement it mentions
    print("\n========== USER QUERY ==========\n")
    print(question)

    statements = detect_question_statements(question)
    print("\nStatements mentioned:", ", ".join(sorted(statements)) or "none")

    # 5. Search
    results = retrieve(question, document_index, model)

    # 6. Show the best method, or every method if asked
    methods = list(results) if all_methods else [BEST_METHOD]

    for method in methods:

        top_pages = best_chunk_per_page(
            results[method],
            document_index["chunks"],
            top_k
        )

        print_results(method, top_pages, document_index)


def parse_args():

    parser = argparse.ArgumentParser(description="Search one PDF.")

    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=5,
                        help=f"pages to show (default 5, evaluation uses {TOP_K})")
    parser.add_argument("--all-methods", action="store_true",
                        help="show every search method, not only the best one")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_search(args.pdf, args.question, args.top_k, args.all_methods)
