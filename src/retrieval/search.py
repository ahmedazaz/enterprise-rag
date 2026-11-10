"""
Ask one question about one PDF and print what each method finds.

Usage:
    python -m src.retrieval.search
    python -m src.retrieval.search --pdf data/raw/APPLE_2022_10K.pdf --question "..."
"""

import argparse
from pathlib import Path

from src.config import RAW_PDF_DIR, TOP_K
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.pipeline import (
    SEARCH_METHODS,
    build_document_index,
    retrieve
)


DEFAULT_PDF = RAW_PDF_DIR / "3M_2018_10K.pdf"

DEFAULT_QUESTION = (
    "What is the FY2018 capital expenditure amount "
    "(in USD millions) for 3M? Give a response to the "
    "question by relying on the details shown in the "
    "cash flow statement."
)

PREVIEW_CHARS = 500


def print_results(method: str, results: list, chunks: list[dict]):
    """
    Print one method's ranked chunks.
    """

    print(f"\n========== {method} RESULTS ==========\n")

    for rank, (chunk_id, score) in enumerate(results, start=1):

        chunk = chunks[chunk_id]

        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Page: {chunk['metadata']['page']}")
        print("Text:")
        print(chunk["text"][:PREVIEW_CHARS])
        print("-" * 60)


def run_search(pdf_path: Path, question: str, top_k: int):

    # 1. Check the PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 2. Load the embedding model
    model = load_embedding_model()

    # 3. Build the indexes for this PDF
    document_index = build_document_index(pdf_path, model)

    print("Total pages:", len(document_index["pages"]))
    print("Total chunks:", len(document_index["chunks"]))

    print("\n========== USER QUERY ==========\n")
    print(question)

    # 4. Search with all three methods
    results = retrieve(question, document_index, model, top_k)

    # 5. Show the results
    for method in SEARCH_METHODS:
        print_results(method, results[method], document_index["chunks"])


def parse_args():

    parser = argparse.ArgumentParser(description="Search one PDF.")

    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=TOP_K)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_search(args.pdf, args.question, args.top_k)
