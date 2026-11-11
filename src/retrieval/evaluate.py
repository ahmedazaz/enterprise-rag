"""
Measure how well each search method finds the evidence pages
for the FinanceBench questions.

Usage:
    python -m src.retrieval.evaluate --name baseline
    python -m src.retrieval.evaluate --name pages --chunk-size 0
    python -m src.retrieval.evaluate --name rerank --rerank

Steps:
    1. Load the FinanceBench questions
    2. Group the questions by PDF
    3. For each PDF: build indexes, search every question, score it
    4. Save one CSV row per (question, method) and a summary JSON
    5. Print the average score of each method
"""

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EXPERIMENTS_DIR,
    FINANCEBENCH_QUESTIONS,
    RAW_PDF_DIR,
    TOP_K
)
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.pipeline import (
    BASE_METHODS,
    RERANK_METHODS,
    build_document_index,
    retrieve
)
from src.retrieval.reranker import load_reranker


CSV_COLUMNS = [
    "financebench_id",
    "company",
    "doc_name",
    "method",
    "gold_pages",
    "ranked_pages",
    "first_relevant_rank",
    "mrr_at_10",
    "hit_at_1",
    "hit_at_5",
    "recall_at_5",
    "hit_at_10",
    "recall_at_10"
]

SUMMARY_METRICS = [
    "hit_at_1",
    "hit_at_5",
    "recall_at_5",
    "mrr_at_10",
    "hit_at_10",
    "recall_at_10"
]


# ==========================================
# STEP 1: LOAD QUESTIONS
# ==========================================

def load_financebench() -> list[dict]:
    """
    Read the FinanceBench questions (one JSON object per line).
    """

    records = []

    with open(FINANCEBENCH_QUESTIONS, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


# ==========================================
# STEP 2: GROUP QUESTIONS BY PDF
# ==========================================

def group_by_document(records: list[dict]) -> dict:
    """
    Return {doc_name: [questions about that document]}.

    Grouping lets us build each PDF's index only once.
    """

    grouped = defaultdict(list)

    for record in records:
        doc_name = Path(record["doc_name"]).stem
        grouped[doc_name].append(record)

    return grouped


def build_pdf_map() -> dict:
    """
    Return {pdf_name_without_extension: pdf_path}.
    """

    return {
        pdf.stem: pdf
        for pdf in RAW_PDF_DIR.rglob("*.pdf")
    }


# ==========================================
# STEP 3a: FIND THE CORRECT PAGES
# ==========================================

def get_gold_pages(record: dict) -> set[int]:
    """
    Return the page numbers that contain the answer evidence.
    """

    gold_pages = set()

    for evidence in record.get("evidence", []):

        page_number = evidence.get("evidence_page_num")

        if page_number is not None:

            # FinanceBench pages are zero-based (see its README).
            # Our extractor numbers pages from 1.
            gold_pages.add(int(page_number) + 1)

    return gold_pages


# ==========================================
# STEP 3b: TURN RANKED CHUNKS INTO RANKED PAGES
# ==========================================

def chunks_to_pages(
    ranked_chunks: list[tuple[int, float]],
    chunks: list[dict],
    top_k: int = TOP_K
) -> list[int]:
    """
    Convert a chunk ranking into a ranking of DIFFERENT pages.

    Several chunks can come from the same page. Without this step,
    three chunks of page 43 would fill three of the top 10 slots,
    and "top 10" would mean a different number of pages for each
    method. Each page keeps the position of its best chunk.
    """

    pages = []

    for chunk_id, _ in ranked_chunks:

        page = chunks[chunk_id]["metadata"]["page"]

        if page not in pages:
            pages.append(page)

        if len(pages) == top_k:
            break

    return pages


# ==========================================
# STEP 3c: SCORE ONE RANKING
# ==========================================

def calculate_metrics(
    ranked_pages: list[int],
    gold_pages: set[int]
) -> dict:
    """
    Compare the retrieved pages with the correct pages.

    hit_at_k        1 if any correct page is in the top k, else 0
    recall_at_k     share of correct pages found in the top k
    mrr_at_10       1 / rank of the first correct result (0 if none)
    """

    # 1. Rank of the first correct page
    first_relevant_rank = None

    for rank, page in enumerate(ranked_pages[:TOP_K], start=1):

        if page in gold_pages:
            first_relevant_rank = rank
            break

    # 2. Mean Reciprocal Rank
    metrics = {
        "first_relevant_rank": first_relevant_rank or "",
        "mrr_at_10": 1 / first_relevant_rank if first_relevant_rank else 0.0
    }

    # 3. Hit rate at 1, 5 and 10; recall at 5 and 10
    for k in (1, 5, 10):

        found = set(ranked_pages[:k]) & gold_pages

        metrics[f"hit_at_{k}"] = int(len(found) > 0)

        if k != 1:
            metrics[f"recall_at_{k}"] = len(found) / len(gold_pages)

    return metrics


# ==========================================
# STEP 3d: EVALUATE ONE PDF
# ==========================================

def evaluate_document(
    doc_name: str,
    pdf_path: Path,
    questions: list[dict],
    model,
    reranker,
    settings: argparse.Namespace
) -> list[dict]:
    """
    Build the indexes for one PDF and score every question on it.

    Returns the CSV rows for this document. If anything fails,
    an exception is raised and no rows are returned, so the
    results never contain half of a document.
    """

    # 1. Build FAISS + BM25 indexes for this PDF
    document_index = build_document_index(
        pdf_path,
        model,
        settings.chunk_size,
        settings.overlap
    )
    chunks = document_index["chunks"]

    print("Pages:", len(document_index["pages"]))
    print("Chunks:", len(chunks))

    rows = []

    # 2. Search and score every question
    for record in questions:

        gold_pages = get_gold_pages(record)

        results = retrieve(
            record["question"],
            document_index,
            model,
            reranker=reranker
        )

        for method, ranked_chunks in results.items():

            # 3. Convert chunk ids to different page numbers
            ranked_pages = chunks_to_pages(ranked_chunks, chunks)

            # 4. Score the ranking and store one row
            rows.append({
                "financebench_id": record["financebench_id"],
                "company": record["company"],
                "doc_name": doc_name,
                "method": method,
                "gold_pages": ";".join(map(str, sorted(gold_pages))),
                "ranked_pages": ";".join(map(str, ranked_pages)),
                **calculate_metrics(ranked_pages, gold_pages)
            })

    return rows


# ==========================================
# STEP 4: SAVE RESULTS
# ==========================================

def save_results(rows: list[dict], name: str):

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EXPERIMENTS_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], methods: list[str]) -> dict:
    """
    Return {method: {metric: average}}.
    """

    summary = {}

    for method in methods:

        method_rows = [row for row in rows if row["method"] == method]

        if not method_rows:
            continue

        summary[method] = {
            metric: sum(row[metric] for row in method_rows) / len(method_rows)
            for metric in SUMMARY_METRICS
        }
        summary[method]["questions"] = len(method_rows)

    return summary


# ==========================================
# STEP 5: PRINT SUMMARY
# ==========================================

def print_summary(summary: dict, report: dict, name: str):

    print("\n========== EVALUATION SUMMARY ==========")

    for method, scores in summary.items():

        print(f"\nMethod: {method}")
        print("Evaluated questions:", scores["questions"])

        for metric in SUMMARY_METRICS:
            print(f"{metric}: {scores[metric]:.4f} ({scores[metric] * 100:.2f}%)")

    print("\n========== PROCESSING REPORT ==========\n")

    print("Total questions:", report["total"])
    print("Evaluated questions:", report["evaluated"])
    print("Skipped - no gold pages:", report["no_gold"])
    print("Skipped - PDF not found:", report["missing_pdf"])
    print("Skipped - PDF failed:", report["failed_pdf"])

    for doc_name, error in report["errors"]:
        print(f"  {doc_name}: {error}")

    print("\nResults saved to:", EXPERIMENTS_DIR / f"{name}.csv")


# ==========================================
# MAIN
# ==========================================

def run_evaluation(settings: argparse.Namespace):

    # 1. Load questions
    print("\nLoading FinanceBench dataset...")
    records = load_financebench()
    print("Total questions:", len(records))

    # 2. Group questions by PDF and find the PDF files
    grouped = group_by_document(records)
    pdf_map = build_pdf_map()
    print("Unique documents:", len(grouped))

    # 3. Load the models once
    print("\nLoading models...")
    model = load_embedding_model()
    reranker = load_reranker() if settings.rerank else None

    methods = BASE_METHODS + (RERANK_METHODS if settings.rerank else [])

    all_rows = []

    # All counts are in QUESTIONS, so they add up to the total.
    report = {
        "total": len(records),
        "evaluated": 0,
        "no_gold": 0,
        "missing_pdf": 0,
        "failed_pdf": 0,
        "errors": []
    }

    # 4. Evaluate each PDF
    for number, (doc_name, questions) in enumerate(grouped.items(), start=1):

        print(f"\nDocument {number}/{len(grouped)}: {doc_name}")

        # 4a. Drop questions we cannot score
        scorable = [q for q in questions if get_gold_pages(q)]
        report["no_gold"] += len(questions) - len(scorable)

        if not scorable:
            continue

        # 4b. Skip PDFs we do not have
        pdf_path = pdf_map.get(doc_name)

        if pdf_path is None:
            print("PDF not found.")
            report["missing_pdf"] += len(scorable)
            continue

        # 4c. Search and score; keep rows only if the whole PDF worked
        try:
            rows = evaluate_document(
                doc_name, pdf_path, scorable, model, reranker, settings
            )

        except Exception as error:
            print("Document processing failed:", error)
            report["failed_pdf"] += len(scorable)
            report["errors"].append((doc_name, repr(error)))
            continue

        all_rows.extend(rows)
        report["evaluated"] += len(scorable)

        # 4d. Free this PDF's indexes before the next one
        gc.collect()

    # 5. Save and print
    summary = summarize(all_rows, methods)

    save_results(all_rows, settings.name)

    with open(EXPERIMENTS_DIR / f"{settings.name}_summary.json", "w") as file:
        json.dump({"settings": vars(settings), "report": report, "summary": summary},
                  file, indent=2)

    print_summary(summary, report, settings.name)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Evaluate retrieval on FinanceBench.")

    parser.add_argument("--name", default="baseline",
                        help="experiment name, used for the output files")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="characters per chunk; 0 = one chunk per page")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--rerank", action="store_true",
                        help="also evaluate FAISS and Hybrid with a reranker")

    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
