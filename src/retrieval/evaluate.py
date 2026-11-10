"""
Measure how well FAISS, BM25 and Hybrid search find the
evidence pages for the FinanceBench questions.

Usage:
    python -m src.retrieval.evaluate

Steps:
    1. Load the FinanceBench questions
    2. Group the questions by PDF
    3. For each PDF: build indexes, search every question, score it
    4. Save one CSV row per (question, method)
    5. Print the average score of each method
"""

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

from src.config import (
    EVALUATION_OUTPUT,
    FINANCEBENCH_QUESTIONS,
    RAW_PDF_DIR,
    TOP_K
)
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.pipeline import (
    SEARCH_METHODS,
    build_document_index,
    retrieve
)


CSV_COLUMNS = [
    "financebench_id",
    "company",
    "doc_name",
    "method",
    "gold_pages",
    "ranked_pages",
    "first_relevant_rank",
    "mrr_at_10",
    "hit_at_5",
    "recall_at_5",
    "hit_at_10",
    "recall_at_10"
]

SUMMARY_METRICS = [
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
# STEP 3b: SCORE ONE RANKING
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

    # 3. Hit rate and recall at 5 and 10
    for k in (5, 10):

        found = set(ranked_pages[:k]) & gold_pages

        metrics[f"hit_at_{k}"] = int(len(found) > 0)
        metrics[f"recall_at_{k}"] = len(found) / len(gold_pages)

    return metrics


# ==========================================
# STEP 3c: EVALUATE ONE PDF
# ==========================================

def evaluate_document(
    doc_name: str,
    pdf_path: Path,
    questions: list[dict],
    model
) -> list[dict]:
    """
    Build the indexes for one PDF and score every question on it.

    Returns the CSV rows for this document. If anything fails,
    an exception is raised and no rows are returned, so the
    results never contain half of a document.
    """

    # 1. Build FAISS + BM25 indexes for this PDF
    document_index = build_document_index(pdf_path, model)
    chunks = document_index["chunks"]

    print("Pages:", len(document_index["pages"]))
    print("Chunks:", len(chunks))

    rows = []

    # 2. Search and score every question
    for record in questions:

        gold_pages = get_gold_pages(record)

        results = retrieve(record["question"], document_index, model)

        for method in SEARCH_METHODS:

            # 3. Convert chunk ids to page numbers
            ranked_pages = [
                chunks[chunk_id]["metadata"]["page"]
                for chunk_id, _ in results[method]
            ]

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

def save_results(rows: list[dict]):

    EVALUATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(EVALUATION_OUTPUT, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# STEP 5: PRINT SUMMARY
# ==========================================

def print_summary(rows: list[dict], report: dict):

    print("\n========== EVALUATION SUMMARY ==========")

    for method in SEARCH_METHODS:

        method_rows = [row for row in rows if row["method"] == method]

        if not method_rows:
            continue

        print(f"\nMethod: {method}")
        print("Evaluated questions:", len(method_rows))

        for metric in SUMMARY_METRICS:

            average = sum(row[metric] for row in method_rows) / len(method_rows)

            print(f"{metric}: {average:.4f} ({average * 100:.2f}%)")

    print("\n========== PROCESSING REPORT ==========\n")

    print("Total questions:", report["total"])
    print("Evaluated questions:", report["evaluated"])
    print("Skipped - no gold pages:", report["no_gold"])
    print("Skipped - PDF not found:", report["missing_pdf"])
    print("Skipped - PDF failed:", report["failed_pdf"])

    for doc_name, error in report["errors"]:
        print(f"  {doc_name}: {error}")

    print("\nDetailed results saved to:", EVALUATION_OUTPUT)


# ==========================================
# MAIN
# ==========================================

def run_evaluation():

    # 1. Load questions
    print("\nLoading FinanceBench dataset...")
    records = load_financebench()
    print("Total questions:", len(records))

    # 2. Group questions by PDF and find the PDF files
    grouped = group_by_document(records)
    pdf_map = build_pdf_map()
    print("Unique documents:", len(grouped))

    # 3. Load the embedding model once
    print("\nLoading embedding model...")
    model = load_embedding_model()

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
            rows = evaluate_document(doc_name, pdf_path, scorable, model)

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
    save_results(all_rows)
    print_summary(all_rows, report)


if __name__ == "__main__":
    run_evaluation()
