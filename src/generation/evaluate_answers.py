"""
Answer every FinanceBench question with the full RAG system, then let
the judge grade each answer.

Usage:
    python -m src.generation.evaluate_answers --limit 3    # quick test
    python -m src.generation.evaluate_answers              # all questions

Steps:
    1. Load the questions and group them by PDF
    2. For each PDF: build the search indexes once
    3. For each question: answer it, then judge the answer
    4. Save one CSV row per question (written as we go)
    5. Print the accuracy
"""

import argparse
import csv
import gc
from collections import defaultdict

from src.config import EXPERIMENTS_DIR
from src.generation.answer import answer_question
from src.generation.judge import judge_answer
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.evaluate import (
    build_pdf_map,
    get_gold_pages,
    group_by_document,
    load_financebench
)
from src.retrieval.pipeline import build_document_index


CSV_COLUMNS = [
    "financebench_id",
    "company",
    "doc_name",
    "question_type",
    "question",
    "gold_answer",
    "system_answer",
    "gold_pages",
    "pages_read",
    "page_found",
    "verdict",
    "judge_reason"
]


# ==========================================
# STEP 3: ANSWER AND JUDGE ONE QUESTION
# ==========================================

def evaluate_question(record: dict, doc_name: str, document_index: dict, model) -> dict:
    """
    Answer one question with the RAG system and grade the answer.
    """

    # 1. The RAG system answers
    result = answer_question(record["question"], document_index, model)

    # 2. The judge grades the answer
    grade = judge_answer(record["question"], record["answer"], result["answer"])

    # 3. Did the LLM even see a correct page? (helps explain mistakes)
    gold_pages = get_gold_pages(record)
    page_found = bool(gold_pages & set(result["pages"]))

    return {
        "financebench_id": record["financebench_id"],
        "company": record["company"],
        "doc_name": doc_name,
        "question_type": record["question_type"],
        "question": record["question"],
        "gold_answer": record["answer"],
        "system_answer": result["answer"],
        "gold_pages": ";".join(map(str, sorted(gold_pages))),
        "pages_read": ";".join(map(str, result["pages"])),
        "page_found": int(page_found),
        "verdict": grade["verdict"],
        "judge_reason": grade["reason"]
    }


# ==========================================
# STEP 5: PRINT THE ACCURACY
# ==========================================

def print_accuracy(rows: list[dict]):

    def accuracy(subset):
        correct = sum(row["verdict"] == "CORRECT" for row in subset)
        return f"{correct}/{len(subset)} = {correct / len(subset):.1%}" if subset else "-"

    print("\n========== ANSWER ACCURACY ==========\n")
    print("All questions:", accuracy(rows))

    # 1. By question type
    print("\nBy question type:")
    by_type = defaultdict(list)

    for row in rows:
        by_type[row["question_type"]].append(row)

    for question_type, subset in sorted(by_type.items()):
        print(f"  {question_type:<20} {accuracy(subset)}")

    # 2. Split by whether search found a correct page.
    #    This shows if mistakes come from SEARCH or from the LLM.
    print("\nBy search result:")
    print("  correct page was read     ", accuracy([r for r in rows if r["page_found"]]))
    print("  correct page was NOT read ", accuracy([r for r in rows if not r["page_found"]]))

    unknown = sum(row["verdict"] == "UNKNOWN" for row in rows)
    if unknown:
        print(f"\nJudge replies that could not be read: {unknown}")


# ==========================================
# MAIN
# ==========================================

def run(settings: argparse.Namespace):

    # 1. Load questions, grouped by PDF
    records = load_financebench()[: settings.limit or None]
    grouped = group_by_document(records)
    pdf_map = build_pdf_map()

    print(f"Questions: {len(records)} | Documents: {len(grouped)}")

    model = load_embedding_model()

    output = EXPERIMENTS_DIR / f"{settings.name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    with open(output, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for number, (doc_name, questions) in enumerate(grouped.items(), start=1):

            print(f"\nDocument {number}/{len(grouped)}: {doc_name}")

            pdf_path = pdf_map.get(doc_name)

            if pdf_path is None:
                print("PDF not found, skipped.")
                continue

            # 2. Build the search indexes once per PDF
            document_index = build_document_index(pdf_path, model)

            # 3. Answer and judge every question
            for record in questions:

                row = evaluate_question(record, doc_name, document_index, model)
                rows.append(row)

                # 4. Save each row right away, so progress survives a crash
                writer.writerow(row)
                file.flush()

                print(f"  {row['verdict']:<9} {row['financebench_id']}")

            gc.collect()

    print_accuracy(rows)
    print("\nResults saved to:", output)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Answer and judge FinanceBench questions.")

    parser.add_argument("--name", default="answers",
                        help="output file name (in the experiments folder)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only use the first N questions (0 = all)")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
