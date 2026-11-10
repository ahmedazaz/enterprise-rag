# Enterprise RAG: a learning project

Building a Retrieval-Augmented Generation (RAG) system from scratch,
one notebook at a time, on real financial reports (FinanceBench).

## Lessons

| # | Notebook | Topic | Status |
|---|---|---|---|
| 1 | [01_read_pdf.ipynb](notebooks/01_read_pdf.ipynb) | Extract text from a PDF | In progress |
| 2 | | Split text into chunks | |
| 3 | | Keyword search (BM25) | |
| 4 | | What is an embedding? | |
| 5 | | Semantic search (FAISS) | |
| 6 | | Measure retrieval quality | |
| 7 | | Hybrid search | |
| 8 | | Generate answers with an LLM | |
| 9 | | LLM-as-a-Judge | |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ipykernel
```

## Data (not in this repo, too large)

- PDFs: clone [FinanceBench](https://github.com/patronus-ai/financebench)
  and copy `pdfs/*.pdf` into `data/raw/`.
- Questions: copy `data/financebench_open_source.jsonl` into `data/processed/`.

## Earlier version

The first version of the pipeline (before the rewrite) is kept on the
[`old-version`](../../tree/old-version) branch.
