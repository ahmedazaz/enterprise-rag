"""
The full retrieval pipeline for one PDF:

    PDF -> pages -> chunks -> (FAISS index + BM25 index) -> search
                                                         -> (optional) rerank

Both search.py and evaluate.py use these two functions.
"""

from pathlib import Path

from src.config import CANDIDATES, CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.chunker import chunk_pages
from src.ingestion.extractor import extract_text_from_pdf
from src.retrieval.bm25_search import build_bm25_index, search_bm25
from src.retrieval.embeddings import embed_documents, embed_query
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.statements import (
    boost_statement_chunks,
    detect_page_statement,
    detect_question_statements
)
from src.retrieval.vector_store import build_faiss_index, search_faiss


BASE_METHODS = ["FAISS", "BM25", "Hybrid_RRF", "FAISS+Statement"]

RERANK_METHODS = ["FAISS+Rerank", "Hybrid+Rerank"]


def build_document_index(
    pdf_path: Path,
    model,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> dict:
    """
    Read one PDF and build everything needed to search it.

    Returns a dict with:
        "pages"  - extracted pages
        "chunks" - text chunks with metadata
        "faiss"  - vector index over the chunks
        "bm25"   - keyword index over the chunks
        "page_statements" - {page number: "cash_flow" / "balance_sheet" / "income"}
                            for pages that show a financial statement
    """

    # 1. Extract text from every page
    pages = extract_text_from_pdf(pdf_path)

    # 2. Split pages into chunks
    chunks = chunk_pages(pages, chunk_size, overlap)

    if not chunks:
        raise ValueError(f"No text could be extracted from {pdf_path.name}")

    # 3. Embed every chunk and build the vector index
    vectors = embed_documents(
        [chunk["text"] for chunk in chunks],
        model
    )
    faiss_index = build_faiss_index(vectors)

    # 4. Build the keyword index
    bm25_index = build_bm25_index(chunks)

    # 5. Label pages that show a financial statement
    page_statements = {}

    for page in pages:

        statement = detect_page_statement(page["text"])

        if statement:
            page_statements[page["metadata"]["page"]] = statement

    return {
        "pages": pages,
        "chunks": chunks,
        "faiss": faiss_index,
        "bm25": bm25_index,
        "page_statements": page_statements
    }


def retrieve(
    question: str,
    document_index: dict,
    model,
    candidates: int = CANDIDATES,
    reranker=None
) -> dict:
    """
    Search one document with every method.

    Returns {method_name: [(chunk_id, score), ...]}, best first.
    Each list has up to `candidates` chunks.

    If a reranker is given, two extra methods are added that
    re-order the FAISS and Hybrid candidates with it.
    """

    chunks = document_index["chunks"]

    # 1. Vector search
    faiss_results = search_faiss(
        embed_query(question, model),
        document_index["faiss"],
        candidates
    )

    # 2. Keyword search
    bm25_results = search_bm25(
        question,
        document_index["bm25"],
        candidates
    )

    # 3. Hybrid: merge both rankings with RRF
    hybrid_results = reciprocal_rank_fusion([
        [chunk_id for chunk_id, _ in faiss_results],
        [chunk_id for chunk_id, _ in bm25_results]
    ])[:candidates]

    # 4. Statement boost: if the question names a financial statement
    #    (e.g. "cash flow statement"), move that statement's pages to the top
    wanted = detect_question_statements(question)

    if wanted:

        # Rank ALL chunks, so a statement page outside the top
        # candidates can still be moved up
        all_chunks_ranked = search_faiss(
            embed_query(question, model),
            document_index["faiss"],
            len(chunks)
        )

        statement_results = boost_statement_chunks(
            all_chunks_ranked,
            chunks,
            document_index["page_statements"],
            wanted
        )[:candidates]

    else:
        statement_results = faiss_results

    results = {
        "FAISS": faiss_results,
        "BM25": bm25_results,
        "Hybrid_RRF": hybrid_results,
        "FAISS+Statement": statement_results
    }

    # 5. Optional: re-order the candidates with the cross-encoder
    if reranker is not None:

        # 5a. Score every candidate chunk once
        #     (FAISS and Hybrid candidates mostly overlap)
        all_ids = list(dict.fromkeys(
            chunk_id
            for chunk_id, _ in faiss_results + hybrid_results
        ))
        rerank_scores = dict(rerank(question, all_ids, chunks, reranker))

        # 5b. Sort each method's candidates by the reranker score
        for method, source in (("FAISS+Rerank", faiss_results),
                               ("Hybrid+Rerank", hybrid_results)):

            results[method] = sorted(
                ((chunk_id, rerank_scores[chunk_id]) for chunk_id, _ in source),
                key=lambda item: item[1],
                reverse=True
            )

    return results
