"""
The full retrieval pipeline for one PDF:

    PDF -> pages -> chunks -> (FAISS index + BM25 index) -> search

Both search.py and evaluate.py use these two functions.
"""

from pathlib import Path

from src.config import TOP_K
from src.ingestion.chunker import chunk_pages
from src.ingestion.extractor import extract_text_from_pdf
from src.retrieval.bm25_search import build_bm25_index, search_bm25
from src.retrieval.embeddings import embed_documents, embed_query
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.vector_store import build_faiss_index, search_faiss


SEARCH_METHODS = ["FAISS", "BM25", "Hybrid_RRF"]


def build_document_index(pdf_path: Path, model) -> dict:
    """
    Read one PDF and build everything needed to search it.

    Returns a dict with:
        "pages"  - extracted pages
        "chunks" - text chunks with metadata
        "faiss"  - vector index over the chunks
        "bm25"   - keyword index over the chunks
    """

    # 1. Extract text from every page
    pages = extract_text_from_pdf(pdf_path)

    # 2. Split pages into chunks
    chunks = chunk_pages(pages)

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

    return {
        "pages": pages,
        "chunks": chunks,
        "faiss": faiss_index,
        "bm25": bm25_index
    }


def retrieve(
    question: str,
    document_index: dict,
    model,
    top_k: int = TOP_K
) -> dict:
    """
    Search one document with all three methods.

    Returns {method_name: [(chunk_id, score), ...]}, best first.
    """

    # 1. Vector search
    faiss_results = search_faiss(
        embed_query(question, model),
        document_index["faiss"],
        top_k
    )

    # 2. Keyword search
    bm25_results = search_bm25(
        question,
        document_index["bm25"],
        top_k
    )

    # 3. Hybrid: merge both rankings with RRF
    hybrid_results = reciprocal_rank_fusion([
        [chunk_id for chunk_id, _ in faiss_results],
        [chunk_id for chunk_id, _ in bm25_results]
    ])[:top_k]

    return {
        "FAISS": faiss_results,
        "BM25": bm25_results,
        "Hybrid_RRF": hybrid_results
    }
