"""
Re-order search results with a cross-encoder.

FAISS compares a question vector with chunk vectors that were made
separately. A cross-encoder reads the question and the chunk
together, so it judges relevance more accurately, but it is much
slower. So we use it only on the top results of a fast search.
"""

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL


def load_reranker() -> CrossEncoder:
    """
    Load the cross-encoder (downloads it the first time).
    """
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def rerank(
    question: str,
    chunk_ids: list[int],
    chunks: list[dict],
    reranker: CrossEncoder
) -> list[tuple[int, float]]:
    """
    Score each chunk against the question and sort best first.

    Returns a list of (chunk_id, score).
    """

    if not chunk_ids:
        return []

    # 1. Make (question, chunk text) pairs
    pairs = [
        (question, chunks[chunk_id]["text"])
        for chunk_id in chunk_ids
    ]

    # 2. Score every pair
    scores = reranker.predict(pairs, batch_size=16)

    # 3. Sort chunks by score, highest first
    return sorted(
        zip(chunk_ids, map(float, scores)),
        key=lambda item: item[1],
        reverse=True
    )
