"""
Keyword search with BM25.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Split text into lowercase words and numbers.
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index from the chunk texts.
    """

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def search_bm25(
    query: str,
    bm25: BM25Okapi,
    top_k: int
) -> list[tuple[int, float]]:
    """
    Find the chunks that best match the query words.

    Returns a list of (chunk_id, score), best first.
    """

    # 1. Score every chunk against the query words
    scores = bm25.get_scores(tokenize(query))

    # 2. Sort chunk ids from highest to lowest score.
    #    We sort by -score with a stable sort, so chunks with
    #    equal scores stay in document order.
    ranked_ids = np.argsort(-scores, kind="stable")[:top_k]

    # 3. Return ids together with their scores
    return [
        (int(chunk_id), float(scores[chunk_id]))
        for chunk_id in ranked_ids
    ]
