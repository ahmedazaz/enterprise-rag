"""
Vector (semantic) search with FAISS.
"""

import faiss
import numpy as np


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for cosine similarity.

    The vectors are already normalized, so inner product
    (IndexFlatIP) is the same as cosine similarity.
    """

    # 1. FAISS needs float32
    vectors = np.asarray(vectors, dtype="float32")

    # 2. Create an empty index with the right vector size
    index = faiss.IndexFlatIP(vectors.shape[1])

    # 3. Add all chunk vectors
    index.add(vectors)

    return index


def search_faiss(
    query_vector: np.ndarray,
    index: faiss.Index,
    top_k: int
) -> list[tuple[int, float]]:
    """
    Find the chunks most similar to the query.

    Returns a list of (chunk_id, score), best first.
    """

    # 1. Never ask for more results than there are chunks
    top_k = min(top_k, index.ntotal)

    # 2. FAISS expects a 2D array: one row per query
    query_matrix = np.asarray([query_vector], dtype="float32")

    # 3. Search
    scores, ids = index.search(query_matrix, top_k)

    # 4. Keep valid results (FAISS uses -1 for "no result")
    return [
        (int(chunk_id), float(score))
        for chunk_id, score in zip(ids[0], scores[0])
        if chunk_id >= 0
    ]
