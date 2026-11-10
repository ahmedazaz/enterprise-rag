"""
Combine several rankings into one with Reciprocal Rank Fusion.
"""

from src.config import RRF_K


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = RRF_K
) -> list[tuple[int, float]]:
    """
    Merge ranked lists of chunk ids into one ranking.

    Each chunk gets 1 / (k + rank) from every list it appears in.
    Chunks ranked high by several methods end up on top.

    Returns a list of (chunk_id, rrf_score), best first.
    """

    scores = {}

    # 1. Add up the RRF score from every ranking
    for ranking in rankings:

        for rank, chunk_id in enumerate(ranking, start=1):

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)

    # 2. Sort by combined score, highest first
    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
