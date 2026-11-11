"""
Turn text into vectors with a sentence-transformer model.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    QUERY_PREFIX
)


def load_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model (downloads it the first time).
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> np.ndarray:
    """
    Convert document chunks into normalized vectors.

    Returns an array with one row per text.
    """
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True
    )


def embed_query(
    query: str,
    model: SentenceTransformer
) -> np.ndarray:
    """
    Convert a search question into a normalized vector.

    BGE models were trained with an instruction prefix on
    queries, so we add it here (documents get no prefix).
    """
    return model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True
    )
