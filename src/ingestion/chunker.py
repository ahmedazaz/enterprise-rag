"""
Split page text into overlapping chunks.
"""

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Split every page into chunks of `chunk_size` characters.

    Chunks never cross page boundaries, so every chunk
    belongs to exactly one page number.
    """

    # 1. Check the settings make sense
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than "
            f"chunk_size ({chunk_size})"
        )

    step = chunk_size - overlap

    chunks = []

    # 2. Go through every page
    for page in pages:

        text = page["text"]

        # 3. Skip pages with no text
        if not text.strip():
            continue

        # 4. Cut the page into chunks
        start = 0

        while True:

            end = min(start + chunk_size, len(text))

            chunks.append({
                "text": text[start:end],
                "metadata": {
                    **page["metadata"],
                    "chunk_start": start,
                    "chunk_end": end
                }
            })

            # 5. Stop once this chunk reached the end of the page.
            #    (Without this check, the last chunk would be
            #    repeated inside an extra, smaller chunk.)
            if end == len(text):
                break

            start += step

    return chunks
