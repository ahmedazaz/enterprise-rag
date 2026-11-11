"""
All project settings in one place.

Change a value here and every script picks it up.
"""

from pathlib import Path


# ==========================================
# 1. FOLDERS AND FILES
# ==========================================

RAW_PDF_DIR = Path("data/raw")

PROCESSED_DIR = Path("data/processed")

FINANCEBENCH_QUESTIONS = PROCESSED_DIR / "financebench_open_source.jsonl"

EVALUATION_OUTPUT = PROCESSED_DIR / "retrieval_evaluation.csv"

# Each evaluation experiment saves its results here.
EXPERIMENTS_DIR = PROCESSED_DIR / "experiments"


# ==========================================
# 2. CHUNKING
# ==========================================

# The embedding model only reads the first 512 tokens of a text.
# 1600 characters is roughly 350-450 tokens, so the whole chunk
# fits inside the model's window and nothing is silently cut off.
CHUNK_SIZE = 1600

# How many characters each chunk shares with the previous one.
CHUNK_OVERLAP = 300


# ==========================================
# 3. EMBEDDINGS
# ==========================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models expect this prefix on QUERIES (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBEDDING_BATCH_SIZE = 32


# ==========================================
# 4. SEARCH
# ==========================================

# How many PAGES we score (hit@10, mrr@10, ...).
TOP_K = 10

# How many CHUNKS each method retrieves before we turn them into pages.
# Several chunks can come from the same page, so we fetch more chunks
# than pages, then keep the first TOP_K different pages.
CANDIDATES = 50

# Reciprocal Rank Fusion constant (60 is the standard value).
RRF_K = 60


# ==========================================
# 5. RERANKING
# ==========================================

# A cross-encoder reads the question and a chunk TOGETHER and scores
# how well the chunk answers it. Slower than FAISS, but more accurate,
# so we only use it to re-order the top CANDIDATES chunks.
RERANKER_MODEL = "BAAI/bge-reranker-base"


# ==========================================
# 6. ANSWER GENERATION (LLM)
# ==========================================

# Local model served by Ollama (free, nothing leaves this computer).
LLM_MODEL = "qwen3:4b-instruct"

# Ollama listens on this computer only.
OLLAMA_URL = "http://localhost:11434/api/generate"

# How many of the top pages the model reads before answering.
CONTEXT_PAGES = 5

# The model's reading window in tokens. 5 full pages are roughly
# 5,000-8,000 tokens, so 16k leaves room for the question and answer.
LLM_CONTEXT_TOKENS = 16384
