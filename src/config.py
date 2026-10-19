import os
from pathlib import Path

MODEL_NAME = os.environ.get("RAG_MODEL", "multi-qa-MiniLM-L6-cos-v1")
CHUNK_WORDS = int(os.environ.get("RAG_CHUNK_WORDS", 150))
OVERLAP = int(os.environ.get("RAG_OVERLAP", 30))
INDEX_DIR = Path("data/index") / f"{MODEL_NAME}_c{CHUNK_WORDS}_o{OVERLAP}"

# The eval questions were written against these chunks. Never change these two.
EVAL_CHUNK_WORDS = 150
EVAL_OVERLAP = 30