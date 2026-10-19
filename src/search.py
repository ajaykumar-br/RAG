"""Ask a question, get the top-k most similar chunks.

Run:  uv run python search.py "your question here"
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from cosine_similarity import cosine_similarity

from config import MODEL_NAME, INDEX_DIR


def load_index():
    embeddings = torch.from_numpy(np.load(INDEX_DIR / "embeddings.npy"))        # (N, 384)
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return embeddings, records


def search(query, model, embeddings, records, k=5):
    """Return the k best records as (score, record) pairs, best first."""
    # TODO 1: embed the query -> shape (384,). Use the SAME model and the SAME
    #         normalize_embeddings setting as build_index.py, then torch.from_numpy(...)
    # TODO 2: scores = your cosine_similarity(q, embeddings)   -> shape (N,)
    # TODO 3: torch.topk(scores, k) gives values and indices; return the matching records
    
    q = torch.from_numpy(model.encode(query, normalize_embeddings=True))
    scores = cosine_similarity(q, embeddings)
    top = torch.topk(scores, k)
    
    return [(s.item(), records[i.item()]) for s, i in zip(top.values, top.indices)]


def main():
    query = " ".join(sys.argv[1:])
    model = SentenceTransformer(MODEL_NAME)
    embeddings, records = load_index()
    for score, r in search(query, model, embeddings, records):
        print(f"{score:.3f}  {r['video_id']}  chunk {r['chunk_index']}")
        print("   ", r["text"][:300], "...\n")


if __name__ == "__main__":
    main()