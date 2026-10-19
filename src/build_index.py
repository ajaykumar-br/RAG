"""Chunk every transcript, embed the chunks, save the index.

Run:  uv run python build_index.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import chunk_text

from config import CHUNK_WORDS, INDEX_DIR, MODEL_NAME, OVERLAP

TRANSCRIPT_DIR = Path("data/transcripts")


def load_chunks():
    step = CHUNK_WORDS - OVERLAP
    records = []
    for path in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text, CHUNK_WORDS, OVERLAP)):
            start = i * step
            records.append({"video_id": path.stem, "chunk_index": i,
                            "start_word": start, "end_word": start + len(chunk.split()),
                            "text": chunk})
    return records


def main():
    records = load_chunks()
    n_videos = len({r["video_id"] for r in records})
    print(f"{len(records)} chunks from {n_videos} videos")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model reads at most {model.max_seq_length} tokens per chunk")

    # Does a 150-word chunk fit inside what the model can read?
    n_tokens = np.array([len(model.tokenizer.encode(r["text"])) for r in records])
    too_long = (n_tokens > model.max_seq_length).mean()
    print(f"Median tokens per chunk: {int(np.median(n_tokens))}   "
          f"max: {n_tokens.max()}   chunks over the limit: {too_long:.1%}")

    embeddings = model.encode([r["text"] for r in records], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True)
    print("Embeddings:", embeddings.shape, embeddings.dtype)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved index to {INDEX_DIR}/")


if __name__ == "__main__":
    main()