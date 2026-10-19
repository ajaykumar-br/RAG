"""Measure how often the right passage is retrieved.

Questions were written against 150-word chunks (EVAL_* in config.py). A retrieved chunk is a hit
if it overlaps that target passage enough, whatever chunk size the index uses.

Run:  uv run python src/evaluate.py
"""
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import CHUNK_WORDS, EVAL_CHUNK_WORDS, EVAL_OVERLAP, MODEL_NAME, OVERLAP
from search import load_index, search

QUESTIONS = Path("data/eval/questions.txt")
K = 10


def load_questions():
    items = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#") or "|" not in line:
            continue
        head, question = line.split("|", 1)
        question = question.strip()
        if not question:
            continue
        video_id, chunk_index = head.split()
        items.append((video_id, int(chunk_index), question))
    return items


def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_hit(r, video_id, chunk_index, strict=True):
    if r["video_id"] != video_id:
        return False
    t_start = chunk_index * (EVAL_CHUNK_WORDS - EVAL_OVERLAP)
    t_end = t_start + EVAL_CHUNK_WORDS
    shared = overlap(r["start_word"], r["end_word"], t_start, t_end)
    if strict:      # at least half of the smaller span is shared
        return shared >= 0.5 * min(r["end_word"] - r["start_word"], EVAL_CHUNK_WORDS)
    return shared >= 20     # loose: any real overlap (counts neighbouring chunks)


def rank_of(results, video_id, chunk_index, strict=True):
    for rank, (_, r) in enumerate(results, start=1):
        if is_hit(r, video_id, chunk_index, strict):
            return rank
    return None


def main():
    items = load_questions()
    model = SentenceTransformer(MODEL_NAME)
    embeddings, records = load_index()
    print(f"model {MODEL_NAME}, chunks {CHUNK_WORDS}/{OVERLAP} words, {len(records)} chunks\n")

    strict_ranks, loose_ranks = [], []
    print(f"{'strict':>6} {'loose':>6}  question")
    for video_id, chunk_index, question in items:
        results = search(question, model, embeddings, records, k=K)
        s = rank_of(results, video_id, chunk_index, strict=True)
        l = rank_of(results, video_id, chunk_index, strict=False)
        strict_ranks.append(s)
        loose_ranks.append(l)
        print(f"{s or '-':>6} {l or '-':>6}  {question}")

    n = len(items)
    print()
    for label, ranks in (("strict", strict_ranks), ("loose", loose_ranks)):
        h1 = sum(r == 1 for r in ranks)
        h5 = sum(r is not None and r <= 5 for r in ranks)
        hk = sum(r is not None for r in ranks)
        print(f"{label:>6}: hit@1 {h1}/{n}   hit@5 {h5}/{n}   hit@{K} {hk}/{n}")


if __name__ == "__main__":
    main()