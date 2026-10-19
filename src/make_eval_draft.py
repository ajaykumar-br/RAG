"""Pick random chunks (one per video) to write evaluation questions about.

Run:  uv run python src/make_eval_draft.py --n 10 --seed 0
Prints the chunks, and writes data/eval/questions.txt with one "video_id chunk | " line each.
"""
import argparse
import json
import random
from pathlib import Path

INDEX = Path("data/index/chunks.jsonl")
OUT = Path("data/eval/questions.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(INDEX, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    by_video = {}
    for r in records:
        if len(r["text"].split()) >= 120:            # skip short tail fragments
            by_video.setdefault(r["video_id"], []).append(r)

    rng = random.Random(args.seed)
    picks = [rng.choice(by_video[v]) for v in rng.sample(sorted(by_video), args.n)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in picks:
            f.write(f"{r['video_id']} {r['chunk_index']} | \n")
            print(f"--- {r['video_id']} chunk {r['chunk_index']}\n{r['text']}\n")


if __name__ == "__main__":
    main()