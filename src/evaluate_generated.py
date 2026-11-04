import json, re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, CHUNK_WORDS, OVERLAP
from search import load_index, search

K = 10

def norm(s):
    return " ".join(re.sub(r"[^\w\s]", "", s.lower()).split())

def main():
    rows = [json.loads(l) for l in Path("data/eval/generated.jsonl").read_text(encoding="utf-8").splitlines()]
    model = SentenceTransformer(MODEL_NAME)
    embeddings, records = load_index()
    ranks = []
    for row in rows:
        results = search(row["question"], model, embeddings, records, k=K)
        phrase = norm(row["answer_phrase"])
        rank = next((i for i, (_, r) in enumerate(results, 1) if phrase in norm(r["text"])), None)
        ranks.append(rank)
        if rank is None:
            print(f"MISS: {row['question']}\n   target: {row['video_id']} chunk {row['chunk_index']} | answer: \"{row['answer_phrase']}\"")
    n = len(rows)
    for k in (1, 5, 10):
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"hit@{k}: {hits}/{n} = {hits / n:.0%}")

if __name__ == "__main__":
    main()