import json, random, re
from pathlib import Path
from config import MODEL_NAME, EVAL_CHUNK_WORDS, EVAL_OVERLAP
from llm import complete

N, SEED = 60, 0
OUT = Path("data/eval/generated.jsonl")   # separate from your hand-written questions.txt
CHUNKS = Path("data/index") / f"{MODEL_NAME}_c{EVAL_CHUNK_WORDS}_o{EVAL_OVERLAP}" / "chunks.jsonl"

SYSTEM = """You write test questions for a search system over YouTube transcripts.
Given a passage, return ONLY a JSON object: {"question": "...", "answer_phrase": "..."}
Rules:
- The question must make sense to someone who has not seen the passage. Never say "the speaker", "this passage" or "the video".
- Ask about ONE specific fact stated in the passage.
- answer_phrase is 3-12 words copied EXACTLY from the passage, containing the answer.
- If the passage has no specific fact worth asking about, return {"skip": true}."""

def norm(s):
    return " ".join(re.sub(r"[^\w\s]", "", s.lower()).split())

def main():
    records = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if len(r["text"].split()) >= 120]
    random.Random(SEED).shuffle(records)

    per_video, picked = {}, []
    for r in records:
        if per_video.get(r["video_id"], 0) < 2:
            per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
            picked.append(r)
        if len(picked) == N:
            break

    stats = dict(kept=0, skipped=0, bad_json=0, phrase_missing=0)
    cost = 0.0
    with OUT.open("w", encoding="utf-8") as f:
        for r in picked:
            text, usage = complete(f"Passage:\n{r['text']}", system=SYSTEM, max_tokens=200)
            cost += usage.get("cost", 0) or 0
            text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
            try:
                d = json.loads(text)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if d.get("skip"):
                stats["skipped"] += 1
                continue
            if norm(d.get("answer_phrase", "")) not in norm(r["text"]):
                stats["phrase_missing"] += 1
                continue
            row = {"video_id": r["video_id"], "chunk_index": r["chunk_index"],
                   "question": d["question"], "answer_phrase": d["answer_phrase"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    print(stats, f"cost ${cost:.4f}")

if __name__ == "__main__":
    main()