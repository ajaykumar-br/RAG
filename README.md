# RAG over 40 YouTube transcripts

Ask a question about a collection of 40 YouTube-channel transcripts and get an answer built only from the matching passages, with `[n]` citations pointing at the source video and chunk.

## How it works

```
clean transcripts -> chunk (150 words, 30 overlap) -> embed (multi-qa-MiniLM)  -> save index      (run once)

question -> embed -> top-20 by cosine -> cross-encoder rerank -> top-5 -> prompt -> LLM -> cited answer
```

| Stage | Choice |
|---|---|
| Corpus | 40 transcripts, about 317k words, 2,646 chunks |
| Chunking | 150 words, 30 words of overlap |
| Embeddings | `multi-qa-MiniLM-L6-cos-v1` (384 dimensions, normalized, so dot product equals cosine) |
| Retrieval | cosine similarity over all chunks, top 20 |
| Reranking | cross-encoder `ms-marco-MiniLM-L6-v2` rescored the 20, keep top 5 |
| Generation | Anthropic Sonnet through OpenRouter, temperature 0, answer capped near 150 words |

Cost is about $0.006 per question. Reranking runs locally and adds no API cost.

## Results

Evaluated on 50 questions that an LLM wrote from random chunks (at most 2 per video). A question counts as a hit when a retrieved chunk contains the exact answer phrase.

| Setup | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| Embeddings only | 52% | 82% | 84% |
| Embeddings, then rerank 20 candidates | 80% | 90% | 92% |

- Reranking fixed ordering: the right chunk was often retrieved at ranks 2 to 5 but not first.
- Raising k alone barely helped (hit@5 to hit@10 gained 2 questions).
- The earlier hand-written set (8 questions) gave the same picture and was used to compare embedding models and chunk sizes. `multi-qa-MiniLM` was slightly ahead of `all-MiniLM-L6-v2` at the top ranks, but 8 questions is too few to call it conclusive. Smaller chunks (60 words) did not help.

**Caveats.** The questions were generated from the chunks, so their wording is probably closer to the answer text than real questions would be, and the real-world gain is likely smaller. With 50 questions, one question is worth 2 points and the margin of error is roughly 10 points, so only large differences mean anything.

## Failure types seen

1. **Numbers and rare or garbled terms.** Embeddings match meaning, not exact tokens. "Noida" was transcribed as "noa", and questions about "5 LPA", "50k to 60k" or "TCS hired three" are still missed after reranking. Likely fix: hybrid search that adds keyword matching.
2. **Chunk dilution.** When the answer is a small part of a long chunk, one vector averages it with other topics. Example: the "monetary part first" advice sat in the middle third of a chunk that also covered LinkedIn offers and burnout, and plain embedding search missed it. The reranker fixed it by reading question and chunk together. Other options: smaller chunks, or retrieve small and return the neighbours.
3. **Answers that need several chunks.** "How many Super 30 students have been placed in total?" needs figures from two videos. Top-5 and top-10 embedding search each missed one. With reranking both appear in the top 5, and the model reports each figure with its source instead of adding them, because the excerpts do not say whether the batches overlap.

## Run it

```bash
uv run python src/build_index.py                      # once; embeds all chunks
uv run python src/answer.py "your question"           # add --k 10, --show-prompt as needed
uv run python src/evaluate_generated.py               # retrieval metrics on the generated set
uv run python src/make_eval_generated.py              # rebuild the generated questions (about $0.10)
```

Put `OPENROUTER_API_KEY=...` in a git-ignored `.env`. Optional: `LLM_MODEL` picks the model, and `RAG_MODEL`, `RAG_CHUNK_WORDS`, `RAG_OVERLAP` change the embedding model and chunking (each combination gets its own index folder).

## Files

| File | Role |
|---|---|
| `src/config.py` | model name, chunk size, overlap, index path, frozen settings for the eval set |
| `src/chunking.py` | `chunk_text()` |
| `src/similarity.py` | `cosine_similarity()` |
| `src/build_index.py` | chunk, embed and save `embeddings.npy` and `chunks.jsonl` |
| `src/search.py` | load the index; `search()` returns the top k chunks |
| `src/rerank.py` | cross-encoder reranking |
| `src/llm.py` | OpenRouter client |
| `src/answer.py` | the full question-to-cited-answer path |
| `src/evaluate.py`, `src/inspect_misses.py` | evaluation on the 8 hand-written questions |
| `src/make_eval_generated.py`, `src/evaluate_generated.py` | LLM-written questions and answer-phrase scoring |
| `data/transcripts/` | cleaned transcripts, one file per video ID |
| `data/eval/` | question sets |

## Not built yet

- Hybrid keyword plus embedding search, for numbers and rare terms
- Query rewriting or multiple queries per question
- A score threshold so the system says "I don't know" without relying on the prompt alone
- A test for unfaithful answers (so far the model has refused or hedged correctly in every run, but that is not a measured rate)
- Optional: a LoRA fine-tune