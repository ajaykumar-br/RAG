"""Ask a question; retrieve chunks; have the LLM answer from them with citations.

Run:  uv run python src/answer.py "your question"
      uv run python src/answer.py --show-prompt "your question"
"""
import argparse

from sentence_transformers import SentenceTransformer

from config import MODEL_NAME
from llm import complete
from search import load_index, search

SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. The sources are excerpts "
    "from YouTube video transcripts (spoken language, so they can be messy). "
    "Cite the sources you use like [1] or [2][3] right after each claim. "
    "If the excerpts do not contain the answer, say 'the retrieved excerpts don't contain this'. "
    "They are only a few passages from many videos, so their silence does not mean the videos never say it."
    "Do not use outside knowledge."
    "Keep the answer under about 150 words."
)


def build_prompt(question, hits):
    """hits: list of (score, record) pairs from search(). Return the user message for the LLM."""
    # TODO: build a string that lists each source as
    #   [1] (video <video_id>, chunk <chunk_index>)
    #   <chunk text>
    # with a blank line between sources, then the question at the end, e.g. "Question: ..."
    parts = []
    for i, (_, r) in enumerate(hits, start=1):
        parts.append(f"[{i}] (video {r['video_id']}, chunk {r['chunk_index']})\n{r['text']}")
    sources = "\n\n".join(parts)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--show-prompt", action="store_true")
    args = p.parse_args()
    question = " ".join(args.question)

    model = SentenceTransformer(MODEL_NAME)
    embeddings, records = load_index()
    hits = search(question, model, embeddings, records, k=args.k)
    prompt = build_prompt(question, hits)
    if args.show_prompt:
        print(prompt, "\n" + "=" * 80)

    MAX_TOKENS = 1000      # near the top, below SYSTEM

    # in main(), use it and warn if it still hits the limit:
    text, usage = complete(prompt, system=SYSTEM, max_tokens=MAX_TOKENS)
    print(text)
    if usage.get("completion_tokens", 0) >= MAX_TOKENS:
        print("\n(warning: the answer hit the token limit and was cut off)")

    print("\nSources:")
    for i, (score, r) in enumerate(hits, start=1):
        print(f"  [{i}] {r['video_id']} chunk {r['chunk_index']}  (score {score:.3f})")
    print(f"\ntokens: {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out;  "
          f"cost ${usage.get('cost', 0):.5f}")


if __name__ == "__main__":
    main()