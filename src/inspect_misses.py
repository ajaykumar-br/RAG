"""For each question whose target chunk was not retrieved, show the target next to the top result.

Run:  uv run python src/inspect_misses.py
"""
from sentence_transformers import SentenceTransformer

from evaluate import K, load_questions, rank_of
from search import MODEL_NAME, load_index, search


def main():
    model = SentenceTransformer(MODEL_NAME)
    embeddings, records = load_index()
    text_of = {(r["video_id"], r["chunk_index"]): r["text"] for r in records}

    for video_id, chunk_index, question in load_questions():
        results = search(question, model, embeddings, records, k=K)
        if rank_of(results, video_id, chunk_index) is not None:
            continue
        score, top = results[0]
        print("=" * 90)
        print(f"QUESTION: {question}")
        print(f"\nTARGET  ({video_id} chunk {chunk_index}):\n{text_of[(video_id, chunk_index)]}")
        print(f"\nTOP-1   ({top['video_id']} chunk {top['chunk_index']}, score {score:.3f}):\n{top['text']}\n")


if __name__ == "__main__":
    main()