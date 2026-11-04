from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

def load_reranker():
    return CrossEncoder(RERANK_MODEL)

def rerank(query, hits, reranker, top_n=5):
    # hits: list of (score, record) from search()
    pairs = [(query, r["text"]) for _, r in hits]
    scores = reranker.predict(pairs)    # one relevance score per pair
    new = [(float(s), r) for s, (_, r) in zip(scores, hits)]
    new.sort(key=lambda x: x[0], reverse=True)
    return new[:top_n]