def chunk_text(text, chunk_words=150, overlap=30):
    assert 0 <= overlap < chunk_words
    words = text.split()
    step = chunk_words - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_words]))
        if i + chunk_words >= len(words):
            break
    return chunks