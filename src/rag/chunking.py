def chunk_text(text: str, chunk_size=900, overlap=150):
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end == len(text):
            break
    return chunks
