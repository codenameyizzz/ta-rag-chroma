import os
import pandas as pd
from pypdf import PdfReader
import chromadb

OUT_DIR = "chroma_store"
COLLECTION = "toba_places"

def chunk_text(text: str, chunk_size=900, overlap=150):
    text = " ".join((text or "").split())
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end == len(text):
            break
    return chunks

def load_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    docs = []
    for _, r in df.iterrows():
        place = str(r.get("place_name", "")).strip()
        cat = str(r.get("category", "")).strip()
        loc = str(r.get("location", "")).strip()
        desc = str(r.get("description", "")).strip()
        tags = str(r.get("tags", "")).strip()
        reviews = str(r.get("reviews_summary", "")).strip()
        text = f"Nama: {place}\nKategori: {cat}\nLokasi: {loc}\nDeskripsi: {desc}\nTag: {tags}\nRingkasan Review: {reviews}".strip()
        if text:
            docs.append({"source": csv_path, "title": place or "row", "text": text})
    return docs

def load_from_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    docs = []
    for i, p in enumerate(reader.pages):
        t = (p.extract_text() or "").strip()
        if t:
            docs.append({"source": pdf_path, "title": f"page_{i+1}", "text": t})
    return docs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=OUT_DIR)
    col = client.get_or_create_collection(name=COLLECTION)

    # kumpulkan dokumen dari ./data
    inputs = []
    for root, _, files in os.walk("data"):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.lower().endswith(".csv"):
                inputs.extend(load_from_csv(path))
            elif fn.lower().endswith(".pdf"):
                inputs.extend(load_from_pdf(path))

    if not inputs:
        raise RuntimeError("Taruh CSV/PDF di folder ./data dulu.")

    ids, docs, metas = [], [], []
    n = 0
    for d in inputs:
        for j, ch in enumerate(chunk_text(d["text"])):
            ids.append(f"{d['source']}::{d['title']}::chunk{j}")
            docs.append(ch)
            metas.append({"source": d["source"], "title": d["title"]})
            n += 1

    # Chroma bisa auto-embed pakai default embedding function,
    # tapi untuk kontrol & kualitas, biasanya kita set embedding model sendiri.
    # Untuk tahap awal, kita pakai default dulu (biar jalan cepat).
    col.add(ids=ids, documents=docs, metadatas=metas)

    print(f"✅ Chroma collection '{COLLECTION}' berisi {n} chunks di {OUT_DIR}")

if __name__ == "__main__":
    main()
