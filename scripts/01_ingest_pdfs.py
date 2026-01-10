import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

# ====== KONFIG ======
DATA_DIR = Path("data/raw/pdfs")
CHROMA_DIR = "stores/chroma"
COLLECTION_NAME = "danau_toba_pdfs"

# Chunking (karakter-based, simple & stabil)
CHUNK_SIZE = 1200
OVERLAP = 200

# Embedding model (kita kontrol sendiri)
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Kalau embedding ini berat / lambat di mesin kamu, ganti ke:
# EMBED_MODEL = "intfloat/multilingual-e5-base"

BATCH_SIZE = 32


def chunk_text(text: str, chunk_size: int, overlap: int):
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        t = doc[i].get_text("text")
        if t and t.strip():
            pages.append((i + 1, t))
    return pages


def main():
    if not DATA_DIR.exists():
        raise RuntimeError(f"Folder tidak ditemukan: {DATA_DIR.resolve()}")

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"Tidak ada PDF di {DATA_DIR.resolve()}")

    # --- Chroma persistent client ---
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    # --- Embedding model ---
    embedder = SentenceTransformer(EMBED_MODEL)

    # Kumpulkan semua chunks
    all_ids = []
    all_texts = []
    all_metas = []

    for pdf in tqdm(pdfs, desc="Extract PDFs"):
        pages = extract_pdf(pdf)
        for page_no, page_text in pages:
            chunks = chunk_text(page_text, CHUNK_SIZE, OVERLAP)
            for c_idx, ch in enumerate(chunks):
                doc_id = f"{pdf.name}::p{page_no}::c{c_idx}::{uuid.uuid4().hex[:8]}"
                all_ids.append(doc_id)
                all_texts.append(ch)
                all_metas.append({
                    "source_file": pdf.name,
                    "page": page_no,
                    "chunk": c_idx
                })

    print(f"Total chunks: {len(all_texts)}")

    # Embed + add ke Chroma dalam batch
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Embed+Upsert"):
        batch_texts = all_texts[i:i+BATCH_SIZE]
        batch_ids = all_ids[i:i+BATCH_SIZE]
        batch_metas = all_metas[i:i+BATCH_SIZE]

        # normalize_embeddings=True penting untuk cosine-ish retrieval
        batch_emb = embedder.encode(batch_texts, normalize_embeddings=True)
        batch_emb = np.asarray(batch_emb, dtype=np.float32).tolist()

        col.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=batch_emb
        )

    print("\n✅ Selesai ingest ke ChromaDB.")
    print(f"Chroma path: {Path(CHROMA_DIR).resolve()}")
    print(f"Collection: {COLLECTION_NAME}")
    print("Catatan: stores/chroma sebaiknya tidak di-commit ke git.")


if __name__ == "__main__":
    main()