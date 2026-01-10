import chromadb
from .chunking import chunk_text

def get_collection(store_dir: str, collection: str):
    client = chromadb.PersistentClient(path=store_dir)
    return client.get_or_create_collection(name=collection)

def build_chroma_index(col, raw_docs, chunk_size=900, chunk_overlap=150):
    ids, docs, metas = [], [], []
    for d in raw_docs:
        chunks = chunk_text(d["text"], chunk_size=chunk_size, overlap=chunk_overlap)
        for j, ch in enumerate(chunks):
            ids.append(f"{d['source']}::{d['title']}::chunk{j}")
            docs.append(ch)
            metas.append({"source": d["source"], "title": d["title"]})

    if not docs:
        raise RuntimeError("Tidak ada dokumen/chunk yang bisa di-index.")

    # Tambahkan ke collection
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)

def query_chroma(col, query_text: str, n_results=6):
    return col.query(query_texts=[query_text], n_results=n_results)

def format_context(res):
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    lines = []
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        lines.append(f"[{i}] {m.get('title')} ({m.get('source')})\n{d}\n")
    return "\n".join(lines).strip()
