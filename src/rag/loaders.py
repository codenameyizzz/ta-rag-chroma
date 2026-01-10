import os
import pandas as pd
from pypdf import PdfReader

def load_docs_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    docs = []

    # Sesuaikan nama kolom dataset kamu di sini
    for _, r in df.iterrows():
        place = str(r.get("place_name", "")).strip()
        cat = str(r.get("category", "")).strip()
        loc = str(r.get("location", "")).strip()
        desc = str(r.get("description", "")).strip()
        tags = str(r.get("tags", "")).strip()
        reviews = str(r.get("reviews_summary", "")).strip()

        text = (
            f"Nama: {place}\nKategori: {cat}\nLokasi: {loc}\n"
            f"Deskripsi: {desc}\nTag: {tags}\nRingkasan Review: {reviews}"
        ).strip()

        if text and place:
            docs.append({"source": csv_path, "title": place, "text": text})

    return docs

def load_docs_from_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    docs = []
    for i, p in enumerate(reader.pages):
        t = (p.extract_text() or "").strip()
        if t:
            docs.append({"source": pdf_path, "title": f"page_{i+1}", "text": t})
    return docs

def scan_data_dir(data_dir: str):
    inputs = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.lower().endswith(".csv"):
                inputs.extend(load_docs_from_csv(path))
            elif fn.lower().endswith(".pdf"):
                inputs.extend(load_docs_from_pdf(path))
    return inputs
