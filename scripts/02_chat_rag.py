import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== KONFIG ======
CHROMA_DIR = "stores/chroma"
COLLECTION_NAME = "danau_toba_pdfs"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "Qwen/Qwen3-1.7B"

TOP_K = 6
MAX_CONTEXT_CHARS = 12000


def route_doc_types(q: str):
    ql = q.lower()
    if "itinerary" in ql or "hari" in ql or "malam" in ql:
        # itinerary butuh campuran kategori
        return ["wisata", "akomodasi", "cafe", "rumah_makan", "transportasi"]

    want = []
    if "wisata" in ql or "destinasi" in ql:
        want.append("wisata")
    if "hotel" in ql or "akomodasi" in ql or "penginapan" in ql:
        want.append("akomodasi")
    if "cafe" in ql:
        want.append("cafe")
    if "rumah makan" in ql or "restoran" in ql or "kuliner" in ql:
        want.append("rumah_makan")
    if "transport" in ql or "angkutan" in ql or "bus" in ql:
        want.append("transportasi")
    if "oleh" in ql or "oleh-oleh" in ql or "souvenir" in ql:
        want.append("oleh_oleh")
    if "fasum" in ql or "fasilitas" in ql:
        want.append("fasum")
    return want  # bisa kosong => fallback all

def detect_mode(q: str) -> str:
    ql = q.lower()
    itinerary_keywords = [
        "itinerary", "rencana perjalanan", "jadwal", "susun perjalanan",
        "hari", "malam", "2d", "3d", "4d", "day"
    ]
    if any(k in ql for k in itinerary_keywords):
        return "itinerary"
    return "recommendation"

SYSTEM_ITINERARY = (
    "Kamu adalah asisten perencana itinerary Danau Toba.\n"
    "ATURAN KETAT:\n"
    "1) Semua nama tempat/objek wisata/hotel/restoran/fasilitas yang kamu sebut harus ada di KONTEKS.\n"
    "2) Kamu boleh kreatif pada susunan itinerary, tapi fakta (nama, lokasi, fasilitas, harga) hanya dari KONTEKS.\n"
    "3) Jika KONTEKS tidak cukup, tulis 'data tidak tersedia' dan jelaskan kekurangannya.\n"
    "4) Jangan mengulang bullet/kalimat yang sama.\n"
    "FORMAT OUTPUT:\n"
    "- Judul itinerary\n"
    "- Hari 1 s.d Hari N (Pagi/Siang/Sore/Malam)\n"
    "- Catatan penting (maks 5 bullet)\n"
    "- Sumber: file + page\n"
)

SYSTEM_RECO = (
    "Kamu adalah asisten rekomendasi pariwisata Danau Toba.\n"
    "ATURAN:\n"
    "1) Semua nama tempat/objek wisata/hotel/restoran/fasilitas yang kamu sebut harus ada di KONTEKS.\n"
    "2) Boleh menulis narasi yang kreatif dan persuasif, tapi fakta (nama, lokasi, fasilitas, harga) hanya dari KONTEKS.\n"
    "3) Jangan memformat sebagai itinerary kecuali diminta.\n"
    "4) Jika KONTEKS tidak cukup, tulis 'data tidak tersedia'.\n"
    "FORMAT:\n"
    "- Beri 5–8 rekomendasi yang relevan (atau sesuai permintaan jumlah)\n"
    "- Tiap item: Nama — Lokasi — Alasan singkat\n"
    "- Tambahkan paragraf penutup yang ramah (opsional)\n"
    "- Sumber: file + page\n"
)


def dedup_docs(docs, metas, limit=10):
    out_docs, out_metas = [], []
    seen = set()
    for d, m in zip(docs, metas):
        key = (m.get("source_file"), m.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out_docs.append(d)
        out_metas.append(m)
        if len(out_docs) >= limit:
            break
    return out_docs, out_metas


def build_context(docs, metas):
    parts = []
    total = 0
    for d, m in zip(docs, metas):
        header = f"[{m.get('source_file')} | page {m.get('page')} | chunk {m.get('chunk')}]"
        block = f"{header}\n{d}".strip()
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def retrieve(col, q_emb, doc_types, top_k=TOP_K):
    docs, metas = [], []

    if not doc_types:
        res = col.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return docs, metas

    per_k = max(2, top_k // len(doc_types))
    for dt in doc_types:
        res = col.query(
            query_embeddings=q_emb,
            n_results=per_k,
            where={"doc_type": dt},
            include=["documents", "metadatas", "distances"]
        )
        docs.extend(res["documents"][0])
        metas.extend(res["metadatas"][0])

    return docs, metas


@torch.inference_mode()
def main():
    # --- Chroma ---
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(name=COLLECTION_NAME)

    # --- Embedder ---
    embedder = SentenceTransformer(EMBED_MODEL)

    # --- LLM ---
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    # model = AutoModelForCausalLM.from_pretrained(
    #     LLM_MODEL,
    #     dtype="auto",          # ganti dari torch_dtype agar tidak warning
    #     device_map="auto"
    # ).eval()

    # Using fully GPU (CUDA)
    model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    dtype="auto"
    ).to("cuda").eval()

    print("RAG Chat siap. Ketik pertanyaan, ENTER kosong untuk keluar.\n")

    while True:
        q = input("User> ").strip()
        if not q:
            break

        q_emb = embedder.encode([q], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32).tolist()

        doc_types = route_doc_types(q)
        docs, metas = retrieve(col, q_emb, doc_types, top_k=TOP_K)

        # dedup + build context
        docs, metas = dedup_docs(docs, metas, limit=10)
        context = build_context(docs, metas)

        mode = detect_mode(q)  # detect mode

        # pilih prompt
        system = SYSTEM_ITINERARY if mode == "itinerary" else SYSTEM_RECO

        user = f"KONTEKS:\n{context}\n\nPERTANYAAN:\n{q}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15
        )

        answer_ids = out[0][inputs["input_ids"].shape[-1]:]
        ans = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        print("\nAssistant>\n" + ans)
        print("\nSumber konteks (Top-K after routing+dedup):")
        for m in metas:
            print(f"- {m.get('source_file')} page {m.get('page')} chunk {m.get('chunk')} (type={m.get('doc_type')})")
        print()


if __name__ == "__main__":
    main()
