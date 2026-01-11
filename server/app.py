from typing import List, Optional
import threading

import numpy as np
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Konfig (samakan dengan project kamu) =====
CHROMA_DIR = "stores/chroma"
COLLECTION_NAME = "danau_toba_pdfs"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "Qwen/Qwen3-1.7B"

TOP_K = 15
MAX_CONTEXT_CHARS = 12000

# ===== Helpers (pakai punya kamu) =====
def route_doc_types(q: str):
    ql = q.lower()
    if "itinerary" in ql or "hari" in ql or "malam" in ql:
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
    return want

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
    parts, total = [], 0
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
        res = col.query(query_embeddings=q_emb, n_results=top_k, include=["documents","metadatas","distances"])
        return res["documents"][0], res["metadatas"][0]

    per_k = max(3, top_k // len(doc_types))
    for dt in doc_types:
        res = col.query(
            query_embeddings=q_emb,
            n_results=per_k,
            where={"doc_type": dt},
            include=["documents","metadatas","distances"]
        )
        docs.extend(res["documents"][0])
        metas.extend(res["metadatas"][0])
    return docs, metas

# ===== FastAPI =====
app = FastAPI(title="TA RAG API")

# CORS supaya Next.js localhost bisa hit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class Source(BaseModel):
    source_file: str
    page: int
    chunk: int
    doc_type: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

# ===== Load resource sekali saat startup =====
client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_collection(name=COLLECTION_NAME)

embedder = SentenceTransformer(EMBED_MODEL)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    dtype="auto",
    device_map="auto"
).eval()

# supaya aman kalau ada request bersamaan
gen_lock = threading.Lock()

SYSTEM_PROMPT = (
    "Kamu adalah asisten perencana itinerary Danau Toba.\n"
    "ATURAN KETAT:\n"
    "1) Semua nama tempat/objek wisata/hotel/restoran yang kamu sebut harus ada di KONTEKS.\n"
    "2) Kamu boleh kreatif menyusun itinerary, tapi fakta (nama, lokasi, fasilitas, harga) hanya dari KONTEKS.\n"
    "3) Jika konteks tidak cukup, tulis 'data tidak tersedia' dan jelaskan kekurangannya.\n"
    "4) Jika user meminta itinerary N hari, usahakan selesaikan semua hari sampai tuntas.\n"
)

@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(req: ChatRequest):
    q = req.message.strip()
    q_emb = embedder.encode([q], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32).tolist()

    doc_types = route_doc_types(q)
    docs, metas = retrieve(col, q_emb, doc_types, top_k=TOP_K)
    docs, metas = dedup_docs(docs, metas, limit=10)
    context = build_context(docs, metas)

    user_msg = f"KONTEKS:\n{context}\n\nPERTANYAAN:\n{q}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with gen_lock:
        out = model.generate(
            **inputs,
            max_new_tokens=1400,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15,
        )

    answer_ids = out[0][inputs["input_ids"].shape[-1]:]
    ans = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    sources = [
        Source(
            source_file=m.get("source_file"),
            page=int(m.get("page")),
            chunk=int(m.get("chunk")),
            doc_type=m.get("doc_type"),
        )
        for m in metas
    ]
    return ChatResponse(answer=ans, sources=sources)
