from pathlib import Path
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


@torch.inference_mode()
def main():
    # --- Chroma ---
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(name=COLLECTION_NAME)

    # --- Embedder ---
    embedder = SentenceTransformer(EMBED_MODEL)

    # --- LLM ---
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype="auto",
        device_map="auto"
    ).eval()

    print("✅ RAG Chat siap. Ketik pertanyaan, ENTER kosong untuk keluar.\n")

    while True:
        q = input("User> ").strip()
        if not q:
            break

        q_emb = embedder.encode([q], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32).tolist()

        res = col.query(
            query_embeddings=q_emb,
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        context = build_context(docs, metas)

        system = (
            "Kamu adalah asisten sistem rekomendasi pariwisata Danau Toba. "
            "Gunakan KONTEKS sebagai sumber utama. "
            "Jika jawabannya tidak ada di konteks, katakan 'data tidak tersedia'. "
            "Jawab Bahasa Indonesia, ringkas, dan jelas."
        )

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
            max_new_tokens=280,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        answer_ids = out[0][inputs["input_ids"].shape[-1]:]
        ans = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        print("\nAssistant>\n" + ans)
        print("\nSumber konteks (Top-K):")
        for m in metas:
            print(f"- {m.get('source_file')} page {m.get('page')} chunk {m.get('chunk')}")
        print()


if __name__ == "__main__":
    main()