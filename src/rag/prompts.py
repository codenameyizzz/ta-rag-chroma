def build_messages(question: str, context: str):
    system = (
        "Kamu adalah asisten sistem rekomendasi pariwisata Danau Toba. "
        "Gunakan KONTEN KANDIDAT sebagai sumber utama. "
        "Jika info tidak ada di kandidat, jawab: 'data tidak tersedia'. "
        "Output: Top-3 rekomendasi + alasan singkat untuk tiap item."
    )

    user = f"""Pertanyaan:
{question}

KONTEN KANDIDAT (hasil retrieval):
{context}
"""
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
