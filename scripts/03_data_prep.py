import pandas as pd
import json
import os

# --- KONFIGURASI ---
INPUT_DIR = "data/raw/files"
OUTPUT_FILE = "data/processed/train.jsonl"
os.makedirs("data/processed", exist_ok=True)

final_dataset = []

print("🚀 Memulai proses konversi data menjadi format 'RICH & CREATIVE'...")

# Fungsi helper load csv
def load_csv(filename):
    path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --- 1. DATA ATRAKSI (SUMBER UTAMA KECERDASAN) ---
# Kita buat format yang sangat lengkap: Deskripsi + Lokasi + Harga + Sejarah
df_atraksi = load_csv("data-gabungan.xlsx - Attractions Info.csv")
if df_atraksi is not None:
    print("✅ Memproses Atraksi menjadi format Artikel Lengkap...")
    for _, row in df_atraksi.iterrows():
        row = row.fillna('')
        nama = row.get('Nama Atraksi', '')
        if not nama: continue

        # AMBIL SEMUA DATA
        desc = row.get('Deskripsi', 'Tempat ini sangat indah.')
        loc = row.get('Lokasi', 'Kawasan Danau Toba')
        price = row.get('Ticket Prices', 'Cek di lokasi')
        hours = row.get('Opening Hours', '08.00 - 17.00')
        history = row.get('Historical & Cultural Background', '')

        # --- RAHASIA KREATIFITAS: FORMATTING ---
        # Kita susun jawaban model supaya panjang dan cantik (Markdown style)
        rich_response = f"""
🌟 **Rekomendasi Wisata: {nama}**

📖 **Tentang Tempat Ini:**
{desc}

📍 **Informasi Penting:**
- **Lokasi:** {loc}
- **Jam Buka:** {hours}
- **Tiket Masuk:** {price}

💡 **Fakta Menarik/Sejarah:**
{history if history else "Tempat ini memiliki nilai budaya yang kental dengan masyarakat lokal."}

✨ **Kenapa Harus ke Sini?**
Cocok untuk Anda yang mencari pengalaman otentik di Danau Toba. Jangan lupa bawa kamera untuk mengabadikan momen!
"""
        # Masukkan ke dataset
        final_dataset.append({
            "messages": [
                {"role": "user", "content": f"Berikan rekomendasi lengkap tentang {nama}"},
                {"role": "assistant", "content": rich_response.strip()}
            ]
        })
        
        # Variasi pertanyaan agar model pintar
        final_dataset.append({
            "messages": [
                {"role": "user", "content": f"Apa yang menarik di {nama}?"},
                {"role": "assistant", "content": rich_response.strip()}
            ]
        })

# --- 2. DATA EVENT (FORMAT BERITA) ---
df_event = load_csv("data-gabungan.xlsx - Event KDT 2025.csv")
if df_event is not None:
    print("✅ Memproses Event menjadi format Berita...")
    for _, row in df_event.iterrows():
        nama = row.get('Nama Event', '')
        if not nama: continue
        
        jadwal = row.get('Jadwal dan Lokasi Event', '')
        penyelenggara = row.get('Penyelenggara Event', '')
        jenis = row.get('Jenis Event', '')

        rich_response = f"""
🎉 **Event Seru: {nama}**

📅 **Jadwal & Lokasi:**
{jadwal}

🎭 **Detail Acara:**
Ini adalah {jenis} yang sangat dinantikan! Acara ini diselenggarakan oleh **{penyelenggara}** dan akan memeriahkan suasana Danau Toba.

📌 **Tips:**
Pastikan Anda datang lebih awal untuk mendapatkan tempat terbaik!
"""
        final_dataset.append({
            "messages": [
                {"role": "user", "content": f"Ada event apa saja? Ceritakan tentang {nama}"},
                {"role": "assistant", "content": rich_response.strip()}
            ]
        })

# --- 3. DATA REVIEW (FORMAT TESTIMONI) ---
# Kita ambil review, tapi kita bungkus dengan kalimat pembuka yang manis
df_review = load_csv("wisata-toba-cleaned.csv") # Atau file review lainnya
if df_review is not None:
    # Filter rating 5
    if 'rating' in df_review.columns:
        df_review = df_review[df_review['rating'] == 5.0]
    
    # Ambil sampel saja biar tidak kebanyakan (misal 5000 review terbaik)
    df_review = df_review.head(5000) 
    print("✅ Memproses Review menjadi format Testimoni...")

    for _, row in df_review.iterrows():
        place = row.get('place_name', '')
        review = row.get('reviews', '')
        if not place or not review: continue

        rich_response = f"""
👍 **Ulasan Pengunjung untuk {place}**

Berdasarkan pengalaman wisatawan terbaru, tempat ini sangat direkomendasikan!

🗣️ **Kata Mereka:**
"{review}"

⭐ **Rating:** 5/5 Sempurna!
"""
        final_dataset.append({
            "messages": [
                {"role": "user", "content": f"Bagaimana pendapat orang tentang {place}?"},
                {"role": "assistant", "content": rich_response.strip()}
            ]
        })

# --- SIMPAN ---
print(f"💾 Menyimpan {len(final_dataset)} data 'Rich Format' ke {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for entry in final_dataset:
        f.write(json.dumps(entry) + "\n")

print("🎉 Selesai! Data sekarang jauh lebih estetik dan informatif.")