import pandas as pd
import json
import os

# --- KONFIGURASI ---
INPUT_DIR = "data/raw/files"
OUTPUT_FILE = "data/processed/train.jsonl"
os.makedirs("data/processed", exist_ok=True)

final_dataset = []

print("🚀 Memulai proses konversi data ulasan (3 File Spesifik)...")

# Fungsi helper untuk membaca CSV dengan aman
def load_csv(filename):
    path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(path):
        print(f"✅ Membaca file: {filename}")
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"❌ Gagal membaca {filename}: {e}")
    else:
        print(f"⚠️ File tidak ditemukan di {path}")
    return None

# --- FILE 1: data-resto-hotel-v2.csv ---
# Kolom: place-name, reviewer-rating, review-text
df1 = load_csv("data-resto-hotel-v2.csv")
if df1 is not None:
    # Filter hanya rating bagus (>= 4) agar model belajar merekomendasikan yang baik
    # Pastikan kolom rating numerik
    df1['reviewer-rating'] = pd.to_numeric(df1['reviewer-rating'], errors='coerce')
    df1 = df1[df1['reviewer-rating'] >= 4.0]
    
    count = 0
    for _, row in df1.iterrows():
        place = row.get('place-name', '')
        review = row.get('review-text', '')
        
        # Validasi data tidak kosong
        if pd.isna(place) or pd.isna(review) or str(review).strip() == "":
            continue
            
        # Format Instruksi: User tanya pendapat -> Assistant kasih review
        final_dataset.append({
            "messages": [
                {"role": "user", "content": f"Bagaimana kualitas penginapan/resto di {place}?"},
                {"role": "assistant", "content": f"Menurut pengalaman pengunjung: \"{review}\""}
            ]
        })
        count += 1
    print(f"   -> Berhasil memproses {count} ulasan hotel/resto.")

# --- FILE 2: data-wisata-v2.csv ---
# Kolom: place-name, reviewer-rating, review-text
df2 = load_csv("data-wisata-v2.csv")
if df2 is not None:
    df2['reviewer-rating'] = pd.to_numeric(df2['reviewer-rating'], errors='coerce')
    df2 = df2[df2['reviewer-rating'] >= 4.0]
    
    count = 0
    for _, row in df2.iterrows():
        place = row.get('place-name', '')
        review = row.get('review-text', '')
        
        if pd.isna(place) or pd.isna(review) or str(review).strip() == "":
            continue
            
        final_dataset.append({
            "messages": [
                {"role": "user", "content": f"Apa kata wisatawan tentang {place}?"},
                {"role": "assistant", "content": f"Wisatawan memberikan ulasan positif: \"{review}\""}
            ]
        })
        count += 1
    print(f"   -> Berhasil memproses {count} ulasan objek wisata.")

# --- FILE 3: wisata-toba-cleaned.csv ---
# Kolom: place_name, rating, reviews (Perhatikan beda nama kolom: underscore vs dash)
df3 = load_csv("wisata-toba-cleaned.csv")
if df3 is not None:
    df3['rating'] = pd.to_numeric(df3['rating'], errors='coerce')
    df3 = df3[df3['rating'] >= 4.0]
    
    count = 0
    for _, row in df3.iterrows():
        place = row.get('place_name', '')
        review = row.get('reviews', '')
        category = row.get('category', 'Wisata Toba')
        
        if pd.isna(place) or pd.isna(review) or str(review).strip() == "":
            continue
            
        # Variasi Prompt: Minta rekomendasi kategori
        final_dataset.append({
            "messages": [
                {"role": "user", "content": f"Berikan rekomendasi tempat {category} yang bagus."},
                {"role": "assistant", "content": f"Saya merekomendasikan {place}. Pengunjung mengatakan: \"{review}\""}
            ]
        })
        count += 1
    print(f"   -> Berhasil memproses {count} data wisata cleaned.")

# --- SIMPAN HASIL ---
print(f"\n💾 Menyimpan total {len(final_dataset)} baris data latih ke {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for entry in final_dataset:
        f.write(json.dumps(entry) + "\n")

print("🎉 Selesai! Data reviews siap digunakan untuk Training.")