import torch
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- KONFIGURASI ---
ADAPTER_PATH = "outputs/qwen-toba-adapter"
CONFIG_PATH = "configs/model.yaml"

# 1. Cek Apakah Adapter Ada
if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"Adapter tidak ditemukan di {ADAPTER_PATH}. Training dulu!")

# 2. Ambil Nama Base Model dari Config
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct" # Default
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        base_model_id = config.get("generator_model", base_model_id)

print(f"⚙️  Base Model: {base_model_id}")
print(f"⚙️  Adapter: {ADAPTER_PATH}")

# 3. Load Base Model (4-Bit agar hemat RAM)
print("📥 Memuat Base Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 4. GABUNGKAN ADAPTER (INFERENSI)
# Inilah langkah kuncinya: Menempelkan hasil training ke otak model
print("🔗 Menggabungkan Adapter LoRA...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("\n" + "="*50)
print("🤖 QWEN-TOBA CHATBOT (TESTING MODE)")
print("   Ketik 'exit' atau 'keluar' untuk berhenti.")
print("="*50 + "\n")

# 5. LOOP CHATTING
while True:
    try:
        query = input("User (Anda): ")
        if query.lower() in ["exit", "keluar", "quit"]:
            print("Sampai jumpa!")
            break
        
        # Format Chat Template
        messages = [
            {"role": "system", "content": "Anda adalah asisten AI ahli wisata Danau Toba yang cerdas, kreatif, dan informatif. Berikan jawaban yang lengkap, terstruktur, menggunakan emoji yang relevan, dan gaya bahasa yang menarik seperti pemandu wisata profesional."},
            {"role": "user", "content": query}
        ]
        
        # Ubah teks jadi token
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        # Generate Jawaban
        # Generate Jawaban dengan Parameter yang Lebih Cerdas
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,   # Naikkan jadi 512 agar jawabannya bisa panjang
            temperature=0.7,      # 0.7 = Kreatif tapi tetap nyambung
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1, # Mencegah pengulangan kata
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Ambil hanya jawaban baru (potong prompt input)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Assistant: {response}\n")
        print("-" * 30)

    except KeyboardInterrupt:
        print("\nBerhenti paksa.")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")