import torch
import yaml
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# --- 1. KONFIGURASI ---
MODEL_CONFIG_PATH = "configs/model.yaml"
DATASET_PATH = "data/processed/train.jsonl"
OUTPUT_DIR = "outputs/qwen-toba-adapter"

print("⚙️  Menyiapkan konfigurasi training (Mode Stabil)...")

if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, "r") as f:
        config_yaml = yaml.safe_load(f)
        model_id = config_yaml.get("generator_model", "Qwen/Qwen2.5-1.5B-Instruct")
else:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"🤖 Base Model: {model_id}")

# --- 2. SETUP DATASET ---
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Dataset tidak ditemukan di {DATASET_PATH}")
    
print(f"📚 Memuat dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# --- 3. SETUP QUANTIZATION ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# --- 4. LOAD MODEL & TOKENIZER ---
print("📥 Memuat Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

# --- 5. LORA CONFIG ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 6. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./outputs/checkpoints",
    max_steps=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    optim="paged_adamw_32bit",
    report_to="none",
    remove_unused_columns=True
)

# --- [PERBAIKAN UTAMA] FUNGSI FORMATTER ---
# Fungsi ini mengubah List Dictionary menjadi String Format Qwen
def formatting_prompts_func(example):
    output_texts = []
    # Loop setiap baris data
    for message in example['messages']:
        # Gunakan template bawaan tokenizer untuk mengubah list jadi string
        # Hasilnya: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi...<|im_end|>"
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

# --- 7. MULAI TRAINING ---
print("🔥 Memulai Training QLoRA...")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    # KITA GANTI 'dataset_text_field' DENGAN 'formatting_func'
    formatting_func=formatting_prompts_func, 
)

trainer.train()

# --- 8. SIMPAN ---
print(f"💾 Training Selesai! Menyimpan adapter ke: {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("\n🎉 SELAMAT! Model berhasil dilatih.")