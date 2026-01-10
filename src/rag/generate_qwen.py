import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_qwen(generator_model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(generator_model)
    mdl = AutoModelForCausalLM.from_pretrained(
        generator_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).eval()
    return tok, mdl

@torch.inference_mode()
def generate(tok, mdl, messages, max_new_tokens=350, temperature=0.7, top_p=0.9):
    prompt_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(mdl.device)

    out = mdl.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return tok.decode(out[0], skip_special_tokens=True)
