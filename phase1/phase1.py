import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

# --- CONFIG ---
SOURCE_ID = "meta-llama/Llama-2-7b-chat-hf"
TARGET_ID = "mistralai/Mistral-7B-Instruct-v0.1" # The Alien Model
SAVE_DIR = "alignment_data"
NUM_SAMPLES = 2000  # Enough to learn a stable rotation
LAYER_IDX = 16      # Middle layers represent "concepts" best

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. GET SENTENCES ---
print("[*] Loading Dataset...")
# We use wikitext for neutral, high-quality English
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
# Filter for reasonable length sentences (50-100 chars)
sentences = [x['text'] for x in dataset if 50 < len(x['text']) < 200][:NUM_SAMPLES]
print(f"[*] Harvested {len(sentences)} sentences.")

# --- 2. EXTRACT SOURCE VECTORS (LLAMA) ---
print(f"[*] Loading Source: {SOURCE_ID}")
tok_src = AutoTokenizer.from_pretrained(SOURCE_ID)
model_src = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float16, device_map="auto")

src_tensors = []
print("   Scanning Llama...")
for i, s in enumerate(sentences):
    inputs = tok_src(s, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_src(**inputs, output_hidden_states=True)
        # Grab the last token at the middle layer
        # [1, seq, dim] -> [dim]
        vec = out.hidden_states[LAYER_IDX][0, -1, :].cpu()
        src_tensors.append(vec)
    if i % 100 == 0: print(f"   {i}/{NUM_SAMPLES}", end="\r")

# Save Source
torch.save(torch.stack(src_tensors), f"{SAVE_DIR}/src_llama.pt")
print(f"\n[+] Saved Source vectors. Shape: {torch.stack(src_tensors).shape}")

# Cleanup to free VRAM for Mistral
del model_src, tok_src
torch.cuda.empty_cache()


# --- 3. EXTRACT TARGET VECTORS (MISTRAL) ---
print(f"[*] Loading Target: {TARGET_ID}")
tok_tgt = AutoTokenizer.from_pretrained(TARGET_ID)
model_tgt = AutoModelForCausalLM.from_pretrained(TARGET_ID, torch_dtype=torch.float16, device_map="auto")

tgt_tensors = []
print("   Scanning Mistral...")
for i, s in enumerate(sentences):
    inputs = tok_tgt(s, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_tgt(**inputs, output_hidden_states=True)
        # Mistral also has 4096 dims, but the geometry is different
        vec = out.hidden_states[LAYER_IDX][0, -1, :].cpu()
        tgt_tensors.append(vec)
    if i % 100 == 0: print(f"   {i}/{NUM_SAMPLES}", end="\r")

# Save Target
torch.save(torch.stack(tgt_tensors), f"{SAVE_DIR}/tgt_mistral.pt")
print(f"\n[+] Saved Target vectors. Shape: {torch.stack(tgt_tensors).shape}")