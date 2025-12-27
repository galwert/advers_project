import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

# --- CONFIGURATION ---
# The Grand List (20 Models)
MODELS = {
    # --- The Llama Family (Baseline) ---
    "Llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "Llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Vicuna-7b": "lmsys/vicuna-7b-v1.5",

    # --- The Mistral Family (Strong/Sparse) ---
    "Mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "Zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "Hermes-2": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "Starling-7b": "berkeley-nest/Starling-LM-7B-alpha",
    "OpenChat-3.5": "openchat/openchat_3.5",

    # --- Google & Microsoft (Aliens) ---
    "Gemma-7b": "google/gemma-7b-it",
    "Phi-2": "microsoft/phi-2",  # 2560 dims (Distinct)
    "Orca-2-7b": "microsoft/Orca-2-7b",

    # --- The "Eastern" Models (Different Data) ---
    "Qwen1.5-7b": "Qwen/Qwen1.5-7B-Chat",
    "Yi-6b": "01-ai/Yi-6B-Chat",
    "Baichuan2-7b": "baichuan-inc/Baichuan2-7B-Chat",
    "DeepSeek-7b": "deepseek-ai/deepseek-llm-7b-chat",
    "InternLM2-7b": "internlm/internlm2-chat-7b",

    # --- Distinct Architectures ---
    "Falcon-7b": "tiiuae/falcon-7b-instruct",  # 4544 dims
    "Solar-10.7b": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "NeuralChat-7b": "Intel/neural-chat-7b-v3-1",
    "StableZephyr-3b": "stabilityai/stablelm-zephyr-3b"  # 2560 dims
}

DATA_DIR = "matrix_data_grand"
SAVE_PLOT = "grand_geometry_matrix.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 2000  # High statistical significance
BATCH_SIZE = 4  # Conservative batch size to prevent OOM on big models
LAYER_IDX = -1  # Last layer (Universal output space)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ==========================================
# PHASE 1: BATCH HARVESTING
# ==========================================
print(f"=== PHASE 1: HARVESTING {NUM_SAMPLES} VECTORS ===")
# Use Wikitext for neutral alignment
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
sentences = [x['text'] for x in dataset if 50 < len(x['text']) < 200][:NUM_SAMPLES]

# We need to ensure we have enough sentences
if len(sentences) < NUM_SAMPLES:
    print(f"Warning: Only found {len(sentences)} valid sentences.")

for name, model_id in MODELS.items():
    save_path = f"{DATA_DIR}/{name}.pt"

    if os.path.exists(save_path):
        print(f"[+] Skipping {name} (Already Harvested)")
        continue

    print(f"[*] Scanning {name} ({model_id})...")
    try:
        # Load Tokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        all_vecs = []

        # Batch Loop
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i: i + BATCH_SIZE]

            # Tokenize
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                # Extract Last Layer, Last Token
                # Shape: [Batch, Seq, Dim] -> [Batch, Dim]
                hidden = out.hidden_states[LAYER_IDX]

                # Careful: We want the last *actual* token, not the padding.
                # But for simple Wikitext sentences, simply taking the last index
                # or the index before padding is usually sufficient for alignment.
                # Here we take the last position to keep it fast and compatible.
                vecs = hidden[:, -1, :].cpu()
                all_vecs.append(vecs)

            print(f"    Batch {i}/{NUM_SAMPLES}", end="\r")

        # Save
        final_tensor = torch.cat(all_vecs, dim=0)
        torch.save(final_tensor, save_path)
        print(f"\n    -> Saved {final_tensor.shape} to {save_path}")

        # Clean Memory
        del model, tok
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"\n[-] FAILED to load {name}: {e}")

# ==========================================
# PHASE 2: ALIGNMENT CALCULATION
# ==========================================
print("\n=== PHASE 2: CALCULATING PROCRUSTES ALIGNMENT ===")
# Filter only models that successfully saved
valid_models = sorted([m for m in MODELS.keys() if os.path.exists(f"{DATA_DIR}/{m}.pt")])
n = len(valid_models)
matrix = np.zeros((n, n))


def align_and_score(X, Y):
    # 1. Normalize
    X = torch.nn.functional.normalize(X.float(), dim=1)
    Y = torch.nn.functional.normalize(Y.float(), dim=1)

    # 2. Handle Dimension Mismatch (e.g. Phi-2's 2560 vs Llama's 4096)
    # Strategy: Truncate to the shared lower dimension
    min_dim = min(X.shape[1], Y.shape[1])
    X = X[:, :min_dim]
    Y = Y[:, :min_dim]

    # 3. Procrustes Solution (Analytical Rotation)
    try:
        # Align X to Y
        M = torch.matmul(Y.t(), X)
        U, S, V = torch.svd(M)
        W = torch.matmul(U, V.t())

        # Apply Rotation
        X_aligned = torch.matmul(X, W.t())

        # Score (Cosine Sim)
        sim = torch.nn.functional.cosine_similarity(X_aligned, Y).mean().item()
        return sim
    except Exception as e:
        print(f"Math Error: {e}")
        return 0.0


# Compute Pairwise
for i, name_a in enumerate(valid_models):
    for j, name_b in enumerate(valid_models):
        if i > j: continue  # Matrix is symmetric

        vec_a = torch.load(f"{DATA_DIR}/{name_a}.pt")
        vec_b = torch.load(f"{DATA_DIR}/{name_b}.pt")

        # Match lengths (in case of partial harvest)
        min_len = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:min_len]
        vec_b = vec_b[:min_len]

        print(f"[*] {name_a} <-> {name_b} (N={min_len})", end="\r")
        score = align_and_score(vec_a, vec_b)

        matrix[i, j] = score
        matrix[j, i] = score

print("\n[+] Calculation Complete.")

# ==========================================
# PHASE 3: PLOTTING
# ==========================================
print("=== PHASE 3: GENERATING HEATMAP ===")
plt.figure(figsize=(20, 18))  # Massive figure for 20x20
sns.set(font_scale=1.0)

# Mask upper triangle
mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

sns.heatmap(matrix,
            annot=True,
            fmt=".2f",
            xticklabels=valid_models,
            yticklabels=valid_models,
            cmap="viridis",
            vmin=0.5, vmax=1.0,  # Scale focus on high alignment
            square=True,
            cbar_kws={"shrink": .8})

plt.title(f"The Grand Geometry of LLMs: 20-Model Procrustes Alignment\n(N={NUM_SAMPLES}, Layer={LAYER_IDX})")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(SAVE_PLOT)
print(f"[SUCCESS] Grand Matrix saved to {SAVE_PLOT}")