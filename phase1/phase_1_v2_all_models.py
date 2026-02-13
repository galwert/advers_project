import torch
import numpy as np
import csv
import gc
import os
import requests
import pandas as pd
import io
from itertools import combinations
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "alignment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_DIM = 2048
TOP_K = 10
# Note: N_SAMPLES is determined dynamically by dataset loading below
print("TARGET_DIM:", TARGET_DIM)
MODELS = {
    "Llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "Llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "Mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "Zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "Hermes-2": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "Starling-7b": "berkeley-nest/Starling-LM-7B-alpha",
    "OpenChat-3.5": "openchat/openchat_3.5",
    "Gemma-7b": "google/gemma-7b-it",
    "Phi-2": "microsoft/phi-2",
    "Orca-2-7b": "microsoft/Orca-2-7b",
    "Qwen1.5-7b": "Qwen/Qwen1.5-7B-Chat",
    "Yi-6b": "01-ai/Yi-6B-Chat",
    "Baichuan2-7b": "baichuan-inc/Baichuan2-7B-Chat",
    "DeepSeek-7b": "deepseek-ai/deepseek-llm-7b-chat",
    "InternLM2-7b": "internlm/internlm2-chat-7b",
    "Falcon-7b": "tiiuae/falcon-7b-instruct",
    "Solar-10.7b": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "NeuralChat-7b": "Intel/neural-chat-7b-v3-1",
    "StableZephyr-3b": "stabilityai/stablelm-zephyr-3b"
}

# --- Load Sentences ---
print("Loading datasets...")

# 1. Load WikiText (Target: ~1040 valid samples)
wiki_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
wiki_sentences = []
for x in wiki_dataset:
    text = x['text']
    # Filter out empty/whitespace-only lines to prevent tokenizer crashes
    if text.strip() and len(text.strip()) > 10:
        wiki_sentences.append(text)
    if len(wiki_sentences) >= 1040:
        break

print(f"Loaded {len(wiki_sentences)} WikiText sentences")

# 2. Load AdvBench (Target: ~520 * 2 = ~1040)
try:
    ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    print(f"[*] Downloading AdvBench from {ADVBENCH_URL}...")

    response = requests.get(ADVBENCH_URL)
    response.raise_for_status()

    # Read CSV
    df = pd.read_csv(io.StringIO(response.text))

    # Extract only the 'goal' column (the harmful prompt) as a list of strings
    adv_prompts = df['goal'].tolist()

    # Duplicate the list to reach target size
    hb_sentences = adv_prompts * 2

    print(f"Loaded {len(adv_prompts)} AdvBench behaviors (Duplicated to {len(hb_sentences)})")

except Exception as e:
    print(f"[-] Error loading AdvBench: {e}")
    hb_sentences = []

# 3. Combine both lists
sentences = wiki_sentences + hb_sentences
N_SAMPLES = len(sentences)

print(f"Total sentences to process: {N_SAMPLES}")
if len(sentences) > 0:
    print(f"Sample 0 (Wiki): {sentences[0][:50]}...")
    print(f"Sample {len(wiki_sentences)} (AdvBench): {sentences[len(wiki_sentences)][:50]}...")

# Clean up memory
del wiki_dataset, df
gc.collect()

# --- Phase 1: Extract and cache embeddings for all models ---
embeddings_cache = {}
embeddings_pca_cache = {}
variance_cache = {}

for model_name, model_path in MODELS.items():
    cache_path = f"{OUTPUT_DIR}/{model_name}_embeddings.pt"

    if os.path.exists(cache_path):
        print(f"[+] Loading cached {model_name}")
        embeddings_cache[model_name] = torch.load(cache_path)
        continue

    print(f"[*] Processing {model_name}...")

    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.pad_token_id

        n_layers = None
        layer_embeddings = None

        valid_samples_count = 0

        for i, sentence in enumerate(sentences):
            # Skip empty strings
            if not sentence.strip():
                continue

            inputs = tok(sentence, return_tensors="pt").to(DEVICE)

            # Skip if tokenizer produced 0 tokens
            if inputs.input_ids.shape[1] == 0:
                continue

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            if n_layers is None:
                n_layers = len(out.hidden_states)
                layer_embeddings = [[] for _ in range(n_layers)]

            # Check for valid output shape before accessing
            valid_layer = True
            temp_layer_data = []

            for layer_idx, layer_hidden in enumerate(out.hidden_states):
                if layer_hidden.shape[1] > 0:
                    last_token = layer_hidden[0, -1, :].cpu()
                    temp_layer_data.append(last_token)
                else:
                    valid_layer = False
                    break

            if valid_layer:
                for idx, val in enumerate(temp_layer_data):
                    layer_embeddings[idx].append(val)
                valid_samples_count += 1


        if layer_embeddings is None or len(layer_embeddings[0]) == 0:
            print(f"[-] WARNING: {model_name} produced 0 valid embeddings.")
            continue

        result = torch.stack([torch.stack(layer) for layer in layer_embeddings])
        embeddings_cache[model_name] = result
        torch.save(result, cache_path)
        print(f"    Saved: {result.shape} (Valid samples: {valid_samples_count})")

        del model, tok
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"[-] FAILED {model_name}: {e}")
        # import traceback
        # traceback.print_exc()

# --- Phase 2: Apply PCA to all models (Robust) ---
print("\n=== Applying PCA (Robust) ===")

for model_name, embeddings in embeddings_cache.items():
    n_layers = embeddings.shape[0]
    reduced = []
    variances = []

    print(f"[*] PCA for {model_name}...")

    for layer_idx in range(n_layers):
        # 1. Convert to float32 (fixes precision issues)
        X = embeddings[layer_idx].float().numpy()

        # 2. Check for NaNs/Infs and replace with 0
        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Check variance to prevent "invalid value encountered in divide"
        if np.var(X) < 1e-9:
            # If layer is dead/constant, return zeros
            X_reduced = np.zeros((X.shape[0], TARGET_DIM))
            explained_var = 0.0
        else:
            pca = PCA(n_components=min(TARGET_DIM, X.shape[0], X.shape[1]))
            X_reduced = pca.fit_transform(X)

            # Fix "invalid value in divide" warning
            total_var = pca.explained_variance_.sum()
            if total_var > 0:
                explained_var = pca.explained_variance_ratio_.sum()
            else:
                explained_var = 0.0

            # Pad with zeros if we got fewer components than TARGET_DIM
            if X_reduced.shape[1] < TARGET_DIM:
                padding = np.zeros((X_reduced.shape[0], TARGET_DIM - X_reduced.shape[1]))
                X_reduced = np.hstack([X_reduced, padding])

        reduced.append(torch.tensor(X_reduced, dtype=torch.float32))
        variances.append(explained_var)

    embeddings_pca_cache[model_name] = torch.stack(reduced)
    variance_cache[model_name] = variances

    last_var = variances[-1] if variances else 0
    print(f"    Shape: {embeddings_pca_cache[model_name].shape}, Last Layer Var: {last_var:.3f}")

# --- Phase 3: Compute alignments for all pairs ---
print("\n=== Computing Alignments ===")

model_names = list(embeddings_pca_cache.keys())
best_scores = {}

for name_a, name_b in combinations(model_names, 2):
    emb_a = embeddings_pca_cache[name_a]
    emb_b = embeddings_pca_cache[name_b]

    # --- CRITICAL FIX: Sync lengths ---
    len_a = emb_a.shape[1]
    len_b = emb_b.shape[1]
    min_len = min(len_a, len_b)

    if abs(len_a - len_b) > 0:
        print(f"[*] {name_a} ({len_a}) <-> {name_b} ({len_b}) | Truncating to {min_len}")
    else:
        print(f"[*] {name_a} <-> {name_b}")

    if min_len < 10:
        print(f"    [!] Skipping: Not enough common samples ({min_len})")
        continue

    n_layers_a = emb_a.shape[0]
    n_layers_b = emb_b.shape[0]

    alignment_matrix = np.zeros((n_layers_a - 1, n_layers_b - 1))

    for i in range(1, n_layers_a):
        for j in range(1, n_layers_b):
            # Slice both to the minimum common length
            X = emb_a[i, :min_len, :]
            Y = emb_b[j, :min_len, :]

            # Least Squares Alignment
            n = X.shape[0]
            X_aug = torch.cat([X.float(), torch.ones(n, 1)], dim=1)

            # lstsq requires matching row counts
            W, _, _, _ = torch.linalg.lstsq(X_aug, Y.float())
            X_aligned = X_aug @ W

            X_norm = torch.nn.functional.normalize(X_aligned, dim=1)
            Y_norm = torch.nn.functional.normalize(Y.float(), dim=1)
            score = (X_norm * Y_norm).sum(dim=1).mean().item()

            alignment_matrix[i - 1, j - 1] = score

    # Save alignment matrix
    matrix_path = f"{OUTPUT_DIR}/{name_a}_{name_b}_matrix.csv"
    np.savetxt(matrix_path, alignment_matrix, delimiter=",", fmt="%.4f")

    # Get top-k pairs
    flat = alignment_matrix.flatten()
    top_k_flat = np.argsort(flat)[::-1][:TOP_K]

    top_pairs = []
    for idx in top_k_flat:
        i = idx // (n_layers_b - 1)
        j = idx % (n_layers_b - 1)
        top_pairs.append((i + 1, j + 1, alignment_matrix[i, j]))

    # Save top pairs
    pairs_path = f"{OUTPUT_DIR}/{name_a}_{name_b}_top_pairs.csv"
    with open(pairs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_a", "layer_b", "score"])
        for layer_a, layer_b, score in top_pairs:
            writer.writerow([layer_a, layer_b, f"{score:.4f}"])

    best_score = top_pairs[0][2]
    best_scores[(name_a, name_b)] = best_score
    best_scores[(name_b, name_a)] = best_score

    print(f"    Best: layers ({top_pairs[0][0]}, {top_pairs[0][1]}) = {best_score:.4f}")

# --- Phase 4: Save summary matrix ---
print("\n=== Saving Summary ===")

n = len(model_names)
summary_matrix = np.zeros((n, n))

for i, name_a in enumerate(model_names):
    for j, name_b in enumerate(model_names):
        if i == j:
            summary_matrix[i, j] = 1.0
        else:
            summary_matrix[i, j] = best_scores.get((name_a, name_b), 0)

summary_path = f"{OUTPUT_DIR}/summary_matrix.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + model_names)
    for i, name in enumerate(model_names):
        writer.writerow([name] + [f"{summary_matrix[i, j]:.4f}" for j in range(n)])

print(f"Saved summary to {summary_path}")
print("Done!")