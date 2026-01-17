import torch
import numpy as np
import csv
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.decomposition import PCA
from itertools import combinations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "alignment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


from huggingface_hub import login

login(token="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")

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

TARGET_DIM = 64
N_SAMPLES = 200
TOP_K = 10

# --- Load sentences ---
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
sentences = [x['text'] for x in dataset][:N_SAMPLES]
print(f"Loaded {len(sentences)} sentences")

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
        tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        n_layers = None
        layer_embeddings = None

        for i, sentence in enumerate(sentences):
            inputs = tok(sentence, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            if n_layers is None:
                n_layers = len(out.hidden_states)
                layer_embeddings = [[] for _ in range(n_layers)]

            for layer_idx, layer_hidden in enumerate(out.hidden_states):
                last_token = layer_hidden[0, -1, :].cpu()
                layer_embeddings[layer_idx].append(last_token)

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(sentences)}")

        result = torch.stack([torch.stack(layer) for layer in layer_embeddings])
        embeddings_cache[model_name] = result
        torch.save(result, cache_path)
        print(f"    Saved: {result.shape}")

        del model, tok
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"[-] FAILED {model_name}: {e}")

# --- Phase 2: Apply PCA to all models ---
print("\n=== Applying PCA ===")

for model_name, embeddings in embeddings_cache.items():
    n_layers = embeddings.shape[0]
    reduced = []
    variances = []

    for layer_idx in range(n_layers):
        X = embeddings[layer_idx].numpy()
        pca = PCA(n_components=TARGET_DIM)
        X_reduced = pca.fit_transform(X)
        reduced.append(torch.tensor(X_reduced))
        variances.append(pca.explained_variance_ratio_.sum())

    embeddings_pca_cache[model_name] = torch.stack(reduced)
    variance_cache[model_name] = variances
    print(f"{model_name}: {embeddings_pca_cache[model_name].shape}, var[-1]={variances[-1]:.3f}")

# --- Phase 3: Compute alignments for all pairs ---
print("\n=== Computing Alignments ===")

model_names = list(embeddings_pca_cache.keys())
best_scores = {}  # For summary matrix

for name_a, name_b in combinations(model_names, 2):
    print(f"\n[*] {name_a} <-> {name_b}")

    emb_a = embeddings_pca_cache[name_a]
    emb_b = embeddings_pca_cache[name_b]

    n_layers_a = emb_a.shape[0]
    n_layers_b = emb_b.shape[0]

    # Compute alignment matrix (skip layer 0)
    alignment_matrix = np.zeros((n_layers_a - 1, n_layers_b - 1))

    for i in range(1, n_layers_a):
        for j in range(1, n_layers_b):
            X = emb_a[i]
            Y = emb_b[j]

            n = X.shape[0]
            X_aug = torch.cat([X.float(), torch.ones(n, 1)], dim=1)
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
        layer_a = i + 1
        layer_b = j + 1
        score = alignment_matrix[i, j]
        top_pairs.append((layer_a, layer_b, score))

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