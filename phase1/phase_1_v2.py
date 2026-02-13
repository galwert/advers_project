import torch
import numpy as np
import csv
import gc
import os
import requests
import pandas as pd
import io
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "alignment_results_cka"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of models (Your original list)
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
# --- Helper: CKA Function ---
def linear_cka(X, Y):
    """
    Computes Linear CKA (Centered Kernel Alignment)
    X, Y: (n_samples, n_features)
    """
    # Center the matrices
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute Gram matrices (dot product similarity)
    # Note: We compute the dot products of Gram matrices efficiently
    # HSIC = ||Y^T X||_F^2 / (n-1)^2

    numerator = torch.norm(torch.mm(X.T, Y), p='fro') ** 2
    denominator_x = torch.norm(torch.mm(X.T, X), p='fro')
    denominator_y = torch.norm(torch.mm(Y.T, Y), p='fro')

    return (numerator / (denominator_x * denominator_y)).item()


# --- Phase 1: Load Data ---
print("Loading datasets...")

# 1. WikiText (Benign)
wiki_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
wiki_sentences = []
for x in wiki_dataset:
    text = x['text']
    if text.strip() and len(text.strip()) > 10:
        wiki_sentences.append(text)
    if len(wiki_sentences) >= 1040:
        break
print(f"Loaded {len(wiki_sentences)} WikiText sentences")

# 2. HarmBench (Malicious) - NO DUPLICATION
try:
    ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(ADVBENCH_URL)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    hb_sentences = df['goal'].tolist()  # Use unique 520
    print(f"Loaded {len(hb_sentences)} AdvBench behaviors")
except Exception as e:
    print(f"[-] Error loading AdvBench: {e}")
    hb_sentences = []

# Combine
sentences = wiki_sentences + hb_sentences
split_index = len(wiki_sentences)  # The boundary between benign and malicious
print(f"Total samples: {len(sentences)} (Split index: {split_index})")

# --- Phase 2: Extract Embeddings (No PCA) ---
embeddings_cache = {}

for model_name, model_path in MODELS.items():
    cache_path = f"{OUTPUT_DIR}/{model_name}_raw_embeddings.pt"

    if os.path.exists(cache_path):
        print(f"[+] Loading cached {model_name}")
        embeddings_cache[model_name] = torch.load(cache_path, map_location="cpu")  # Keep on CPU until needed
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
                    mean_embedding = layer_hidden[0].mean(dim=0).cpu()
                    temp_layer_data.append(mean_embedding)
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
        print(f"Error {model_name}: {e}")

# --- Phase 3: Compute SPLIT Alignments ---
print("\n=== Computing Alignments (CKA) ===")
model_names = list(embeddings_cache.keys())

# We will store two matrices per pair: Benign and Malicious
for name_a, name_b in combinations(model_names, 2):
    print(f"[*] {name_a} <-> {name_b}")

    # Load to GPU only when needed
    emb_a = embeddings_cache[name_a].to(DEVICE).float()
    emb_b = embeddings_cache[name_b].to(DEVICE).float()

    n_layers_a = emb_a.shape[0]
    n_layers_b = emb_b.shape[0]

    # Create two matrices
    matrix_benign = np.zeros((n_layers_a, n_layers_b))
    matrix_malicious = np.zeros((n_layers_a, n_layers_b))

    for i in range(n_layers_a):
        for j in range(n_layers_b):
            # Extract full layers
            X_full = emb_a[i]  # Shape: (Total_Samples, Hidden_Dim)
            Y_full = emb_b[j]

            # 1. Benign Slice (WikiText)
            X_ben = X_full[:split_index]
            Y_ben = Y_full[:split_index]
            score_ben = linear_cka(X_ben, Y_ben)
            matrix_benign[i, j] = score_ben

            # 2. Malicious Slice (HarmBench)
            X_mal = X_full[split_index:]
            Y_mal = Y_full[split_index:]
            score_mal = linear_cka(X_mal, Y_mal)
            matrix_malicious[i, j] = score_mal

    # Save BOTH matrices
    # The thesis goal: Check if matrix_malicious correlates with ASR better than matrix_benign
    np.savetxt(f"{OUTPUT_DIR}/{name_a}_{name_b}_benign.csv", matrix_benign, delimiter=",", fmt="%.4f")
    np.savetxt(f"{OUTPUT_DIR}/{name_a}_{name_b}_malicious.csv", matrix_malicious, delimiter=",", fmt="%.4f")

    # Clean up GPU
    del emb_a, emb_b
    torch.cuda.empty_cache()

print("Done! Check separate CSVs for Benign vs Malicious alignment.")