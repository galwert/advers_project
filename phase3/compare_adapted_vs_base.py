#!/usr/bin/env python3
"""
Compare Adapted (LoRA) Model vs Base Model - Geometric Similarity Analysis

This script computes all v2 metrics from correlation_analysis_v2.py comparing:
1. Adapted model (with LoRA defense) vs Base model
2. Generates comprehensive comparison report

Usage:
    python compare_adapted_vs_base.py --model llama3
    python compare_adapted_vs_base.py --model llama3 --layer-percent 0.5
    python compare_adapted_vs_base.py --all-models  # Run for all 7 adapted models
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import sys

sys.modules["flash_attn"] = None
warnings.filterwarnings('ignore')

# Check peft/transformers compatibility early
try:
    from peft import PeftModel
    print("PEFT imported successfully")
except ImportError as e:
    print(f"ERROR: PEFT import failed: {e}")
    print("\nThis is likely a version mismatch. Try one of:")
    print("  pip install --upgrade transformers")
    print("  pip install --upgrade peft")
    print("  pip install transformers==4.38.0 peft==0.9.0")
    raise

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configurations - maps safe_name to HuggingFace ID and model_type for prompt formatting
MODEL_CONFIGS = {
    "llama3": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_type": "llama3",
        "target_layer": 18,
    },
    "llama2": {
        "hf_id": "meta-llama/Llama-2-7b-chat-hf",
        "model_type": "llama2",
        "target_layer": 16,
    },
    "vicuna": {
        "hf_id": "lmsys/vicuna-7b-v1.5",
        "model_type": "vicuna",
        "target_layer": 16,
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_type": "mistral",
        "target_layer": 16,
    },
    "zephyr": {
        "hf_id": "HuggingFaceH4/zephyr-7b-beta",
        "model_type": "zephyr",
        "target_layer": 16,
    },
    "starling": {
        "hf_id": "berkeley-nest/Starling-LM-7B-alpha",
        "model_type": "starling",
        "target_layer": 16,
    },
    "hermes2": {
        "hf_id": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "model_type": "chatml",
        "target_layer": 16,
    },
}

# Paths
ADAPTER_DIR = Path("./adapters")
OUTPUT_DIR = Path("./defense_comparison_output")
DATASETS_DIR = Path("../datasets")
# GCG prompts with suffixes - primary source for harmful prompts
GCG_PROMPTS_PATH = Path("./outputs/advbench_suffixes_all_models_fixed.csv")
ADVBENCH_PATH = Path("./crl-llm-defense/evaluation/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv")

# V2 Metrics to compute
V2_METRICS = [
    'cka_clean', 'cka_harm', 'cka_combined',
    'rsa_clean', 'rsa_harm',
    'pca_gram_sim', 'pca_refusal_dir',
    'cluster_sep_corr', 'neighborhood_clean', 'neighborhood_harm',
    'var_explained_corr', 'distance_ratio_sim',
    'composite'
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_clean_prompts(path: Path = None, max_samples: int = 200, min_length: int = 50) -> List[str]:
    """Load clean/benign prompts for evaluation.

    Uses WikiText dataset as clean prompts (same as patchscopes_transfer.py).
    """
    if path is None:
        path = DATASETS_DIR / "alpaca_clean_200.csv"

    # Try loading from CSV first
    if path.exists():
        df = pd.read_csv(path)
        if 'instruction' in df.columns:
            prompts = df['instruction'].tolist()
        elif 'prompt' in df.columns:
            prompts = df['prompt'].tolist()
        elif 'text' in df.columns:
            prompts = df['text'].tolist()
        else:
            prompts = df.iloc[:, 0].tolist()
        print(f"Loaded {len(prompts)} clean prompts from {path}")
        return prompts[:max_samples]

    # Fallback: Use WikiText from HuggingFace
    try:
        from datasets import load_dataset
        print(f"Loading WikiText clean prompts (up to {max_samples} samples)...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

        texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) >= min_length and not text.startswith('='):
                # Clean up the text - take first sentence or so
                text = text.split('\n')[0].strip()
                if len(text) >= min_length:
                    texts.append(text[:500])  # Limit length
                    if len(texts) >= max_samples:
                        break

        print(f"Loaded {len(texts)} clean prompts from WikiText")
        return texts

    except Exception as e:
        print(f"Warning: Could not load WikiText: {e}")
        print("Using default clean prompts.")
        default_prompts = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "Write a poem about the ocean.",
            "What are the benefits of exercise?",
            "How do computers work?",
            "Describe the water cycle.",
            "What is machine learning?",
            "Explain the theory of relativity.",
            "How does the internet work?",
            "What causes earthquakes?",
        ]
        return (default_prompts * (max_samples // len(default_prompts) + 1))[:max_samples]


def load_harmful_prompts(path: Path = None, max_samples: int = 200) -> List[str]:
    """Load harmful prompts for evaluation.

    Priority:
    1. GCG prompts with suffixes (advbench_suffixes_all_models_fixed.csv)
    2. AdvBench behaviors (advbench_behaviors.csv)
    3. Other fallbacks
    """
    # Try GCG prompts first (these have adversarial suffixes)
    if path is None:
        if GCG_PROMPTS_PATH.exists():
            path = GCG_PROMPTS_PATH
            print(f"Using GCG prompts from: {path}")
        else:
            path = DATASETS_DIR / "advbench_harmful_200.csv"

    if not path.exists():
        # Try alternative paths
        alt_paths = [
            GCG_PROMPTS_PATH,  # GCG prompts with suffixes
            ADVBENCH_PATH,  # HarmBench advbench behaviors
            DATASETS_DIR / "harmful_behaviors.csv",
            DATASETS_DIR / "advbench.csv",
            Path("../crl-llm-defense/evaluation/HarmBench/data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv"),
        ]
        for alt in alt_paths:
            if alt.exists():
                path = alt
                print(f"Found harmful prompts at: {path}")
                break
        else:
            print(f"Warning: No harmful prompts file found. Using placeholders.")
            return ["How to make a bomb?"] * 200  # Placeholder

    df = pd.read_csv(path)
    print(f"Loaded harmful prompts CSV with columns: {list(df.columns)}")

    # Handle different CSV formats
    # GCG format from advbench_suffixes_all_models_fixed.csv:
    # model_index, model, example_index, prompt, target, suffix, loss
    if 'prompt' in df.columns and 'suffix' in df.columns:
        # Combine prompt + suffix for GCG attacked prompts
        # Handle potential NaN values in suffix
        df['suffix'] = df['suffix'].fillna('')
        prompts = (df['prompt'] + ' ' + df['suffix']).tolist()
        print(f"Combined prompt + suffix for GCG format")
    elif 'full_prompt' in df.columns:
        # Use full prompt with suffix if available
        prompts = df['full_prompt'].tolist()
    elif 'goal' in df.columns and 'suffix' in df.columns:
        # Combine goal + suffix
        df['suffix'] = df['suffix'].fillna('')
        prompts = (df['goal'] + ' ' + df['suffix']).tolist()
    elif 'Behavior' in df.columns:
        # advbench_behaviors.csv format
        prompts = df['Behavior'].tolist()
    elif 'goal' in df.columns:
        prompts = df['goal'].tolist()
    elif 'instruction' in df.columns:
        prompts = df['instruction'].tolist()
    elif 'prompt' in df.columns:
        prompts = df['prompt'].tolist()
    else:
        prompts = df.iloc[:, 0].tolist()

    print(f"Loaded {len(prompts)} harmful prompts")
    return prompts[:max_samples]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_base_model(model_name: str):
    """Load base model without adapter."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    config = MODEL_CONFIGS[model_name]
    hf_id = config["hf_id"]

    print(f"[Base] Loading {hf_id}...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()

    return model, tokenizer


def load_adapted_model(model_name: str, adapter_path: Path = None):
    """Load model with LoRA adapter."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    # PeftModel already imported at top level

    config = MODEL_CONFIGS[model_name]
    hf_id = config["hf_id"]

    if adapter_path is None:
        adapter_path = ADAPTER_DIR / f"{model_name}_defense"

    print(f"[Adapted] Loading {hf_id} with adapter from {adapter_path}...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    return model, tokenizer


def format_prompt(text: str, model_type: str) -> str:
    """Format prompt according to model type."""
    if model_type == "llama3":
        return f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "llama2":
        return f"[INST] {text} [/INST]"
    elif model_type == "vicuna":
        return f"USER: {text} ASSISTANT:"
    elif model_type == "mistral":
        return f"<s>[INST] {text} [/INST]"
    elif model_type == "zephyr":
        return f"<|user|>\n{text}</s>\n<|assistant|>\n"
    elif model_type == "starling":
        return f"GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:"
    elif model_type == "chatml":
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return text


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def extract_embeddings(
    model,
    tokenizer,
    prompts: List[str],
    model_type: str,
    layer_percent: float = 0.5,
    batch_size: int = 8,
    max_length: int = 128
) -> Dict[int, torch.Tensor]:
    """Extract embeddings at specified layer percentage."""
    device = next(model.parameters()).device

    # Format prompts
    formatted = [format_prompt(p, model_type) for p in prompts]

    all_embeddings = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(formatted), batch_size), desc="Extracting embeddings"):
            batch = formatted[i:i+batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

            n_layers = len(hidden_states)
            target_layer = int(layer_percent * (n_layers - 1))

            # Get last token embedding from target layer
            layer_emb = hidden_states[target_layer][:, -1, :].cpu()

            if target_layer not in all_embeddings:
                all_embeddings[target_layer] = []
            all_embeddings[target_layer].append(layer_emb)

    # Concatenate batches
    for layer in all_embeddings:
        all_embeddings[layer] = torch.cat(all_embeddings[layer], dim=0)

    return all_embeddings


# =============================================================================
# METRIC COMPUTATION (from robust_geometry_matrix_v2.py)
# =============================================================================

def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA - works across ANY dimensions."""
    n = X.shape[0]
    if n < 3:
        return 0.0

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    K = X @ X.T
    L = Y @ Y.T

    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    hsic_kl = np.sum(K_c * L_c)
    hsic_kk = np.sum(K_c * K_c)
    hsic_ll = np.sum(L_c * L_c)

    if hsic_kk <= 0 or hsic_ll <= 0:
        return 0.0

    cka = hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return float(np.clip(cka, 0, 1))


def compute_rsa(X: np.ndarray, Y: np.ndarray) -> float:
    """RSA - dimension agnostic."""
    n = X.shape[0]
    if n < 5:
        return 0.0

    dist_X = pdist(X, metric='cosine')
    dist_Y = pdist(Y, metric='cosine')

    valid = ~(np.isnan(dist_X) | np.isnan(dist_Y))
    if valid.sum() < 10:
        return 0.0

    r, _ = spearmanr(dist_X[valid], dist_Y[valid])
    return float(r) if not np.isnan(r) else 0.0


def compute_pca_gram_sim(X: np.ndarray, Y: np.ndarray, n_components: int = 64) -> float:
    """PCA-projected Gram matrix similarity."""
    n_comp = min(n_components, X.shape[1], Y.shape[1], X.shape[0] - 1)
    if n_comp < 2:
        return 0.0

    try:
        pca_X = PCA(n_components=n_comp, random_state=42)
        pca_Y = PCA(n_components=n_comp, random_state=42)

        X_pca = pca_X.fit_transform(X)
        Y_pca = pca_Y.fit_transform(Y)

        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-10)
        Y_pca = Y_pca / (np.linalg.norm(Y_pca, axis=1, keepdims=True) + 1e-10)

        K_X = X_pca @ X_pca.T
        K_Y = Y_pca @ Y_pca.T

        r, _ = pearsonr(K_X.flatten(), K_Y.flatten())
        return float(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


def compute_pca_refusal_dir(
    clean_X: np.ndarray, harm_X: np.ndarray,
    clean_Y: np.ndarray, harm_Y: np.ndarray,
    n_components: int = 64
) -> float:
    """Compare refusal directions in PCA space."""
    n_comp = min(n_components, clean_X.shape[1], clean_Y.shape[1], clean_X.shape[0] - 1)
    if n_comp < 2:
        return 0.0

    try:
        all_X = np.vstack([clean_X, harm_X])
        all_Y = np.vstack([clean_Y, harm_Y])

        pca_X = PCA(n_components=n_comp, random_state=42)
        pca_Y = PCA(n_components=n_comp, random_state=42)

        all_X_pca = pca_X.fit_transform(all_X)
        all_Y_pca = pca_Y.fit_transform(all_Y)

        n_clean = clean_X.shape[0]
        clean_X_pca = all_X_pca[:n_clean]
        harm_X_pca = all_X_pca[n_clean:]
        clean_Y_pca = all_Y_pca[:n_clean]
        harm_Y_pca = all_Y_pca[n_clean:]

        dir_X = harm_X_pca.mean(axis=0) - clean_X_pca.mean(axis=0)
        dir_Y = harm_Y_pca.mean(axis=0) - clean_Y_pca.mean(axis=0)

        dir_X = dir_X / (np.linalg.norm(dir_X) + 1e-10)
        dir_Y = dir_Y / (np.linalg.norm(dir_Y) + 1e-10)

        return float(np.abs(np.dot(dir_X, dir_Y)))
    except:
        return 0.0


def compute_cluster_sep_corr(
    clean_X: np.ndarray, harm_X: np.ndarray,
    clean_Y: np.ndarray, harm_Y: np.ndarray
) -> float:
    """Compare cluster separation patterns."""
    try:
        clean_cent_X = clean_X.mean(axis=0)
        harm_cent_X = harm_X.mean(axis=0)
        clean_cent_Y = clean_Y.mean(axis=0)
        harm_cent_Y = harm_Y.mean(axis=0)

        def separation_scores(samples, clean_cent, harm_cent):
            dist_clean = np.linalg.norm(samples - clean_cent, axis=1)
            dist_harm = np.linalg.norm(samples - harm_cent, axis=1)
            return dist_harm - dist_clean

        all_X = np.vstack([clean_X, harm_X])
        all_Y = np.vstack([clean_Y, harm_Y])

        scores_X = separation_scores(all_X, clean_cent_X, harm_cent_X)
        scores_Y = separation_scores(all_Y, clean_cent_Y, harm_cent_Y)

        r, _ = spearmanr(scores_X, scores_Y)
        return float(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


def compute_neighborhood(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Neighborhood preservation metric."""
    n = X.shape[0]
    if n < k + 1:
        return 0.0

    try:
        nbrs_X = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
        nbrs_Y = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(Y)

        _, indices_X = nbrs_X.kneighbors(X)
        _, indices_Y = nbrs_Y.kneighbors(Y)

        indices_X = indices_X[:, 1:]
        indices_Y = indices_Y[:, 1:]

        overlaps = []
        for i in range(n):
            overlap = len(set(indices_X[i]) & set(indices_Y[i])) / k
            overlaps.append(overlap)

        return float(np.mean(overlaps))
    except:
        return 0.0


def compute_var_explained_corr(X: np.ndarray, Y: np.ndarray, n_components: int = 64) -> float:
    """Compare PCA variance explained curves."""
    n_comp = min(n_components, X.shape[1], Y.shape[1], X.shape[0] - 1)
    if n_comp < 5:
        return 0.0

    try:
        pca_X = PCA(n_components=n_comp, random_state=42).fit(X)
        pca_Y = PCA(n_components=n_comp, random_state=42).fit(Y)

        r, _ = pearsonr(pca_X.explained_variance_ratio_, pca_Y.explained_variance_ratio_)
        return float(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


def compute_distance_ratio_sim(
    clean_X: np.ndarray, harm_X: np.ndarray,
    clean_Y: np.ndarray, harm_Y: np.ndarray
) -> float:
    """Compare within/between class distance ratios."""
    try:
        def class_distance_ratio(clean, harm):
            within_clean = np.mean(pdist(clean, metric='cosine'))
            within_harm = np.mean(pdist(harm, metric='cosine'))
            within = (within_clean + within_harm) / 2
            between = np.mean(cdist(clean, harm, metric='cosine'))
            return between / (within + 1e-10)

        ratio_X = class_distance_ratio(clean_X, harm_X)
        ratio_Y = class_distance_ratio(clean_Y, harm_Y)

        diff = abs(ratio_X - ratio_Y)
        return float(np.exp(-diff))
    except:
        return 0.0


def compute_all_metrics(
    clean_base: np.ndarray, harm_base: np.ndarray,
    clean_adapted: np.ndarray, harm_adapted: np.ndarray
) -> Dict[str, float]:
    """Compute all v2 metrics between base and adapted model."""

    # Align sample counts
    n = min(clean_base.shape[0], clean_adapted.shape[0],
            harm_base.shape[0], harm_adapted.shape[0])

    clean_base = clean_base[:n]
    harm_base = harm_base[:n]
    clean_adapted = clean_adapted[:n]
    harm_adapted = harm_adapted[:n]

    combined_base = np.vstack([clean_base, harm_base])
    combined_adapted = np.vstack([clean_adapted, harm_adapted])

    metrics = {}

    # CKA metrics
    metrics['cka_clean'] = compute_cka(clean_base, clean_adapted)
    metrics['cka_harm'] = compute_cka(harm_base, harm_adapted)
    metrics['cka_combined'] = compute_cka(combined_base, combined_adapted)

    # RSA metrics
    metrics['rsa_clean'] = compute_rsa(clean_base, clean_adapted)
    metrics['rsa_harm'] = compute_rsa(harm_base, harm_adapted)

    # PCA-based metrics
    metrics['pca_gram_sim'] = compute_pca_gram_sim(combined_base, combined_adapted)
    metrics['pca_refusal_dir'] = compute_pca_refusal_dir(
        clean_base, harm_base, clean_adapted, harm_adapted
    )

    # Structure metrics
    metrics['cluster_sep_corr'] = compute_cluster_sep_corr(
        clean_base, harm_base, clean_adapted, harm_adapted
    )
    metrics['neighborhood_clean'] = compute_neighborhood(clean_base, clean_adapted)
    metrics['neighborhood_harm'] = compute_neighborhood(harm_base, harm_adapted)

    # Variance metrics
    metrics['var_explained_corr'] = compute_var_explained_corr(combined_base, combined_adapted)

    # Distance ratio
    metrics['distance_ratio_sim'] = compute_distance_ratio_sim(
        clean_base, harm_base, clean_adapted, harm_adapted
    )

    # Composite
    composite_keys = ['cka_combined', 'rsa_harm', 'pca_refusal_dir', 'cluster_sep_corr']
    valid_composite = [metrics[k] for k in composite_keys if not np.isnan(metrics[k])]
    metrics['composite'] = np.mean(valid_composite) if valid_composite else 0.0

    return metrics


# =============================================================================
# ADDITIONAL ANALYSIS: Cluster Separation Changes
# =============================================================================

def analyze_cluster_separation(
    clean_base: np.ndarray, harm_base: np.ndarray,
    clean_adapted: np.ndarray, harm_adapted: np.ndarray
) -> Dict[str, float]:
    """Analyze how cluster separation changed after adaptation."""

    def compute_separation(clean, harm):
        """Compute various separation metrics."""
        # Centroid distance
        clean_cent = clean.mean(axis=0)
        harm_cent = harm.mean(axis=0)
        centroid_dist = np.linalg.norm(harm_cent - clean_cent)

        # Within/between ratio
        within_clean = np.mean(pdist(clean, metric='euclidean'))
        within_harm = np.mean(pdist(harm, metric='euclidean'))
        within = (within_clean + within_harm) / 2
        between = np.mean(cdist(clean, harm, metric='euclidean'))
        ratio = between / (within + 1e-10)

        # Linear separability (LDA-like)
        all_data = np.vstack([clean, harm])
        labels = np.array([0] * len(clean) + [1] * len(harm))

        # Project to refusal direction
        refusal_dir = harm_cent - clean_cent
        refusal_dir = refusal_dir / (np.linalg.norm(refusal_dir) + 1e-10)
        projections = all_data @ refusal_dir

        # Compute overlap
        clean_proj = projections[:len(clean)]
        harm_proj = projections[len(clean):]

        clean_mean, clean_std = clean_proj.mean(), clean_proj.std()
        harm_mean, harm_std = harm_proj.mean(), harm_proj.std()

        # Overlap coefficient (lower = better separation)
        overlap = max(0, min(clean_mean + 2*clean_std, harm_mean + 2*harm_std) -
                        max(clean_mean - 2*clean_std, harm_mean - 2*harm_std))
        total_range = max(clean_mean + 2*clean_std, harm_mean + 2*harm_std) - \
                      min(clean_mean - 2*clean_std, harm_mean - 2*harm_std)
        overlap_coef = overlap / (total_range + 1e-10)

        return {
            'centroid_dist': centroid_dist,
            'separation_ratio': ratio,
            'overlap_coefficient': overlap_coef,
        }

    base_sep = compute_separation(clean_base, harm_base)
    adapted_sep = compute_separation(clean_adapted, harm_adapted)

    results = {
        'base_centroid_dist': base_sep['centroid_dist'],
        'adapted_centroid_dist': adapted_sep['centroid_dist'],
        'centroid_dist_change': adapted_sep['centroid_dist'] - base_sep['centroid_dist'],
        'base_separation_ratio': base_sep['separation_ratio'],
        'adapted_separation_ratio': adapted_sep['separation_ratio'],
        'separation_ratio_change': adapted_sep['separation_ratio'] - base_sep['separation_ratio'],
        'base_overlap': base_sep['overlap_coefficient'],
        'adapted_overlap': adapted_sep['overlap_coefficient'],
        'overlap_change': adapted_sep['overlap_coefficient'] - base_sep['overlap_coefficient'],
    }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison_summary(
    metrics: Dict[str, float],
    separation_analysis: Dict[str, float],
    model_name: str,
    output_dir: Path
):
    """Create summary visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Metric bar chart
    ax1 = axes[0, 0]
    metric_names = [m for m in V2_METRICS if m != 'composite']
    metric_values = [metrics[m] for m in metric_names]
    colors = ['steelblue' if v >= 0.8 else 'orange' if v >= 0.5 else 'red' for v in metric_values]

    bars = ax1.barh(metric_names, metric_values, color=colors)
    ax1.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='High similarity (0.8)')
    ax1.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium similarity (0.5)')
    ax1.set_xlabel('Similarity Score')
    ax1.set_title(f'{model_name}: Base vs Adapted Model Similarity')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 1)

    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9)

    # 2. Cluster separation comparison
    ax2 = axes[0, 1]
    sep_metrics = ['centroid_dist', 'separation_ratio']
    x = np.arange(len(sep_metrics))
    width = 0.35

    base_vals = [separation_analysis[f'base_{m}'] for m in sep_metrics]
    adapted_vals = [separation_analysis[f'adapted_{m}'] for m in sep_metrics]

    ax2.bar(x - width/2, base_vals, width, label='Base Model', color='blue', alpha=0.7)
    ax2.bar(x + width/2, adapted_vals, width, label='Adapted Model', color='green', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Centroid Distance', 'Separation Ratio'])
    ax2.set_title('Cluster Separation Analysis')
    ax2.legend()

    # 3. Overlap change
    ax3 = axes[1, 0]
    overlap_data = {
        'Base': separation_analysis['base_overlap'],
        'Adapted': separation_analysis['adapted_overlap']
    }
    colors = ['blue', 'green']
    ax3.bar(overlap_data.keys(), overlap_data.values(), color=colors, alpha=0.7)
    ax3.set_ylabel('Overlap Coefficient (lower = better separation)')
    ax3.set_title('Clean/Harm Overlap Comparison')

    # Add change annotation
    change = separation_analysis['overlap_change']
    change_str = f"Change: {change:+.3f}"
    ax3.annotate(change_str, xy=(0.5, max(overlap_data.values())),
                xytext=(0.5, max(overlap_data.values()) * 1.1),
                ha='center', fontsize=12, fontweight='bold',
                color='green' if change < 0 else 'red')

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    COMPARISON SUMMARY: {model_name.upper()}
    {'='*40}

    Overall Similarity (Composite): {metrics['composite']:.4f}

    Key Findings:
    - CKA (combined): {metrics['cka_combined']:.4f}
    - RSA (harm): {metrics['rsa_harm']:.4f}
    - PCA Refusal Direction: {metrics['pca_refusal_dir']:.4f}
    - Cluster Separation Corr: {metrics['cluster_sep_corr']:.4f}

    Cluster Analysis:
    - Centroid Distance: {separation_analysis['base_centroid_dist']:.2f} -> {separation_analysis['adapted_centroid_dist']:.2f}
    - Separation Ratio: {separation_analysis['base_separation_ratio']:.2f} -> {separation_analysis['adapted_separation_ratio']:.2f}
    - Overlap: {separation_analysis['base_overlap']:.3f} -> {separation_analysis['adapted_overlap']:.3f}

    Interpretation:
    - High CKA/RSA = representations preserved
    - Lower overlap = better harmful/clean separation
    - Higher separation ratio = clearer decision boundary
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_comparison_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {output_dir / f'{model_name}_comparison_summary.png'}")


def plot_embedding_tsne(
    clean_base: np.ndarray, harm_base: np.ndarray,
    clean_adapted: np.ndarray, harm_adapted: np.ndarray,
    model_name: str,
    output_dir: Path,
    n_samples: int = 100
):
    """Create t-SNE visualization of embeddings."""
    from sklearn.manifold import TSNE

    # Subsample for speed
    n = min(n_samples, clean_base.shape[0], harm_base.shape[0])

    clean_b = clean_base[:n]
    harm_b = harm_base[:n]
    clean_a = clean_adapted[:n]
    harm_a = harm_adapted[:n]

    # Combine all
    all_emb = np.vstack([clean_b, harm_b, clean_a, harm_a])

    # Labels: 0=clean_base, 1=harm_base, 2=clean_adapted, 3=harm_adapted
    labels = np.array([0]*n + [1]*n + [2]*n + [3]*n)

    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(all_emb)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['blue', 'red', 'lightblue', 'salmon']
    labels_text = ['Base Clean', 'Base Harm', 'Adapted Clean', 'Adapted Harm']

    # Left: All together
    ax = axes[0]
    for i, (color, label) in enumerate(zip(colors, labels_text)):
        mask = labels == i
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=label, alpha=0.6, s=30)
    ax.legend()
    ax.set_title(f'{model_name}: t-SNE - All Embeddings')

    # Right: Base vs Adapted centroids
    ax = axes[1]
    for i, (color, label) in enumerate(zip(colors, labels_text)):
        mask = labels == i
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=label, alpha=0.4, s=20)
        # Centroid
        centroid = emb_2d[mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=color, s=200, marker='X', edgecolors='black')

    # Draw arrows from base to adapted centroids
    for data_type, base_idx, adapted_idx in [('clean', 0, 2), ('harm', 1, 3)]:
        base_cent = emb_2d[labels == base_idx].mean(axis=0)
        adapted_cent = emb_2d[labels == adapted_idx].mean(axis=0)
        ax.annotate('', xy=adapted_cent, xytext=base_cent,
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.legend()
    ax.set_title(f'{model_name}: Centroid Movement (X = centroid)')

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_tsne.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved t-SNE plot to {output_dir / f'{model_name}_tsne.png'}")


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def compare_model(
    model_name: str,
    layer_percent: float = 0.5,
    max_samples: int = 200,
    output_dir: Path = None,
    adapter_dir: Path = None,
    skip_tsne: bool = False
) -> Dict:
    """Run full comparison for a single model."""

    if output_dir is None:
        output_dir = OUTPUT_DIR
    if adapter_dir is None:
        adapter_dir = ADAPTER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MODEL_CONFIGS[model_name]
    model_type = config["model_type"]

    print(f"\n{'='*60}")
    print(f"COMPARING: {model_name.upper()}")
    print(f"{'='*60}")

    # Load prompts
    print("\nLoading prompts...")
    clean_prompts = load_clean_prompts(max_samples=max_samples)
    harm_prompts = load_harmful_prompts(max_samples=max_samples)
    print(f"Loaded {len(clean_prompts)} clean and {len(harm_prompts)} harmful prompts")

    # Extract base model embeddings
    print("\n--- BASE MODEL ---")
    base_model, tokenizer = load_base_model(model_name)

    base_clean_emb = extract_embeddings(
        base_model, tokenizer, clean_prompts, model_type, layer_percent
    )
    base_harm_emb = extract_embeddings(
        base_model, tokenizer, harm_prompts, model_type, layer_percent
    )

    # Get the target layer
    target_layer = list(base_clean_emb.keys())[0]
    clean_base = base_clean_emb[target_layer].numpy()
    harm_base = base_harm_emb[target_layer].numpy()

    print(f"Base embeddings shape: clean={clean_base.shape}, harm={harm_base.shape}")

    # Free memory
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Extract adapted model embeddings
    print("\n--- ADAPTED MODEL ---")
    adapter_path = adapter_dir / f"{model_name}_defense"
    adapted_model, tokenizer = load_adapted_model(model_name, adapter_path=adapter_path)

    adapted_clean_emb = extract_embeddings(
        adapted_model, tokenizer, clean_prompts, model_type, layer_percent
    )
    adapted_harm_emb = extract_embeddings(
        adapted_model, tokenizer, harm_prompts, model_type, layer_percent
    )

    clean_adapted = adapted_clean_emb[target_layer].numpy()
    harm_adapted = adapted_harm_emb[target_layer].numpy()

    print(f"Adapted embeddings shape: clean={clean_adapted.shape}, harm={harm_adapted.shape}")

    # Free memory
    del adapted_model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute metrics
    print("\n--- COMPUTING METRICS ---")
    metrics = compute_all_metrics(clean_base, harm_base, clean_adapted, harm_adapted)

    # Print metrics
    print("\nSimilarity Metrics (Base vs Adapted):")
    print("-" * 40)
    for m in V2_METRICS:
        print(f"  {m:<25}: {metrics[m]:.4f}")

    # Cluster separation analysis
    print("\n--- CLUSTER SEPARATION ANALYSIS ---")
    separation = analyze_cluster_separation(clean_base, harm_base, clean_adapted, harm_adapted)

    print("\nSeparation Changes:")
    print("-" * 40)
    print(f"  Centroid Distance: {separation['base_centroid_dist']:.2f} -> {separation['adapted_centroid_dist']:.2f} ({separation['centroid_dist_change']:+.2f})")
    print(f"  Separation Ratio:  {separation['base_separation_ratio']:.2f} -> {separation['adapted_separation_ratio']:.2f} ({separation['separation_ratio_change']:+.2f})")
    print(f"  Overlap Coef:      {separation['base_overlap']:.3f} -> {separation['adapted_overlap']:.3f} ({separation['overlap_change']:+.3f})")

    # Create visualizations
    print("\n--- CREATING VISUALIZATIONS ---")
    plot_comparison_summary(metrics, separation, model_name, output_dir)

    if not skip_tsne:
        plot_embedding_tsne(clean_base, harm_base, clean_adapted, harm_adapted,
                           model_name, output_dir)

    # Save results
    results = {
        'model': model_name,
        'layer_percent': layer_percent,
        **metrics,
        **separation
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / f"{model_name}_comparison_results.csv", index=False)

    return results


def compare_all_models(
    layer_percent: float = 0.5,
    max_samples: int = 200,
    output_dir: Path = None,
    adapter_dir: Path = None
) -> pd.DataFrame:
    """Run comparison for all available adapted models."""

    if output_dir is None:
        output_dir = OUTPUT_DIR
    if adapter_dir is None:
        adapter_dir = ADAPTER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in MODEL_CONFIGS.keys():
        adapter_path = adapter_dir / f"{model_name}_defense"

        if not adapter_path.exists():
            print(f"\nSkipping {model_name}: adapter not found at {adapter_path}")
            continue

        try:
            results = compare_model(
                model_name=model_name,
                layer_percent=layer_percent,
                max_samples=max_samples,
                output_dir=output_dir,
                adapter_dir=adapter_dir,
                skip_tsne=True  # Skip t-SNE for batch processing
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError processing {model_name}: {e}")
            continue

    if all_results:
        all_df = pd.DataFrame(all_results)
        all_df.to_csv(output_dir / "all_models_comparison.csv", index=False)

        # Create summary plot
        plot_all_models_summary(all_df, output_dir)

        return all_df

    return pd.DataFrame()


def plot_all_models_summary(df: pd.DataFrame, output_dir: Path):
    """Create summary plot for all models."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = df['model'].tolist()

    # 1. Key metrics comparison
    ax1 = axes[0, 0]
    metrics_to_plot = ['cka_combined', 'rsa_harm', 'pca_refusal_dir', 'cluster_sep_corr']
    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot)/2 + 0.5) * width
        ax1.bar(x + offset, df[metric], width, label=metric)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Similarity Score')
    ax1.set_title('Key Metrics: Base vs Adapted Model Similarity')
    ax1.legend(loc='upper right')
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)

    # 2. Composite score
    ax2 = axes[0, 1]
    colors = ['green' if v >= 0.8 else 'orange' if v >= 0.5 else 'red' for v in df['composite']]
    ax2.bar(models, df['composite'], color=colors)
    ax2.set_ylabel('Composite Score')
    ax2.set_title('Overall Similarity (Composite)')
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High (0.8)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend()

    # 3. Separation ratio change
    ax3 = axes[1, 0]
    change = df['separation_ratio_change']
    colors = ['green' if v > 0 else 'red' for v in change]
    ax3.bar(models, change, color=colors)
    ax3.set_ylabel('Change in Separation Ratio')
    ax3.set_title('Cluster Separation Improvement')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    # 4. Overlap change
    ax4 = axes[1, 1]
    overlap_change = df['overlap_change']
    colors = ['green' if v < 0 else 'red' for v in overlap_change]  # Lower is better
    ax4.bar(models, overlap_change, color=colors)
    ax4.set_ylabel('Change in Overlap (negative = better)')
    ax4.set_title('Clean/Harm Overlap Change')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / "all_models_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved all-models summary to {output_dir / 'all_models_summary.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare Adapted vs Base Model - Geometric Similarity',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, default='llama3',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to compare')
    parser.add_argument('--all-models', action='store_true',
                       help='Run comparison for all available adapted models')
    parser.add_argument('--layer-percent', type=float, default=0.5,
                       help='Layer percentage for embedding extraction')
    parser.add_argument('--max-samples', type=int, default=200,
                       help='Maximum number of samples per dataset')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--adapter-dir', type=str, default=str(ADAPTER_DIR),
                       help='Directory containing LoRA adapters')
    parser.add_argument('--skip-tsne', action='store_true',
                       help='Skip t-SNE visualization (faster)')

    args = parser.parse_args()

    # Use local variables instead of globals
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("ADAPTED VS BASE MODEL COMPARISON")
    print("=" * 60)
    print(f"Adapter directory: {adapter_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Layer percent: {args.layer_percent}")
    print(f"Max samples: {args.max_samples}")

    if args.all_models:
        results = compare_all_models(
            layer_percent=args.layer_percent,
            max_samples=args.max_samples,
            output_dir=output_dir,
            adapter_dir=adapter_dir
        )
        print("\n" + "=" * 60)
        print("ALL MODELS COMPARISON COMPLETE")
        print("=" * 60)
        if not results.empty:
            print(results[['model', 'composite', 'separation_ratio_change', 'overlap_change']].to_string())
    else:
        results = compare_model(
            model_name=args.model,
            layer_percent=args.layer_percent,
            max_samples=args.max_samples,
            output_dir=output_dir,
            adapter_dir=adapter_dir,
            skip_tsne=args.skip_tsne
        )
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
