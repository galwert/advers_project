#!/usr/bin/env python3
"""
Patchscopes-inspired Refusal Direction Transfer Analysis

This script implements cross-model refusal direction transfer using Procrustes alignment,
inspired by the Patchscopes paper methodology for comparing representation spaces across models.

Key features:
1. Extracts embeddings from WikiText (clean) and HarmBench (malicious) prompts
2. Optionally includes GCG-attacked prompts
3. Computes refusal directions at each layer
4. Performs Procrustes alignment to map between model spaces
5. Transfers refusal directions and evaluates effectiveness
6. ALWAYS saves embeddings as .pt files for later use
7. Sanity check: mapping model to itself should yield similarity ~1.0

Usage:
    # Basic usage - llama2 to vicuna
    python patchscopes_transfer.py --source llama2 --target vicuna

    # Sanity check - llama2 to llama2 (should get ~1.0 similarity)
    python patchscopes_transfer.py --source llama2 --target llama2 --sanity-check

    # Load from cached embeddings
    python patchscopes_transfer.py --source llama2 --target vicuna --load-cache
"""

import io
import requests
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from scipy.linalg import orthogonal_procrustes
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import hashlib

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

# Full model list with indices matching the GCG CSV format
# Index in this list = model_index in the CSV
MODELS_LIST = [
    ("llama2", "meta-llama/Llama-2-7b-chat-hf"),           # 0
    ("llama3", "meta-llama/Meta-Llama-3-8B-Instruct"),     # 1
    ("vicuna", "lmsys/vicuna-7b-v1.5"),                    # 2
    ("mistral", "mistralai/Mistral-7B-Instruct-v0.2"),     # 3
    ("zephyr", "HuggingFaceH4/zephyr-7b-beta"),            # 4
    ("hermes2", "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"),  # 5
    ("starling", "berkeley-nest/Starling-LM-7B-alpha"),    # 6
    ("openchat", "openchat/openchat_3.5"),                 # 7
    ("gemma", "google/gemma-7b-it"),                       # 8
    ("phi2", "microsoft/phi-2"),                           # 9
    ("qwen", "Qwen/Qwen1.5-7B-Chat"),                      # 10
    ("yi", "01-ai/Yi-6B-Chat"),                            # 11
    ("baichuan2", "baichuan-inc/Baichuan2-7B-Chat"),       # 12
    ("deepseek", "deepseek-ai/deepseek-llm-7b-chat"),      # 13
    ("internlm2", "internlm/internlm2-chat-7b"),           # 14
    ("falcon", "tiiuae/falcon-7b-instruct"),               # 15
    ("solar", "upstage/SOLAR-10.7B-Instruct-v1.0"),        # 16
    ("orca2", "microsoft/Orca-2-7b"),                      # 17
    ("neuralchat", "Intel/neural-chat-7b-v3-1"),           # 18
    ("stablelm", "stabilityai/stablelm-zephyr-3b"),        # 19
]

# Create lookup dictionaries
MODEL_PATHS = {name: path for name, path in MODELS_LIST}
MODEL_NAME_TO_INDEX = {name: idx for idx, (name, path) in enumerate(MODELS_LIST)}
MODEL_INDEX_TO_NAME = {idx: name for idx, (name, path) in enumerate(MODELS_LIST)}


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    name: str
    path: str
    max_length: int = 512
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class ExperimentConfig:
    """Configuration for the experiment"""
    num_wikitext: int = 520
    num_harmbench: int = 520
    num_gcg: int = 100
    layer_range: Optional[Tuple[int, int]] = None
    batch_size: int = 8
    use_gcg: bool = False
    gcg_path: str = "../outputs/advbench_suffixes_all_models_fixed.csv"
    output_dir: str = "./patchscopes_output"
    cache_dir: str = "./embeddings_cache"
    position: str = "last"  # "last", "first", or "mean"
    sanity_check: bool = False  # If True, source == target for validation


class EmbeddingExtractor:
    """Extracts hidden state embeddings from transformer models"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Get number of layers
        if hasattr(model.config, 'num_hidden_layers'):
            self.num_layers = model.config.num_hidden_layers
        else:
            self.num_layers = len(model.model.layers)

        # Get hidden dimension
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        else:
            self.hidden_dim = None

    @torch.no_grad()
    def extract_embeddings(
            self,
            texts: List[str],
            layers: Optional[List[int]] = None,
            batch_size: int = 8,
            position: str = "last"
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from specified layers

        Args:
            texts: List of input texts
            layers: List of layer indices to extract (None = all layers)
            batch_size: Batch size for processing
            position: Which token position to use ("last", "first", "mean")

        Returns:
            Dictionary mapping layer_idx -> tensor of shape [num_texts, hidden_dim]
        """
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list, got {type(texts)}")

        # Ensure all texts are strings
        texts = [str(t) if not isinstance(t, str) else t for t in texts]

        if layers is None:
            layers = list(range(self.num_layers))

        embeddings = {layer: [] for layer in layers}

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            for layer_idx in layers:
                # hidden_states is tuple: (embed_layer, layer_0, ..., layer_n)
                hidden_state = outputs.hidden_states[layer_idx + 1]

                if position == "last":
                    attention_mask = inputs['attention_mask']
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    batch_embeds = hidden_state[torch.arange(hidden_state.size(0)), seq_lengths]
                elif position == "first":
                    batch_embeds = hidden_state[:, 0, :]
                elif position == "mean":
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    batch_embeds = (hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                else:
                    raise ValueError(f"Unknown position: {position}")

                embeddings[layer_idx].append(batch_embeds.cpu())

        # Concatenate all batches
        for layer_idx in layers:
            embeddings[layer_idx] = torch.cat(embeddings[layer_idx], dim=0)

        return embeddings


def load_wikitext(num_samples: int = 520, min_length: int = 50) -> List[str]:
    """Load WikiText examples"""
    print(f"Loading {num_samples} WikiText examples...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) >= min_length and not text.startswith('='):
            texts.append(text)
        if len(texts) >= num_samples:
            break

    return texts[:num_samples]


def load_harmbench(num_samples: int = 520) -> List[str]:
    """Load HarmBench/AdvBench prompts"""
    print(f"Loading {num_samples} HarmBench examples...")
    try:
        response = requests.get(ADVBENCH_URL)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download: HTTP {response.status_code}")
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={"goal": "prompt"})
        subset = df.head(num_samples)
        print(f"Loaded {len(subset)} behaviors from AdvBench.")
        return subset['prompt'].astype(str).tolist()
    except Exception as e:
        print(f"Failed to download AdvBench: {repr(e)}")
        return []


def load_gcg_prompts(
    csv_path: str,
    model_name: str = None,
    model_index: int = None,
    num_samples: int = 100
) -> List[str]:
    """
    Load GCG-attacked prompts for specific model

    Args:
        csv_path: Path to the GCG suffixes CSV file
        model_name: Friendly model name (e.g., 'llama2', 'vicuna')
        model_index: Direct model index (0-19). If model_name is provided, this is ignored.
        num_samples: Maximum number of samples to load

    Returns:
        List of prompts with GCG suffixes appended

    CSV Format expected:
        model_index,model,example_index,prompt,target,suffix,loss
    """
    print(f"Loading GCG prompts from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Resolve model_index from model_name if provided
    if model_name is not None:
        if model_name not in MODEL_NAME_TO_INDEX:
            available = list(MODEL_NAME_TO_INDEX.keys())
            raise ValueError(f"Unknown model name: {model_name}. Available: {available}")
        model_index = MODEL_NAME_TO_INDEX[model_name]
        print(f"  Model '{model_name}' -> index {model_index}")
    elif model_index is None:
        model_index = 0  # Default to llama2

    # Filter for specific model
    model_df = df[df['model_index'] == model_index]

    if len(model_df) == 0:
        print(f"  WARNING: No GCG prompts found for model_index={model_index}")
        print(f"  Available model indices in CSV: {sorted(df['model_index'].unique())}")
        return []

    # Get unique model name from CSV for verification
    csv_model_name = model_df['model'].iloc[0] if 'model' in model_df.columns else 'unknown'
    print(f"  Found {len(model_df)} GCG examples for model: {csv_model_name}")

    # Take requested number of samples
    model_df = model_df.head(num_samples)

    # Combine prompt + suffix
    gcg_prompts = []
    for _, row in model_df.iterrows():
        prompt = str(row['prompt'])
        suffix = str(row['suffix']) if pd.notna(row['suffix']) else ""
        full_prompt = f"{prompt} {suffix}".strip()
        gcg_prompts.append(full_prompt)

    print(f"  Loaded {len(gcg_prompts)} GCG prompts (requested: {num_samples})")
    return gcg_prompts


def compute_refusal_direction(
        clean_embeds: torch.Tensor,
        malicious_embeds: torch.Tensor,
        method: str = "mean_diff",
        normalize: bool = True
) -> torch.Tensor:
    """
    Compute refusal direction vector

    Args:
        clean_embeds: [num_clean, hidden_dim]
        malicious_embeds: [num_malicious, hidden_dim]
        method: "mean_diff", "pca", or "contrastive"
        normalize: Whether to normalize the direction vector

    Returns:
        Direction vector of shape [hidden_dim]
    """
    if method == "mean_diff":
        direction = malicious_embeds.mean(dim=0) - clean_embeds.mean(dim=0)
    elif method == "pca":
        diff_vectors = []
        for mal_embed in malicious_embeds:
            diffs = mal_embed.unsqueeze(0) - clean_embeds
            diff_vectors.append(diffs.mean(dim=0))
        diff_vectors = torch.stack(diff_vectors)
        U, S, V = torch.pca_lowrank(diff_vectors, q=1)
        direction = V[:, 0]
    elif method == "contrastive":
        mean_clean = clean_embeds.mean(dim=0)
        mean_mal = malicious_embeds.mean(dim=0)
        direction = mean_mal - mean_clean
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        direction = direction / (torch.norm(direction) + 1e-8)

    return direction


def compute_procrustes_alignment(
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        is_same_model: bool = False,
        use_subset: bool = True,
        subset_size: int = 256
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal orthogonal Procrustes rotation matrix

    Args:
        source_embeds: [num_anchors, hidden_dim] from source model
        target_embeds: [num_anchors, hidden_dim] from target model
        is_same_model: If True, return identity matrix (sanity check mode)
        use_subset: If True, use a random subset for numerical stability
        subset_size: Size of subset to use (helps with SVD convergence)

    Returns:
        R: Rotation matrix [hidden_dim, hidden_dim]
        disparity: Frobenius norm of alignment error
    """
    hidden_dim = source_embeds.shape[1]

    # For sanity check: if source == target, use identity
    if is_same_model:
        return np.eye(hidden_dim), 0.0

    X = source_embeds.cpu().numpy().astype(np.float64)
    Y = target_embeds.cpu().numpy().astype(np.float64)

    # Use subset for numerical stability (large matrices can cause SVD issues)
    if use_subset and X.shape[0] > subset_size:
        np.random.seed(42)  # Reproducibility
        indices = np.random.choice(X.shape[0], subset_size, replace=False)
        X = X[indices]
        Y = Y[indices]

    # Handle inf/nan
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        print("Warning: Input contains inf/nan values, cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # Center the data
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Compute Frobenius norms for scaling (more stable than per-dimension std)
    X_norm = np.linalg.norm(X_centered, 'fro') + 1e-10
    Y_norm = np.linalg.norm(Y_centered, 'fro') + 1e-10

    X_normalized = X_centered / X_norm
    Y_normalized = Y_centered / Y_norm

    if not np.isfinite(X_normalized).all() or not np.isfinite(Y_normalized).all():
        print("Warning: Normalized data contains inf/nan, cleaning...")
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        Y_normalized = np.nan_to_num(Y_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # Try Procrustes with fallback methods
    R = None
    disparity = float('inf')

    # Method 1: scipy orthogonal_procrustes
    try:
        R, disparity = orthogonal_procrustes(X_normalized, Y_normalized)
        if not np.isfinite(R).all():
            R = None
    except Exception as e:
        print(f"  Procrustes (scipy) failed: {e}")
        R = None

    # Method 2: Manual SVD with numpy (often more stable)
    if R is None:
        try:
            # Procrustes: find R that minimizes ||X @ R - Y||
            # Solution: R = V @ U.T where U, S, V = svd(Y.T @ X)
            M = Y_normalized.T @ X_normalized
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt
            # Ensure it's a proper rotation (det = 1)
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            disparity = np.linalg.norm(X_normalized @ R - Y_normalized, 'fro')
            print("  Used numpy SVD fallback for Procrustes")
        except Exception as e:
            print(f"  Procrustes (numpy SVD) failed: {e}")
            R = None

    # Method 3: Identity matrix fallback
    if R is None or not np.isfinite(R).all():
        print("  Warning: All Procrustes methods failed. Using identity matrix.")
        R = np.eye(hidden_dim)
        disparity = float('inf')

    return R, disparity


def evaluate_direction_similarity(
        dir1: torch.Tensor,
        dir2: torch.Tensor
) -> Dict[str, float]:
    """Compute similarity metrics between two direction vectors"""
    # Normalize both directions
    dir1_norm = dir1 / (torch.norm(dir1) + 1e-8)
    dir2_norm = dir2 / (torch.norm(dir2) + 1e-8)

    # Cosine similarity
    cosine_sim = torch.dot(dir1_norm, dir2_norm)

    # Euclidean distance (on normalized vectors)
    euclidean_dist = torch.norm(dir1_norm - dir2_norm).item()

    # Angular distance
    angular_dist = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0)).item()

    return {
        "cosine_similarity": cosine_sim.item(),
        "euclidean_distance": euclidean_dist,
        "angular_distance": angular_dist
    }


def get_cache_path(model_name: str, dataset_name: str, cache_dir: str) -> Path:
    """Get standardized cache path for embeddings"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"{model_name}_{dataset_name}_embeddings.pt"


def save_embeddings(embeddings: Dict[int, torch.Tensor], path: Path):
    """Save embeddings to .pt file"""
    torch.save(embeddings, path)
    print(f"Saved embeddings to {path}")


def load_embeddings(path: Path) -> Optional[Dict[int, torch.Tensor]]:
    """Load embeddings from .pt file if exists"""
    if path.exists():
        print(f"Loading cached embeddings from {path}")
        return torch.load(path)
    return None


def compute_cka_similarity(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two sets of embeddings.
    CKA is rotation-invariant, so it measures structural similarity without alignment.

    Args:
        X: [n_samples, dim_x] embeddings from model 1
        Y: [n_samples, dim_y] embeddings from model 2

    Returns:
        CKA similarity score in [0, 1]
    """
    X = X.float()
    Y = Y.float()

    # Center the data
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute Gram matrices (linear kernel)
    K = X @ X.T
    L = Y @ Y.T

    # Center the Gram matrices (HSIC)
    n = K.shape[0]
    H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    # CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    hsic_kl = (K_centered * L_centered).sum()
    hsic_kk = (K_centered * K_centered).sum()
    hsic_ll = (L_centered * L_centered).sum()

    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return cka.item()


def analyze_layer_wise_transfer(
        source_embeddings: Dict[str, Dict[int, torch.Tensor]],
        target_embeddings: Dict[str, Dict[int, torch.Tensor]],
        layers: List[int],
        output_dir: Path,
        is_same_model: bool = False,
        alignment_anchors: str = "clean"  # "clean", "harm", "both", or "combined"
) -> pd.DataFrame:
    """
    Analyze refusal direction transfer across all layers with multiple alignment strategies.

    Args:
        source_embeddings: Dict of dataset_name -> {layer_idx -> embeddings}
        target_embeddings: Dict of dataset_name -> {layer_idx -> embeddings}
        layers: List of layer indices to analyze
        output_dir: Directory to save results
        is_same_model: If True, this is a sanity check (source == target)
        alignment_anchors: Which data to use for Procrustes alignment

    Returns:
        DataFrame with metrics for each layer
    """
    results = []

    for layer_idx in tqdm(layers, desc="Analyzing layers"):
        source_clean = source_embeddings['clean'][layer_idx]
        source_harm = source_embeddings['harm'][layer_idx]
        target_clean = target_embeddings['clean'][layer_idx]
        target_harm = target_embeddings['harm'][layer_idx]

        # Compute refusal directions (normalized)
        source_direction = compute_refusal_direction(source_clean, source_harm, normalize=True)
        target_direction = compute_refusal_direction(target_clean, target_harm, normalize=True)

        # Also compute raw (unnormalized) for diagnostics
        source_direction_raw = compute_refusal_direction(source_clean, source_harm, normalize=False)
        target_direction_raw = compute_refusal_direction(target_clean, target_harm, normalize=False)

        # === BASELINE: Direct comparison without alignment ===
        direct_similarity = evaluate_direction_similarity(source_direction, target_direction)

        # === CKA: Rotation-invariant similarity of the embedding spaces ===
        cka_clean = compute_cka_similarity(source_clean, target_clean)
        cka_harm = compute_cka_similarity(source_harm, target_harm)

        if is_same_model:
            # Sanity check: no alignment needed
            R_clean = np.eye(source_direction.shape[0])
            disparity_clean = 0.0
            transferred_direction = source_direction.clone()
            transferred_via_harm = source_direction.clone()
            transferred_via_combined = source_direction.clone()
        else:
            # === Strategy 1: Align using CLEAN data (original approach) ===
            R_clean, disparity_clean = compute_procrustes_alignment(
                source_clean, target_clean, is_same_model=False
            )
            R_clean_tensor = torch.tensor(R_clean, dtype=source_direction.dtype)
            transferred_direction = R_clean_tensor @ source_direction

            # === Strategy 2: Align using HARMFUL data ===
            R_harm, disparity_harm = compute_procrustes_alignment(
                source_harm, target_harm, is_same_model=False
            )
            R_harm_tensor = torch.tensor(R_harm, dtype=source_direction.dtype)
            transferred_via_harm = R_harm_tensor @ source_direction

            # === Strategy 3: Align using COMBINED (clean + harm) data ===
            source_combined = torch.cat([source_clean, source_harm], dim=0)
            target_combined = torch.cat([target_clean, target_harm], dim=0)
            R_combined, disparity_combined = compute_procrustes_alignment(
                source_combined, target_combined, is_same_model=False
            )
            R_combined_tensor = torch.tensor(R_combined, dtype=source_direction.dtype)
            transferred_via_combined = R_combined_tensor @ source_direction

        # Evaluate all transfer strategies
        sim_via_clean = evaluate_direction_similarity(transferred_direction, target_direction)
        sim_via_harm = evaluate_direction_similarity(transferred_via_harm, target_direction)
        sim_via_combined = evaluate_direction_similarity(transferred_via_combined, target_direction)

        # Compute cluster separability (how well-separated are clean vs harm in each model)
        source_sep = torch.norm(source_direction_raw).item()  # Magnitude of mean difference
        target_sep = torch.norm(target_direction_raw).item()

        result = {
            'layer': layer_idx,
            # Direct (no alignment) - baseline
            'direct_cosine_sim': direct_similarity['cosine_similarity'],
            # Procrustes with clean anchors (original)
            'cosine_similarity': sim_via_clean['cosine_similarity'],  # Keep original name for compatibility
            'procrustes_disparity': disparity_clean,
            # Procrustes with harm anchors
            'cosine_sim_harm_align': sim_via_harm['cosine_similarity'],
            # Procrustes with combined anchors
            'cosine_sim_combined_align': sim_via_combined['cosine_similarity'],
            # CKA (rotation-invariant)
            'cka_clean': cka_clean,
            'cka_harm': cka_harm,
            # Best alignment strategy
            'best_cosine_sim': max(
                sim_via_clean['cosine_similarity'],
                sim_via_harm['cosine_similarity'],
                sim_via_combined['cosine_similarity']
            ),
            # Cluster separability
            'source_separation': source_sep,
            'target_separation': target_sep,
            # Direction norms
            'source_direction_norm': torch.norm(source_direction).item(),
            'target_direction_norm': torch.norm(target_direction).item(),
            # Embedding norms
            'source_clean_mean_norm': torch.norm(source_clean, dim=1).mean().item(),
            'source_harm_mean_norm': torch.norm(source_harm, dim=1).mean().item(),
            'target_clean_mean_norm': torch.norm(target_clean, dim=1).mean().item(),
            'target_harm_mean_norm': torch.norm(target_harm, dim=1).mean().item(),
            # Angular distances
            'angular_distance': sim_via_clean['angular_distance'],
            'euclidean_distance': sim_via_clean['euclidean_distance'],
        }

        # GCG analysis if available
        if 'gcg' in source_embeddings:
            source_gcg = source_embeddings['gcg'][layer_idx]
            gcg_direction = compute_refusal_direction(source_clean, source_gcg)

            if is_same_model:
                transferred_gcg = gcg_direction.clone()
            else:
                R_clean_tensor = torch.tensor(R_clean, dtype=gcg_direction.dtype)
                transferred_gcg = R_clean_tensor @ gcg_direction

            gcg_metrics = evaluate_direction_similarity(transferred_gcg, target_direction)
            result['gcg_cosine_similarity'] = gcg_metrics['cosine_similarity']
            result['gcg_angular_distance'] = gcg_metrics['angular_distance']

        results.append(result)

    df = pd.DataFrame(results)
    output_path = output_dir / "layer_wise_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved layer-wise analysis to {output_path}")

    return df


def plot_analysis_results(df: pd.DataFrame, output_dir: Path, title_prefix: str = ""):
    """Create visualization plots with multiple alignment strategies"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Compare alignment strategies
    ax = axes[0, 0]
    ax.plot(df['layer'], df['direct_cosine_sim'], marker='x', label='No alignment', linewidth=2, alpha=0.7)
    ax.plot(df['layer'], df['cosine_similarity'], marker='o', label='Clean anchors', linewidth=2)
    ax.plot(df['layer'], df['cosine_sim_harm_align'], marker='s', label='Harm anchors', linewidth=2)
    ax.plot(df['layer'], df['cosine_sim_combined_align'], marker='^', label='Combined anchors', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'{title_prefix}Alignment Strategies Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot 2: CKA similarity (rotation-invariant)
    ax = axes[0, 2]
    ax.plot(df['layer'], df['cka_clean'], marker='o', label='CKA (clean)', linewidth=2, color='blue')
    ax.plot(df['layer'], df['cka_harm'], marker='s', label='CKA (harm)', linewidth=2, color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA Similarity')
    ax.set_title(f'{title_prefix}CKA (Rotation-Invariant)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot 3: Best alignment + GCG
    ax = axes[1, 0]
    ax.plot(df['layer'], df['best_cosine_sim'], marker='o', label='Best alignment', linewidth=2, color='green')
    if 'gcg_cosine_similarity' in df.columns:
        ax.plot(df['layer'], df['gcg_cosine_similarity'], marker='s', label='GCG', linewidth=2, color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'{title_prefix}Best Alignment & GCG')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot 4: Cluster separability
    ax = axes[1, 1]
    ax.plot(df['layer'], df['source_separation'], marker='o', label='Source sep.', linewidth=2)
    ax.plot(df['layer'], df['target_separation'], marker='s', label='Target sep.', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('||mean(harm) - mean(clean)||')
    ax.set_title(f'{title_prefix}Cluster Separability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Procrustes disparity
    ax = axes[1, 2]
    ax.plot(df['layer'], df['procrustes_disparity'], marker='o', color='red', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Procrustes Disparity')
    ax.set_title(f'{title_prefix}Alignment Quality')
    ax.grid(True, alpha=0.3)
    if df['procrustes_disparity'].max() > 0:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "analysis_plots.png", dpi=300, bbox_inches='tight')
    print(f"Saved plots to {output_dir / 'analysis_plots.png'}")
    plt.close()


def print_available_models():
    """Print table of available models and their indices"""
    print("\nAvailable models (index corresponds to GCG CSV model_index):")
    print("-" * 60)
    print(f"{'Index':<6} {'Name':<15} {'HuggingFace Path'}")
    print("-" * 60)
    for idx, (name, path) in enumerate(MODELS_LIST):
        print(f"{idx:<6} {name:<15} {path}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Patchscopes-inspired refusal direction transfer analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # List models option
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')

    # Model arguments
    parser.add_argument('--source', type=str, choices=list(MODEL_PATHS.keys()),
                        help='Source model')
    parser.add_argument('--target', type=str, choices=list(MODEL_PATHS.keys()),
                        help='Target model')

    # Data arguments
    parser.add_argument('--num-wikitext', type=int, default=520)
    parser.add_argument('--num-harmbench', type=int, default=520)
    parser.add_argument('--use-gcg', action='store_true',
                        help='Include GCG-attacked prompts in analysis')
    parser.add_argument('--num-gcg', type=int, default=100,
                        help='Number of GCG examples to use')
    parser.add_argument('--gcg-path', type=str, default='../outputs/advbench_suffixes_all_models_fixed.csv',
                        help='Path to GCG suffixes CSV')
    parser.add_argument('--gcg-model', type=str, default=None, choices=list(MODEL_PATHS.keys()),
                        help='Model to load GCG suffixes from (default: same as --source)')

    # Processing arguments
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--position', type=str, default='last', choices=['last', 'first', 'mean'])
    parser.add_argument('--layers', type=int, nargs=2, metavar=('START', 'END'), default=None)

    # Cache/output arguments
    parser.add_argument('--output-dir', type=str, default='./patchscopes_output')
    parser.add_argument('--cache-dir', type=str, default='./embeddings_cache')
    parser.add_argument('--load-cache', action='store_true', help='Load embeddings from cache if available')
    parser.add_argument('--no-plots', action='store_true')

    # Sanity check mode
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity check (source == target should give similarity ~1.0)')

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print_available_models()
        return

    # Validate required arguments
    if not args.source or not args.target:
        parser.error("--source and --target are required (unless using --list-models)")

    # Sanity check validation
    is_same_model = (args.source == args.target)
    if args.sanity_check and not is_same_model:
        print("Warning: --sanity-check specified but source != target. Setting target = source.")
        args.target = args.source
        is_same_model = True

    if is_same_model:
        print("=" * 80)
        print("SANITY CHECK MODE: Source == Target")
        print("Expected: Cosine similarity should be ~1.0 for all layers")
        print("=" * 80)

    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get model paths
    source_path = MODEL_PATHS[args.source]
    target_path = MODEL_PATHS[args.target]

    print("=" * 80)
    print("PATCHSCOPES REFUSAL DIRECTION TRANSFER")
    print("=" * 80)
    print(f"Source: {args.source} ({source_path})")
    print(f"Target: {args.target} ({target_path})")
    print(f"Output: {output_dir}")
    print(f"Cache: {cache_dir}")

    # Load datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    wikitext = load_wikitext(args.num_wikitext)
    harmbench = load_harmbench(args.num_harmbench)

    datasets = {'clean': wikitext, 'harm': harmbench}

    if args.use_gcg and Path(args.gcg_path).exists():
        # Load GCG prompts - use --gcg-model if specified, otherwise use source model
        gcg_model = args.gcg_model if args.gcg_model else args.source
        gcg_prompts = load_gcg_prompts(
            args.gcg_path,
            model_name=gcg_model,
            num_samples=args.num_gcg
        )
        if len(gcg_prompts) > 0:
            datasets['gcg'] = gcg_prompts
        else:
            print(f"WARNING: No GCG prompts found for {gcg_model}, skipping GCG analysis")

    print(f"\nDataset sizes:")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} examples")

    # Load/extract embeddings
    source_embeddings = {}
    target_embeddings = {}

    # Try to load from cache
    source_cache_available = all(
        get_cache_path(args.source, ds, args.cache_dir).exists()
        for ds in datasets.keys()
    )
    target_cache_available = all(
        get_cache_path(args.target, ds, args.cache_dir).exists()
        for ds in datasets.keys()
    )

    if args.load_cache and source_cache_available:
        print("\n" + "=" * 80)
        print("LOADING SOURCE EMBEDDINGS FROM CACHE")
        print("=" * 80)
        for ds_name in datasets.keys():
            cache_path = get_cache_path(args.source, ds_name, args.cache_dir)
            source_embeddings[ds_name] = load_embeddings(cache_path)
        layers = list(source_embeddings['clean'].keys())
    else:
        # Load source model and extract
        print("\n" + "=" * 80)
        print("LOADING SOURCE MODEL")
        print("=" * 80)

        source_model = AutoModelForCausalLM.from_pretrained(
            source_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        source_tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
        if source_tokenizer.pad_token is None:
            source_tokenizer.pad_token = source_tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        source_extractor = EmbeddingExtractor(source_model, source_tokenizer, device)

        # Determine layers
        if args.layers:
            layers = list(range(args.layers[0], args.layers[1]))
        else:
            layers = list(range(source_extractor.num_layers))

        print(f"\nExtracting embeddings from {len(layers)} layers...")

        for ds_name, texts in datasets.items():
            print(f"\n--- {ds_name.upper()} dataset ---")
            source_embeddings[ds_name] = source_extractor.extract_embeddings(
                texts, layers=layers, batch_size=args.batch_size, position=args.position
            )
            # ALWAYS save to cache
            cache_path = get_cache_path(args.source, ds_name, args.cache_dir)
            save_embeddings(source_embeddings[ds_name], cache_path)

        # Free source model memory
        del source_model
        del source_extractor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # For sanity check, target embeddings = source embeddings
    if is_same_model:
        print("\n" + "=" * 80)
        print("SANITY CHECK: Using source embeddings as target")
        print("=" * 80)
        target_embeddings = source_embeddings
    elif args.load_cache and target_cache_available:
        print("\n" + "=" * 80)
        print("LOADING TARGET EMBEDDINGS FROM CACHE")
        print("=" * 80)
        for ds_name in datasets.keys():
            cache_path = get_cache_path(args.target, ds_name, args.cache_dir)
            target_embeddings[ds_name] = load_embeddings(cache_path)
    else:
        # Load target model and extract
        print("\n" + "=" * 80)
        print("LOADING TARGET MODEL")
        print("=" * 80)

        target_model = AutoModelForCausalLM.from_pretrained(
            target_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        target_tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
        if target_tokenizer.pad_token is None:
            target_tokenizer.pad_token = target_tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        target_extractor = EmbeddingExtractor(target_model, target_tokenizer, device)

        # Use same layers as source
        target_layers = [l for l in layers if l < target_extractor.num_layers]

        for ds_name, texts in datasets.items():
            print(f"\n--- {ds_name.upper()} dataset ---")
            target_embeddings[ds_name] = target_extractor.extract_embeddings(
                texts, layers=target_layers, batch_size=args.batch_size, position=args.position
            )
            # ALWAYS save to cache
            cache_path = get_cache_path(args.target, ds_name, args.cache_dir)
            save_embeddings(target_embeddings[ds_name], cache_path)

        del target_model
        del target_extractor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Update layers to common set
        layers = target_layers

    # Analyze transfer
    print("\n" + "=" * 80)
    print("ANALYZING LAYER-WISE TRANSFER")
    print("=" * 80)

    results_df = analyze_layer_wise_transfer(
        source_embeddings,
        target_embeddings,
        layers,
        output_dir,
        is_same_model=is_same_model
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Statistics for different alignment strategies
    print(f"\n=== Alignment Strategy Comparison ===")
    print(f"{'Strategy':<25} {'Mean':<10} {'Max':<10} {'Best Layer':<12}")
    print("-" * 57)

    strategies = [
        ('No alignment', 'direct_cosine_sim'),
        ('Clean anchors', 'cosine_similarity'),
        ('Harm anchors', 'cosine_sim_harm_align'),
        ('Combined anchors', 'cosine_sim_combined_align'),
        ('Best of all', 'best_cosine_sim'),
    ]

    for name, col in strategies:
        mean_val = results_df[col].mean()
        max_val = results_df[col].max()
        best_layer = int(results_df.loc[results_df[col].idxmax(), 'layer'])
        print(f"{name:<25} {mean_val:<10.4f} {max_val:<10.4f} {best_layer:<12}")

    # CKA statistics
    print(f"\n=== CKA (Rotation-Invariant) ===")
    print(f"  Clean embeddings: mean={results_df['cka_clean'].mean():.4f}, max={results_df['cka_clean'].max():.4f}")
    print(f"  Harm embeddings:  mean={results_df['cka_harm'].mean():.4f}, max={results_df['cka_harm'].max():.4f}")

    # Cluster separability
    print(f"\n=== Cluster Separability (||harm_mean - clean_mean||) ===")
    print(f"  Source: mean={results_df['source_separation'].mean():.4f}")
    print(f"  Target: mean={results_df['target_separation'].mean():.4f}")

    mean_sim = results_df['cosine_similarity'].mean()
    max_sim = results_df['cosine_similarity'].max()
    min_sim = results_df['cosine_similarity'].min()
    best_sim = results_df['best_cosine_sim'].max()

    if is_same_model:
        print(f"\n*** SANITY CHECK RESULT ***")
        if mean_sim > 0.99:
            print(f"  PASSED: Mean similarity = {mean_sim:.4f} (expected ~1.0)")
        else:
            print(f"  WARNING: Mean similarity = {mean_sim:.4f} (expected ~1.0)")
            print(f"  Check if embeddings are being computed consistently!")

    print(f"\n=== Top 5 Layers (Best Alignment Strategy) ===")
    top_layers = results_df.nlargest(5, 'best_cosine_sim')[
        ['layer', 'best_cosine_sim', 'direct_cosine_sim', 'cka_clean', 'cka_harm']
    ]
    print(top_layers.to_string(index=False))

    # Identify which layers have good similarity in the 10-14 "safety layers" range
    safety_layers = results_df[(results_df['layer'] >= 10) & (results_df['layer'] <= 14)]
    if len(safety_layers) > 0:
        print(f"\n=== Safety Layers (10-14) Analysis ===")
        print(f"  Best cosine sim mean: {safety_layers['best_cosine_sim'].mean():.4f}")
        print(f"  Best cosine sim max:  {safety_layers['best_cosine_sim'].max():.4f} (layer {int(safety_layers.loc[safety_layers['best_cosine_sim'].idxmax(), 'layer'])})")
        print(f"  CKA clean mean: {safety_layers['cka_clean'].mean():.4f}")
        print(f"  CKA harm mean:  {safety_layers['cka_harm'].mean():.4f}")

    # Create visualizations
    if not args.no_plots:
        title_prefix = f"{args.source} -> {args.target}: "
        if is_same_model:
            title_prefix = f"Sanity Check ({args.source}): "
        plot_analysis_results(results_df, output_dir, title_prefix)

    # Save summary
    summary = {
        'source_model': args.source,
        'target_model': args.target,
        'is_sanity_check': is_same_model,
        'num_layers_analyzed': len(layers),
        'cache_dir': str(cache_dir),
        'datasets': {name: len(data) for name, data in datasets.items()},
        'alignment_comparison': {
            'no_alignment': {
                'mean': float(results_df['direct_cosine_sim'].mean()),
                'max': float(results_df['direct_cosine_sim'].max()),
                'best_layer': int(results_df.loc[results_df['direct_cosine_sim'].idxmax(), 'layer'])
            },
            'clean_anchors': {
                'mean': float(results_df['cosine_similarity'].mean()),
                'max': float(results_df['cosine_similarity'].max()),
                'best_layer': int(results_df.loc[results_df['cosine_similarity'].idxmax(), 'layer'])
            },
            'harm_anchors': {
                'mean': float(results_df['cosine_sim_harm_align'].mean()),
                'max': float(results_df['cosine_sim_harm_align'].max()),
                'best_layer': int(results_df.loc[results_df['cosine_sim_harm_align'].idxmax(), 'layer'])
            },
            'combined_anchors': {
                'mean': float(results_df['cosine_sim_combined_align'].mean()),
                'max': float(results_df['cosine_sim_combined_align'].max()),
                'best_layer': int(results_df.loc[results_df['cosine_sim_combined_align'].idxmax(), 'layer'])
            },
            'best_overall': {
                'mean': float(results_df['best_cosine_sim'].mean()),
                'max': float(results_df['best_cosine_sim'].max()),
                'best_layer': int(results_df.loc[results_df['best_cosine_sim'].idxmax(), 'layer'])
            }
        },
        'cka_similarity': {
            'clean_mean': float(results_df['cka_clean'].mean()),
            'harm_mean': float(results_df['cka_harm'].mean())
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Embeddings cached in: {cache_dir}")
    print(f"\nCached .pt files:")
    for ds_name in datasets.keys():
        print(f"  - {get_cache_path(args.source, ds_name, args.cache_dir)}")
        if not is_same_model:
            print(f"  - {get_cache_path(args.target, ds_name, args.cache_dir)}")


if __name__ == "__main__":
    main()
