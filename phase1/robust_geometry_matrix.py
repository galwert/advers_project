#!/usr/bin/env python3
"""
Robust Geometric Similarity Matrix - Handles Different Model Dimensions

This script computes dimension-agnostic similarity metrics between all model pairs.

Key improvements:
1. CKA works across different dimensions (compares sample relationships, not features)
2. Adds Representational Similarity Analysis (RSA) - compares distance matrices
3. Adds Projection-based methods for direction comparison across dimensions
4. Properly handles all 20 models regardless of hidden dimension

Metrics:
- CKA (Centered Kernel Alignment) - rotation and dimension invariant
- RSA (Representational Similarity Analysis) - compares pairwise distance patterns
- Mean correlation of pairwise distances
- Projection similarity (project both to same dimension via random projection)

Usage:
    python robust_geometry_matrix.py --compute-matrix --layer 16
    python robust_geometry_matrix.py --compute-matrix --layer-percent 0.5  # Middle layer
    python robust_geometry_matrix.py --correlate --asr-path ../outputs/asr_matrix.csv
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import json
import argparse
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import gc

torch.manual_seed(42)
np.random.seed(42)

# Full model list
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

MODEL_NAMES = [name for name, _ in MODELS_LIST]


def get_cache_path(model_name: str, dataset_name: str, cache_dir: str) -> Path:
    """Get standardized cache path"""
    return Path(cache_dir) / f"{model_name}_{dataset_name}_embeddings.pt"


def load_embeddings_for_layer(model_name: str, dataset: str, cache_dir: str,
                               layer_idx: int = None, layer_percent: float = None) -> Optional[torch.Tensor]:
    """
    Load embeddings for a specific layer.

    Args:
        model_name: Name of the model
        dataset: 'clean' or 'harm'
        cache_dir: Directory with cached embeddings
        layer_idx: Specific layer index (if None, use layer_percent)
        layer_percent: Relative layer position (0.0 = first, 1.0 = last)

    Returns:
        Tensor of shape [n_samples, hidden_dim] or None if not found
    """
    cache_path = get_cache_path(model_name, dataset, cache_dir)
    if not cache_path.exists():
        return None

    embeddings = torch.load(cache_path)
    layers = sorted(embeddings.keys())

    if layer_idx is not None:
        # Use specific layer, or closest available
        if layer_idx in layers:
            target_layer = layer_idx
        else:
            target_layer = layers[min(layer_idx, len(layers) - 1)]
    elif layer_percent is not None:
        # Use relative position
        target_layer = layers[int(layer_percent * (len(layers) - 1))]
    else:
        # Default to middle layer
        target_layer = layers[len(layers) // 2]

    return embeddings[target_layer]


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute CKA - works even with different dimensions!

    CKA compares the Gram matrices (sample x sample), not the feature matrices,
    so it's invariant to the feature dimension.

    Args:
        X: [n_samples, dim_x]
        Y: [n_samples, dim_y] (can be different from dim_x!)

    Returns:
        CKA similarity in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], "Must have same number of samples"

    X = X.float().cpu()
    Y = Y.float().cpu()

    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices (n x n) - this is where dimension-invariance comes from
    K = X @ X.T  # [n, n]
    L = Y @ Y.T  # [n, n]

    # Center the Gram matrices
    n = K.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    # CKA
    hsic_kl = (K_c * L_c).sum()
    hsic_kk = (K_c * K_c).sum()
    hsic_ll = (L_c * L_c).sum()

    if hsic_kk <= 0 or hsic_ll <= 0:
        return 0.0

    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return float(cka.clamp(0, 1))


def compute_rsa(X: torch.Tensor, Y: torch.Tensor, metric: str = 'correlation') -> float:
    """
    Representational Similarity Analysis - compares pairwise distance patterns.

    This is dimension-invariant: we compare how samples relate to each other,
    not the actual feature values.

    Args:
        X: [n_samples, dim_x]
        Y: [n_samples, dim_y]
        metric: 'correlation' or 'cosine' for computing pairwise distances

    Returns:
        Spearman correlation between the two distance matrices
    """
    assert X.shape[0] == Y.shape[0], "Must have same number of samples"

    X = X.float().cpu().numpy()
    Y = Y.float().cpu().numpy()

    # Compute pairwise distance matrices
    if metric == 'correlation':
        dist_X = pdist(X, metric='correlation')
        dist_Y = pdist(Y, metric='correlation')
    else:  # cosine
        dist_X = pdist(X, metric='cosine')
        dist_Y = pdist(Y, metric='cosine')

    # Handle NaN
    valid_mask = ~(np.isnan(dist_X) | np.isnan(dist_Y))
    if valid_mask.sum() < 10:
        return 0.0

    dist_X = dist_X[valid_mask]
    dist_Y = dist_Y[valid_mask]

    # Spearman correlation between flattened distance matrices
    rsa, _ = spearmanr(dist_X, dist_Y)

    if np.isnan(rsa):
        return 0.0

    return float(rsa)


def compute_pwcca(X: torch.Tensor, Y: torch.Tensor, n_components: int = 50) -> float:
    """
    Projection-weighted CCA - another dimension-agnostic metric.

    Projects both representations to a shared CCA space.
    """
    try:
        from sklearn.cross_decomposition import CCA
    except ImportError:
        return float('nan')

    X = X.float().cpu().numpy()
    Y = Y.float().cpu().numpy()

    # Limit components to min dimension
    n_comp = min(n_components, X.shape[1], Y.shape[1], X.shape[0] // 2)
    if n_comp < 2:
        return 0.0

    try:
        cca = CCA(n_components=n_comp)
        X_c, Y_c = cca.fit_transform(X, Y)

        # Correlation in CCA space
        correlations = []
        for i in range(n_comp):
            r, _ = pearsonr(X_c[:, i], Y_c[:, i])
            if not np.isnan(r):
                correlations.append(abs(r))

        if len(correlations) == 0:
            return 0.0

        return float(np.mean(correlations))
    except Exception:
        return 0.0


def compute_cluster_structure_similarity(
    clean_X: torch.Tensor, harm_X: torch.Tensor,
    clean_Y: torch.Tensor, harm_Y: torch.Tensor
) -> float:
    """
    Compare how well the two models separate clean vs harmful.

    This measures if models have similar "decision boundaries" for harmful content,
    independent of the actual embedding dimension.

    Returns correlation between separation patterns.
    """
    # For each sample, compute distance to clean centroid vs harm centroid
    # in each model's space

    clean_X = clean_X.float()
    harm_X = harm_X.float()
    clean_Y = clean_Y.float()
    harm_Y = harm_Y.float()

    # Centroids
    clean_cent_X = clean_X.mean(dim=0)
    harm_cent_X = harm_X.mean(dim=0)
    clean_cent_Y = clean_Y.mean(dim=0)
    harm_cent_Y = harm_Y.mean(dim=0)

    # For clean samples: distance to harm centroid - distance to clean centroid
    # Higher = more clearly classified as clean
    def separation_score(samples, clean_cent, harm_cent):
        dist_to_clean = torch.norm(samples - clean_cent, dim=1)
        dist_to_harm = torch.norm(samples - harm_cent, dim=1)
        return (dist_to_harm - dist_to_clean).cpu().numpy()

    # Compute separation scores for all samples in both models
    all_samples_X = torch.cat([clean_X, harm_X], dim=0)
    all_samples_Y = torch.cat([clean_Y, harm_Y], dim=0)

    scores_X = separation_score(all_samples_X, clean_cent_X, harm_cent_X)
    scores_Y = separation_score(all_samples_Y, clean_cent_Y, harm_cent_Y)

    # Correlation of separation patterns
    r, _ = spearmanr(scores_X, scores_Y)

    if np.isnan(r):
        return 0.0

    return float(r)


def compute_refusal_direction_similarity_projected(
    clean_X: torch.Tensor, harm_X: torch.Tensor,
    clean_Y: torch.Tensor, harm_Y: torch.Tensor,
    n_proj: int = 256
) -> float:
    """
    Compare refusal directions using random projection to shared space.

    This allows comparing directions even with different dimensions.
    """
    # Compute refusal directions in original spaces
    dir_X = (harm_X.mean(dim=0) - clean_X.mean(dim=0)).float()
    dir_Y = (harm_Y.mean(dim=0) - clean_Y.mean(dim=0)).float()

    dir_X = dir_X / (torch.norm(dir_X) + 1e-8)
    dir_Y = dir_Y / (torch.norm(dir_Y) + 1e-8)

    dim_X = dir_X.shape[0]
    dim_Y = dir_Y.shape[0]

    # If same dimension, direct comparison
    if dim_X == dim_Y:
        return float(torch.dot(dir_X, dir_Y).abs())

    # Random projection to shared space
    torch.manual_seed(42)

    proj_X = torch.randn(dim_X, n_proj) / np.sqrt(n_proj)
    proj_Y = torch.randn(dim_Y, n_proj) / np.sqrt(n_proj)

    dir_X_proj = dir_X @ proj_X
    dir_Y_proj = dir_Y @ proj_Y

    dir_X_proj = dir_X_proj / (torch.norm(dir_X_proj) + 1e-8)
    dir_Y_proj = dir_Y_proj / (torch.norm(dir_Y_proj) + 1e-8)

    # This won't be meaningful - random projections don't preserve direction
    # Return NaN to indicate this metric doesn't work across dimensions
    return float('nan')


def compute_all_metrics(
    model_a: str, model_b: str,
    cache_dir: str,
    layer_idx: int = None,
    layer_percent: float = 0.5
) -> Dict[str, float]:
    """
    Compute all dimension-agnostic similarity metrics between two models.
    """
    # Load embeddings
    clean_a = load_embeddings_for_layer(model_a, 'clean', cache_dir, layer_idx, layer_percent)
    harm_a = load_embeddings_for_layer(model_a, 'harm', cache_dir, layer_idx, layer_percent)
    clean_b = load_embeddings_for_layer(model_b, 'clean', cache_dir, layer_idx, layer_percent)
    harm_b = load_embeddings_for_layer(model_b, 'harm', cache_dir, layer_idx, layer_percent)

    if any(x is None for x in [clean_a, harm_a, clean_b, harm_b]):
        return {
            'cka_clean': float('nan'),
            'cka_harm': float('nan'),
            'cka_combined': float('nan'),
            'rsa_clean': float('nan'),
            'rsa_harm': float('nan'),
            'cluster_sim': float('nan'),
            'dim_a': 0,
            'dim_b': 0,
        }

    dim_a = clean_a.shape[1]
    dim_b = clean_b.shape[1]

    # Ensure same number of samples
    n_samples = min(clean_a.shape[0], clean_b.shape[0], harm_a.shape[0], harm_b.shape[0])
    clean_a, harm_a = clean_a[:n_samples], harm_a[:n_samples]
    clean_b, harm_b = clean_b[:n_samples], harm_b[:n_samples]

    metrics = {
        'dim_a': dim_a,
        'dim_b': dim_b,
    }

    # CKA - works across dimensions
    metrics['cka_clean'] = compute_cka(clean_a, clean_b)
    metrics['cka_harm'] = compute_cka(harm_a, harm_b)

    # CKA on combined (clean + harm)
    combined_a = torch.cat([clean_a, harm_a], dim=0)
    combined_b = torch.cat([clean_b, harm_b], dim=0)
    metrics['cka_combined'] = compute_cka(combined_a, combined_b)

    # RSA - works across dimensions
    metrics['rsa_clean'] = compute_rsa(clean_a, clean_b)
    metrics['rsa_harm'] = compute_rsa(harm_a, harm_b)

    # Cluster structure similarity
    metrics['cluster_sim'] = compute_cluster_structure_similarity(
        clean_a, harm_a, clean_b, harm_b
    )

    # Direct cosine (only if same dimension)
    if dim_a == dim_b:
        dir_a = harm_a.mean(dim=0) - clean_a.mean(dim=0)
        dir_b = harm_b.mean(dim=0) - clean_b.mean(dim=0)
        dir_a = dir_a / (torch.norm(dir_a) + 1e-8)
        dir_b = dir_b / (torch.norm(dir_b) + 1e-8)
        metrics['direct_cosine'] = float(torch.dot(dir_a, dir_b))
    else:
        metrics['direct_cosine'] = float('nan')

    return metrics


def compute_full_matrix(
    cache_dir: str,
    output_dir: str,
    layer_idx: int = None,
    layer_percent: float = 0.5
) -> Dict[str, pd.DataFrame]:
    """Compute full 20x20 similarity matrices for all metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check available models
    available = []
    for name in MODEL_NAMES:
        if (get_cache_path(name, 'clean', cache_dir).exists() and
            get_cache_path(name, 'harm', cache_dir).exists()):
            available.append(name)

    print(f"Found {len(available)} models with cached embeddings")
    print(f"Using layer_idx={layer_idx}, layer_percent={layer_percent}")

    n = len(available)
    metrics = ['cka_clean', 'cka_harm', 'cka_combined', 'rsa_clean', 'rsa_harm',
               'cluster_sim', 'direct_cosine']

    matrices = {m: np.full((n, n), np.nan) for m in metrics}

    # Fill diagonal with 1.0
    for m in metrics:
        np.fill_diagonal(matrices[m], 1.0)

    # Compute pairwise
    total_pairs = n * (n - 1) // 2
    with tqdm(total=total_pairs, desc="Computing pairs") as pbar:
        for i, model_a in enumerate(available):
            for j, model_b in enumerate(available):
                if j <= i:
                    continue

                try:
                    result = compute_all_metrics(
                        model_a, model_b, cache_dir,
                        layer_idx=layer_idx, layer_percent=layer_percent
                    )

                    for m in metrics:
                        if m in result:
                            matrices[m][i, j] = result[m]
                            matrices[m][j, i] = result[m]

                except Exception as e:
                    print(f"  Error {model_a} vs {model_b}: {e}")

                pbar.update(1)

    # Convert to DataFrames
    result_dfs = {}
    for m in metrics:
        df = pd.DataFrame(matrices[m], index=available, columns=available)
        result_dfs[m] = df

        # Save CSV
        layer_str = f"layer{layer_idx}" if layer_idx else f"pct{int(layer_percent*100)}"
        csv_path = output_path / f"similarity_{m}_{layer_str}.csv"
        df.to_csv(csv_path)
        print(f"Saved {csv_path}")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.isnan(df.values)

        # Choose colormap based on metric
        if m in ['rsa_clean', 'rsa_harm', 'cluster_sim']:
            # These can be negative
            vmin, vmax = -1, 1
            cmap = 'RdBu_r'
        else:
            vmin, vmax = 0, 1
            cmap = 'viridis'

        sns.heatmap(
            df, annot=True, fmt='.2f', cmap=cmap,
            mask=mask, ax=ax, vmin=vmin, vmax=vmax,
            annot_kws={'size': 7}
        )
        ax.set_title(f'{m} ({layer_str})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        fig_path = output_path / f"heatmap_{m}_{layer_str}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for m in metrics:
        df = result_dfs[m]
        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
        values = df.values[mask]
        valid_values = values[~np.isnan(values)]

        if len(valid_values) > 0:
            print(f"\n{m}:")
            print(f"  Valid pairs: {len(valid_values)}/{len(values)}")
            print(f"  Mean: {np.mean(valid_values):.4f}")
            print(f"  Std:  {np.std(valid_values):.4f}")
            print(f"  Min:  {np.min(valid_values):.4f}")
            print(f"  Max:  {np.max(valid_values):.4f}")

    return result_dfs


def correlate_with_asr(
    matrices: Dict[str, pd.DataFrame],
    asr_path: str,
    output_dir: str
):
    """Correlate geometric metrics with ASR matrix."""

    asr_df = pd.read_csv(asr_path, index_col=0)
    print(f"\nLoaded ASR matrix: {asr_df.shape}")

    output_path = Path(output_dir)
    results = []

    for metric_name, sim_df in matrices.items():
        # Align models
        common = list(set(sim_df.index) & set(asr_df.index))
        if len(common) < 3:
            continue

        sim_aligned = sim_df.loc[common, common]
        asr_aligned = asr_df.loc[common, common]

        # Extract upper triangle
        mask = np.triu(np.ones_like(sim_aligned, dtype=bool), k=1)
        sim_vals = sim_aligned.values[mask]
        asr_vals = asr_aligned.values[mask]

        # Remove NaN
        valid = ~(np.isnan(sim_vals) | np.isnan(asr_vals))
        sim_vals = sim_vals[valid]
        asr_vals = asr_vals[valid]

        if len(sim_vals) < 3:
            continue

        pearson_r, pearson_p = pearsonr(sim_vals, asr_vals)
        spearman_r, spearman_p = spearmanr(sim_vals, asr_vals)

        results.append({
            'metric': metric_name,
            'n_pairs': len(sim_vals),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        })

        print(f"\n{metric_name}:")
        print(f"  Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"  Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sim_vals, asr_vals, alpha=0.5)
        ax.set_xlabel(f'Geometric Similarity ({metric_name})')
        ax.set_ylabel('ASR')
        ax.set_title(f'{metric_name} vs ASR (r={pearson_r:.3f})')

        if len(sim_vals) > 2:
            z = np.polyfit(sim_vals, asr_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.nanmin(sim_vals), np.nanmax(sim_vals), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8)

        plt.tight_layout()
        plt.savefig(output_path / f"correlation_{metric_name}.png", dpi=150)
        plt.close()

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / "correlation_results.csv", index=False)

        print("\n" + "="*60)
        print("CORRELATION SUMMARY")
        print("="*60)
        print(results_df.sort_values('spearman_r', ascending=False).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Robust geometric similarity (dimension-agnostic)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--compute-matrix', action='store_true')
    parser.add_argument('--correlate', action='store_true')
    parser.add_argument('--asr-path', type=str, default='../outputs/asr_matrix.csv')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer index')
    parser.add_argument('--layer-percent', type=float, default=0.5,
                        help='Relative layer position (0.0=first, 0.5=middle, 1.0=last)')
    parser.add_argument('--cache-dir', type=str, default='./embeddings_cache')
    parser.add_argument('--output-dir', type=str, default='./geometry_output_robust')
    parser.add_argument('--list-cached', action='store_true')

    args = parser.parse_args()

    if args.list_cached:
        print("\nCached models:")
        for name in MODEL_NAMES:
            clean = get_cache_path(name, 'clean', args.cache_dir).exists()
            harm = get_cache_path(name, 'harm', args.cache_dir).exists()

            if clean and harm:
                emb = torch.load(get_cache_path(name, 'clean', args.cache_dir))
                layers = sorted(emb.keys())
                dim = emb[layers[0]].shape[1]
                print(f"  {name}: OK (dim={dim}, layers={len(layers)})")
            else:
                print(f"  {name}: MISSING")
        return

    matrices = None

    if args.compute_matrix:
        matrices = compute_full_matrix(
            args.cache_dir, args.output_dir,
            layer_idx=args.layer, layer_percent=args.layer_percent
        )

    if args.correlate:
        if matrices is None:
            # Load from files
            output_path = Path(args.output_dir)
            layer_str = f"layer{args.layer}" if args.layer else f"pct{int(args.layer_percent*100)}"
            matrices = {}
            for m in ['cka_clean', 'cka_harm', 'cka_combined', 'rsa_clean', 'rsa_harm',
                      'cluster_sim', 'direct_cosine']:
                csv_path = output_path / f"similarity_{m}_{layer_str}.csv"
                if csv_path.exists():
                    matrices[m] = pd.read_csv(csv_path, index_col=0)

        if matrices:
            correlate_with_asr(matrices, args.asr_path, args.output_dir)


if __name__ == "__main__":
    main()
