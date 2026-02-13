#!/usr/bin/env python3
"""
Robust Geometric Similarity Matrix v2 - Dimension-Agnostic (Phase 1)

Computes pairwise geometric similarity between all 20 LLMs using metrics
that work regardless of hidden dimension differences (e.g. phi-2 at 2560 vs
LLaMA at 4096). This is the main script used for the Phase 1 analysis.

Improvements over v1 (geometric_similarity_matrix.py):
  1. All metrics work across different dimensions (no NaN holes)
  2. PCA-based methods for better mid-range discrimination
  3. Multiple complementary metrics to find best correlation with ASR

Metrics computed:
  - CKA (clean, harm, combined) - rotation & dimension invariant via Gram matrices
  - RSA (Representational Similarity Analysis) - Spearman correlation of pairwise distances
  - PCA-projected Gram similarity - project to shared PCA space, compare Gram matrices
  - PCA refusal direction similarity - compare refusal directions in PCA space
  - Cluster separation correlation - correlate per-sample harmfulness scores
  - Neighborhood preservation - overlap of k-NN sets across representation spaces
  - Variance explained correlation - similarity of PCA eigenspectra
  - Distance ratio similarity - compare within- vs between-class distance ratios

Usage:
    python robust_geometry_matrix_v2.py --list-cached
    python robust_geometry_matrix_v2.py --compute-matrix --layer-percent 0.5
    python robust_geometry_matrix_v2.py --correlate --asr-path ../outputs/asr_matrix.csv
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

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
    return Path(cache_dir) / f"{model_name}_{dataset_name}_embeddings.pt"


def load_embeddings_at_layer(
    model_name: str,
    dataset: str,
    cache_dir: str,
    layer_idx: int = None,
    layer_percent: float = None
) -> Optional[torch.Tensor]:
    """Load embeddings at specified layer (by index or percentage)."""
    cache_path = get_cache_path(model_name, dataset, cache_dir)
    if not cache_path.exists():
        return None

    embeddings = torch.load(cache_path, map_location='cpu')
    layers = sorted(embeddings.keys())

    if layer_idx is not None:
        if layer_idx in layers:
            target = layer_idx
        else:
            # Map to closest available
            target = min(layers, key=lambda x: abs(x - layer_idx))
    elif layer_percent is not None:
        target = layers[int(layer_percent * (len(layers) - 1))]
    else:
        target = layers[len(layers) // 2]

    return embeddings[target].float()


# =============================================================================
# DIMENSION-AGNOSTIC METRICS
# =============================================================================

def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    CKA (Centered Kernel Alignment) - works across ANY dimensions.
    Compares Gram matrices (n x n), not feature matrices.
    """
    n = X.shape[0]
    if n < 3:
        return 0.0

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # Center Gram matrices
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    # CKA
    hsic_kl = np.sum(K_c * L_c)
    hsic_kk = np.sum(K_c * K_c)
    hsic_ll = np.sum(L_c * L_c)

    if hsic_kk <= 0 or hsic_ll <= 0:
        return 0.0

    cka = hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return float(np.clip(cka, 0, 1))


def compute_rsa(X: np.ndarray, Y: np.ndarray) -> float:
    """
    RSA (Representational Similarity Analysis) - dimension agnostic.
    Compares pairwise distance patterns using Spearman correlation.
    """
    n = X.shape[0]
    if n < 5:
        return 0.0

    # Pairwise distances (cosine)
    dist_X = pdist(X, metric='cosine')
    dist_Y = pdist(Y, metric='cosine')

    # Handle NaN
    valid = ~(np.isnan(dist_X) | np.isnan(dist_Y))
    if valid.sum() < 10:
        return 0.0

    r, _ = spearmanr(dist_X[valid], dist_Y[valid])
    return float(r) if not np.isnan(r) else 0.0


def compute_pca_cosine_similarity(
    X: np.ndarray, Y: np.ndarray,
    n_components: int = 64
) -> float:
    """
    Project both to shared PCA space and compare refusal directions.
    Works across different dimensions by projecting to same n_components.
    """
    n_samples = X.shape[0]
    n_comp = min(n_components, X.shape[1], Y.shape[1], n_samples - 1)

    if n_comp < 2:
        return 0.0

    try:
        # PCA on each
        pca_X = PCA(n_components=n_comp, random_state=42)
        pca_Y = PCA(n_components=n_comp, random_state=42)

        X_pca = pca_X.fit_transform(X)
        Y_pca = pca_Y.fit_transform(Y)

        # Normalize
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-10)
        Y_pca = Y_pca / (np.linalg.norm(Y_pca, axis=1, keepdims=True) + 1e-10)

        # Compare Gram matrices in PCA space (CKA-like)
        K_X = X_pca @ X_pca.T
        K_Y = Y_pca @ Y_pca.T

        # Flatten and correlate
        r, _ = pearsonr(K_X.flatten(), K_Y.flatten())
        return float(r) if not np.isnan(r) else 0.0

    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def compute_refusal_direction_pca(
    clean_X: np.ndarray, harm_X: np.ndarray,
    clean_Y: np.ndarray, harm_Y: np.ndarray,
    n_components: int = 64
) -> float:
    """
    Compare refusal directions in PCA-projected space.
    This allows comparing directions even with different original dimensions.
    """
    n_comp = min(n_components,
                 clean_X.shape[1], clean_Y.shape[1],
                 clean_X.shape[0] - 1)

    if n_comp < 2:
        return 0.0

    try:
        # Combine clean+harm for each model to fit PCA
        all_X = np.vstack([clean_X, harm_X])
        all_Y = np.vstack([clean_Y, harm_Y])

        # Fit PCA
        pca_X = PCA(n_components=n_comp, random_state=42)
        pca_Y = PCA(n_components=n_comp, random_state=42)

        all_X_pca = pca_X.fit_transform(all_X)
        all_Y_pca = pca_Y.fit_transform(all_Y)

        n_clean = clean_X.shape[0]
        clean_X_pca = all_X_pca[:n_clean]
        harm_X_pca = all_X_pca[n_clean:]
        clean_Y_pca = all_Y_pca[:n_clean]
        harm_Y_pca = all_Y_pca[n_clean:]

        # Compute refusal directions in PCA space
        dir_X = harm_X_pca.mean(axis=0) - clean_X_pca.mean(axis=0)
        dir_Y = harm_Y_pca.mean(axis=0) - clean_Y_pca.mean(axis=0)

        # Normalize
        dir_X = dir_X / (np.linalg.norm(dir_X) + 1e-10)
        dir_Y = dir_Y / (np.linalg.norm(dir_Y) + 1e-10)

        # Cosine similarity (absolute value since direction sign is arbitrary)
        cosine = np.abs(np.dot(dir_X, dir_Y))
        return float(cosine)

    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def compute_cluster_separation_correlation(
    clean_X: np.ndarray, harm_X: np.ndarray,
    clean_Y: np.ndarray, harm_Y: np.ndarray
) -> float:
    """
    Compare how similarly models separate clean vs harmful samples.
    For each sample, compute its "harmfulness score" (relative position).
    Then correlate these scores across models.
    """
    try:
        # Centroids
        clean_cent_X = clean_X.mean(axis=0)
        harm_cent_X = harm_X.mean(axis=0)
        clean_cent_Y = clean_Y.mean(axis=0)
        harm_cent_Y = harm_Y.mean(axis=0)

        # For each sample: distance to harm centroid - distance to clean centroid
        # Higher = more "clean-like"
        def separation_scores(samples, clean_cent, harm_cent):
            dist_clean = np.linalg.norm(samples - clean_cent, axis=1)
            dist_harm = np.linalg.norm(samples - harm_cent, axis=1)
            return dist_harm - dist_clean

        # Compute for all samples (clean + harm concatenated)
        all_X = np.vstack([clean_X, harm_X])
        all_Y = np.vstack([clean_Y, harm_Y])

        scores_X = separation_scores(all_X, clean_cent_X, harm_cent_X)
        scores_Y = separation_scores(all_Y, clean_cent_Y, harm_cent_Y)

        r, _ = spearmanr(scores_X, scores_Y)
        return float(r) if not np.isnan(r) else 0.0

    except (ValueError, np.linalg.LinAlgError):
        return 0.0


def compute_neighborhood_preservation(
    X: np.ndarray, Y: np.ndarray,
    k: int = 10
) -> float:
    """
    Measure how well neighborhood structure is preserved.
    For each sample, find k nearest neighbors in both spaces.
    Compute overlap of neighbor sets.
    """
    n = X.shape[0]
    if n < k + 1:
        return 0.0

    try:
        # Find k nearest neighbors in each space
        nbrs_X = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
        nbrs_Y = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(Y)

        _, indices_X = nbrs_X.kneighbors(X)
        _, indices_Y = nbrs_Y.kneighbors(Y)

        # Remove self (first neighbor)
        indices_X = indices_X[:, 1:]
        indices_Y = indices_Y[:, 1:]

        # Compute overlap
        overlaps = []
        for i in range(n):
            set_X = set(indices_X[i])
            set_Y = set(indices_Y[i])
            overlap = len(set_X & set_Y) / k
            overlaps.append(overlap)

        return float(np.mean(overlaps))

    except (ValueError, np.linalg.LinAlgError):
        return 0.0


def compute_variance_explained_correlation(
    X: np.ndarray, Y: np.ndarray,
    n_components: int = 50
) -> float:
    """
    Compare PCA variance explained curves.
    Similar models should have similar intrinsic dimensionality.
    """
    n_comp = min(n_components, X.shape[1], Y.shape[1], X.shape[0] - 1)

    if n_comp < 5:
        return 0.0

    try:
        pca_X = PCA(n_components=n_comp, random_state=42).fit(X)
        pca_Y = PCA(n_components=n_comp, random_state=42).fit(Y)

        var_X = pca_X.explained_variance_ratio_
        var_Y = pca_Y.explained_variance_ratio_

        r, _ = pearsonr(var_X, var_Y)
        return float(r) if not np.isnan(r) else 0.0

    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def compute_mean_pairwise_distance_ratio(
    clean_X: np.ndarray, harm_X: np.ndarray,
    clean_Y: np.ndarray, harm_Y: np.ndarray
) -> float:
    """
    Compare the ratio of within-class to between-class distances.
    Models with similar decision boundaries should have similar ratios.
    """
    try:
        def class_distance_ratio(clean, harm):
            # Within-class distances
            within_clean = np.mean(pdist(clean, metric='cosine'))
            within_harm = np.mean(pdist(harm, metric='cosine'))
            within = (within_clean + within_harm) / 2

            # Between-class distances
            between = np.mean(cdist(clean, harm, metric='cosine'))

            return between / (within + 1e-10)

        ratio_X = class_distance_ratio(clean_X, harm_X)
        ratio_Y = class_distance_ratio(clean_Y, harm_Y)

        # Similarity based on how close the ratios are
        # Use exponential decay of absolute difference
        diff = abs(ratio_X - ratio_Y)
        similarity = np.exp(-diff)

        return float(similarity)

    except (ValueError, FloatingPointError):
        return 0.0


def compute_all_metrics(
    model_a: str, model_b: str,
    cache_dir: str,
    layer_idx: int = None,
    layer_percent: float = 0.5
) -> Dict[str, float]:
    """Compute all dimension-agnostic metrics between two models."""

    # Load embeddings
    clean_a = load_embeddings_at_layer(model_a, 'clean', cache_dir, layer_idx, layer_percent)
    harm_a = load_embeddings_at_layer(model_a, 'harm', cache_dir, layer_idx, layer_percent)
    clean_b = load_embeddings_at_layer(model_b, 'clean', cache_dir, layer_idx, layer_percent)
    harm_b = load_embeddings_at_layer(model_b, 'harm', cache_dir, layer_idx, layer_percent)

    if any(x is None for x in [clean_a, harm_a, clean_b, harm_b]):
        return {m: float('nan') for m in [
            'cka_clean', 'cka_harm', 'cka_combined',
            'rsa_clean', 'rsa_harm',
            'pca_gram_sim', 'pca_refusal_dir',
            'cluster_sep_corr', 'neighborhood_clean', 'neighborhood_harm',
            'var_explained_corr', 'distance_ratio_sim',
            'dim_a', 'dim_b'
        ]}

    # Convert to numpy
    clean_a = clean_a.numpy()
    harm_a = harm_a.numpy()
    clean_b = clean_b.numpy()
    harm_b = harm_b.numpy()

    dim_a = clean_a.shape[1]
    dim_b = clean_b.shape[1]

    # Ensure same number of samples
    n = min(clean_a.shape[0], clean_b.shape[0], harm_a.shape[0], harm_b.shape[0])
    clean_a, harm_a = clean_a[:n], harm_a[:n]
    clean_b, harm_b = clean_b[:n], harm_b[:n]

    # Combined datasets
    combined_a = np.vstack([clean_a, harm_a])
    combined_b = np.vstack([clean_b, harm_b])

    metrics = {
        'dim_a': dim_a,
        'dim_b': dim_b,
    }

    # === CKA metrics (most reliable, dimension-agnostic) ===
    metrics['cka_clean'] = compute_cka(clean_a, clean_b)
    metrics['cka_harm'] = compute_cka(harm_a, harm_b)
    metrics['cka_combined'] = compute_cka(combined_a, combined_b)

    # === RSA metrics ===
    metrics['rsa_clean'] = compute_rsa(clean_a, clean_b)
    metrics['rsa_harm'] = compute_rsa(harm_a, harm_b)

    # === PCA-based metrics ===
    metrics['pca_gram_sim'] = compute_pca_cosine_similarity(combined_a, combined_b)
    metrics['pca_refusal_dir'] = compute_refusal_direction_pca(clean_a, harm_a, clean_b, harm_b)

    # === Structure-based metrics ===
    metrics['cluster_sep_corr'] = compute_cluster_separation_correlation(
        clean_a, harm_a, clean_b, harm_b
    )
    metrics['neighborhood_clean'] = compute_neighborhood_preservation(clean_a, clean_b)
    metrics['neighborhood_harm'] = compute_neighborhood_preservation(harm_a, harm_b)

    # === Variance/dimensionality metrics ===
    metrics['var_explained_corr'] = compute_variance_explained_correlation(combined_a, combined_b)

    # === Decision boundary metrics ===
    metrics['distance_ratio_sim'] = compute_mean_pairwise_distance_ratio(
        clean_a, harm_a, clean_b, harm_b
    )

    # === Composite score (average of best metrics) ===
    composite_metrics = ['cka_combined', 'rsa_harm', 'pca_refusal_dir', 'cluster_sep_corr']
    valid_composite = [metrics[m] for m in composite_metrics if not np.isnan(metrics[m])]
    metrics['composite'] = np.mean(valid_composite) if valid_composite else 0.0

    return metrics


def compute_full_matrix(
    cache_dir: str,
    output_dir: str,
    layer_idx: int = None,
    layer_percent: float = 0.5
) -> Dict[str, pd.DataFrame]:
    """Compute 20x20 similarity matrices for all metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find available models
    available = []
    for name in MODEL_NAMES:
        if (get_cache_path(name, 'clean', cache_dir).exists() and
            get_cache_path(name, 'harm', cache_dir).exists()):
            available.append(name)

    print(f"Found {len(available)} models with embeddings")

    n = len(available)
    metrics_list = [
        'cka_clean', 'cka_harm', 'cka_combined',
        'rsa_clean', 'rsa_harm',
        'pca_gram_sim', 'pca_refusal_dir',
        'cluster_sep_corr', 'neighborhood_clean', 'neighborhood_harm',
        'var_explained_corr', 'distance_ratio_sim',
        'composite'
    ]

    matrices = {m: np.full((n, n), np.nan) for m in metrics_list}

    # Diagonal = 1.0 (self-similarity)
    for m in metrics_list:
        np.fill_diagonal(matrices[m], 1.0)

    # Compute pairwise
    total_pairs = n * (n - 1) // 2
    with tqdm(total=total_pairs, desc="Computing pairs") as pbar:
        for i, model_a in enumerate(available):
            for j, model_b in enumerate(available):
                if j <= i:
                    continue

                result = compute_all_metrics(
                    model_a, model_b, cache_dir,
                    layer_idx=layer_idx, layer_percent=layer_percent
                )

                for m in metrics_list:
                    if m in result and not np.isnan(result[m]):
                        matrices[m][i, j] = result[m]
                        matrices[m][j, i] = result[m]

                pbar.update(1)

    # Convert to DataFrames and save
    result_dfs = {}
    layer_str = f"layer{layer_idx}" if layer_idx else f"pct{int(layer_percent*100)}"

    for m in metrics_list:
        df = pd.DataFrame(matrices[m], index=available, columns=available)
        result_dfs[m] = df

        # Save CSV
        df.to_csv(output_path / f"sim_{m}_{layer_str}.csv")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.isnan(df.values)

        # Count valid values
        valid_count = (~mask).sum() - n  # Exclude diagonal
        valid_count = valid_count // 2  # Each pair counted twice

        vmin, vmax = (0, 1) if m not in ['rsa_clean', 'rsa_harm', 'cluster_sep_corr'] else (-1, 1)
        cmap = 'viridis' if vmin == 0 else 'RdBu_r'

        sns.heatmap(
            df, annot=True, fmt='.2f', cmap=cmap,
            mask=mask, ax=ax, vmin=vmin, vmax=vmax,
            annot_kws={'size': 6}
        )
        ax.set_title(f'{m} ({layer_str}) - {valid_count}/{total_pairs} valid pairs')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()

        plt.savefig(output_path / f"heatmap_{m}_{layer_str}.png", dpi=150)
        plt.close()

    # Print summary
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Valid':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)

    for m in metrics_list:
        df = result_dfs[m]
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
        vals = df.values[mask]
        valid = vals[~np.isnan(vals)]

        if len(valid) > 0:
            print(f"{m:<25} {len(valid):<8} {np.mean(valid):<10.4f} {np.std(valid):<10.4f} "
                  f"{np.min(valid):<10.4f} {np.max(valid):<10.4f}")
        else:
            print(f"{m:<25} 0        N/A")

    return result_dfs


def correlate_with_asr(
    matrices: Dict[str, pd.DataFrame],
    asr_path: str,
    output_dir: str
):
    """Correlate all metrics with ASR matrix."""

    asr_df = pd.read_csv(asr_path, index_col=0)
    print(f"\nLoaded ASR matrix: {asr_df.shape}")
    print(f"ASR models: {list(asr_df.index)}")

    output_path = Path(output_dir)
    results = []

    for metric_name, sim_df in matrices.items():
        # Find common models (handle name mismatches)
        sim_models = set(sim_df.index)
        asr_models = set(asr_df.index)

        # Try direct match first
        common = list(sim_models & asr_models)

        # If few matches, try case-insensitive or partial matching
        if len(common) < 5:
            sim_lower = {m.lower(): m for m in sim_models}
            asr_lower = {m.lower(): m for m in asr_models}
            common_lower = set(sim_lower.keys()) & set(asr_lower.keys())
            common = [(sim_lower[c], asr_lower[c]) for c in common_lower]

            if common and isinstance(common[0], tuple):
                # Rename for alignment
                sim_aligned = sim_df.copy()
                asr_aligned = asr_df.copy()
                common_sim = [c[0] for c in common]
                common_asr = [c[1] for c in common]
                sim_aligned = sim_aligned.loc[common_sim, common_sim]
                asr_aligned = asr_aligned.loc[common_asr, common_asr]
            else:
                continue
        else:
            sim_aligned = sim_df.loc[common, common]
            asr_aligned = asr_df.loc[common, common]

        n_models = len(sim_aligned)
        if n_models < 3:
            continue

        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n_models, n_models), dtype=bool), k=1)
        sim_vals = sim_aligned.values[mask]
        asr_vals = asr_aligned.values[mask]

        # Remove NaN
        valid = ~(np.isnan(sim_vals) | np.isnan(asr_vals))
        sim_vals = sim_vals[valid]
        asr_vals = asr_vals[valid]

        if len(sim_vals) < 5:
            continue

        # Compute correlations
        pearson_r, pearson_p = pearsonr(sim_vals, asr_vals)
        spearman_r, spearman_p = spearmanr(sim_vals, asr_vals)
        kendall_r, kendall_p = kendalltau(sim_vals, asr_vals)

        results.append({
            'metric': metric_name,
            'n_models': n_models,
            'n_pairs': len(sim_vals),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_r': kendall_r,
            'kendall_p': kendall_p,
        })

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sim_vals, asr_vals, alpha=0.5, s=30)
        ax.set_xlabel(f'Geometric Similarity ({metric_name})')
        ax.set_ylabel('Cross-Model ASR')
        ax.set_title(f'{metric_name} vs ASR\n'
                     f'Pearson r={pearson_r:.3f} (p={pearson_p:.3f}), '
                     f'Spearman r={spearman_r:.3f}')

        # Trend line
        if len(sim_vals) > 2:
            z = np.polyfit(sim_vals, asr_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.nanmin(sim_vals), np.nanmax(sim_vals), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)

        plt.tight_layout()
        plt.savefig(output_path / f"scatter_{metric_name}.png", dpi=150)
        plt.close()

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('spearman_r', ascending=False)
        results_df.to_csv(output_path / "correlation_results.csv", index=False)

        print("\n" + "=" * 90)
        print("CORRELATION WITH ASR (sorted by Spearman r)")
        print("=" * 90)
        print(f"{'Metric':<25} {'Pairs':<8} {'Pearson':<12} {'Spearman':<12} {'Kendall':<12}")
        print("-" * 90)

        for _, row in results_df.iterrows():
            p_str = f"{row['pearson_r']:.3f}"
            if row['pearson_p'] < 0.05:
                p_str += "*"
            if row['pearson_p'] < 0.01:
                p_str += "*"

            s_str = f"{row['spearman_r']:.3f}"
            if row['spearman_p'] < 0.05:
                s_str += "*"
            if row['spearman_p'] < 0.01:
                s_str += "*"

            k_str = f"{row['kendall_r']:.3f}"
            if row['kendall_p'] < 0.05:
                k_str += "*"

            print(f"{row['metric']:<25} {row['n_pairs']:<8} {p_str:<12} {s_str:<12} {k_str:<12}")

        print("\n* p < 0.05, ** p < 0.01")

        # Best metric recommendation
        best = results_df.iloc[0]
        print(f"\n>>> BEST METRIC: {best['metric']} (Spearman r = {best['spearman_r']:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Robust geometric similarity (dimension-agnostic, v2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--compute-matrix', action='store_true')
    parser.add_argument('--correlate', action='store_true')
    parser.add_argument('--asr-path', type=str, default='../outputs/asr_matrix.csv')
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--layer-percent', type=float, default=0.5)
    parser.add_argument('--cache-dir', type=str, default='./embeddings_cache')
    parser.add_argument('--output-dir', type=str, default='./geometry_output_v2')
    parser.add_argument('--list-cached', action='store_true')

    args = parser.parse_args()

    if args.list_cached:
        print("\nCached models:")
        print("-" * 60)
        for name in MODEL_NAMES:
            clean_path = get_cache_path(name, 'clean', args.cache_dir)
            harm_path = get_cache_path(name, 'harm', args.cache_dir)

            if clean_path.exists() and harm_path.exists():
                emb = torch.load(clean_path, map_location='cpu')
                layers = sorted(emb.keys())
                dim = emb[layers[0]].shape[1]
                n_samples = emb[layers[0]].shape[0]
                print(f"  {name:<15} OK   dim={dim:<5} layers={len(layers):<3} samples={n_samples}")
            else:
                status = []
                if not clean_path.exists():
                    status.append("clean")
                if not harm_path.exists():
                    status.append("harm")
                print(f"  {name:<15} MISSING ({', '.join(status)})")
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

            for f in output_path.glob(f"sim_*_{layer_str}.csv"):
                metric_name = f.stem.replace(f"sim_", "").replace(f"_{layer_str}", "")
                matrices[metric_name] = pd.read_csv(f, index_col=0)

            print(f"Loaded {len(matrices)} matrices from {output_path}")

        if matrices:
            correlate_with_asr(matrices, args.asr_path, args.output_dir)
        else:
            print("No matrices found. Run --compute-matrix first.")


if __name__ == "__main__":
    main()
