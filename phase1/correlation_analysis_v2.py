#!/usr/bin/env python3
"""
Correlation Analysis v2 - Geometric Similarity vs Cross-Model ASR (Phase 1)

Loads the pairwise geometry matrices produced by robust_geometry_matrix_v2.py
and the ASR (Attack Success Rate) data from Phase 2, then computes Pearson and
Spearman correlations between each geometric metric and transfer ASR across all
model pairs. Produces per-metric scatter plots and a combined top-6 summary plot.

Supports two ASR input formats:
  1. Raw per-example CSV (e.g. phase2_scored_detailed_new.csv)
     columns: source_model, target_model, is_jailbroken
  2. Pre-aggregated CSV (e.g. per_source_target_breakdown.csv)
     columns: source_model, target_model, asr_percent, transfer_type

Usage:
    python correlation_analysis_v2.py --asr-path ../phase2/phase2_scored_detailed_new.csv
    python correlation_analysis_v2.py --asr-path ../phase2/per_source_target_breakdown.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, Optional
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
ASR_PATH = Path("../phase2/phase2_scored_detailed_new.csv")
GEOMETRY_DIR = Path("./geometry_output_v2")
OUTPUT_DIR = Path("./correlation_plots_v2")

# Layer string for loading files
LAYER_STR = "pct50"  # 50% layer depth

# Model name mapping: ASR CSV names -> Geometry CSV names
NAME_MAP_ASR_TO_GEO = {
    "Llama2-7b":       "llama2",
    "Llama3-8b":       "llama3",
    "Vicuna-7b":       "vicuna",
    "Mistral-7b":      "mistral",
    "Zephyr-7b":       "zephyr",
    "Hermes-2":        "hermes2",
    "Starling-7b":     "starling",
    "OpenChat-3.5":    "openchat",
    "Gemma-7b":        "gemma",
    "Phi-2":           "phi2",
    "Qwen1.5-7b":      "qwen",
    "Yi-6b":           "yi",
    "Baichuan2-7b":    "baichuan2",
    "DeepSeek-7b":     "deepseek",
    "InternLM2-7b":    "internlm2",
    "Falcon-7b":       "falcon",
    "Solar-10.7b":     "solar",
    "Orca-2-7b":       "orca2",
    "NeuralChat-7b":   "neuralchat",
    "StableZephyr-3b": "stablelm",
}

# All v2 metrics to analyze
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

def load_asr_data(asr_path: Path) -> pd.DataFrame:
    """
    Load ASR data from either raw or aggregated CSV.
    Returns DataFrame with columns: source_geo, target_geo, asr_percent
    (self-attacks excluded).
    """
    df = pd.read_csv(asr_path)
    print(f"Loaded {asr_path.name}: {len(df)} rows, columns={list(df.columns)}")

    if 'asr_percent' in df.columns:
        # --- aggregated format (per_source_target_breakdown.csv) ---
        if 'transfer_type' in df.columns:
            df = df[df['transfer_type'] == 'Transfer'].copy()
        else:
            df = df[df['source_model'] != df['target_model']].copy()
        print(f"  Aggregated format, {len(df)} transfer pairs")

    elif 'is_jailbroken' in df.columns:
        # --- raw per-example format (phase2_scored_detailed_new.csv) ---
        df = df[df['source_model'] != df['target_model']].copy()
        agg = (
            df.groupby(['source_model', 'target_model'])['is_jailbroken']
            .agg(['sum', 'count'])
            .reset_index()
        )
        agg['asr_percent'] = (agg['sum'] / agg['count']) * 100.0
        df = agg.rename(columns={'sum': 'successful', 'count': 'total'})
        print(f"  Raw format, aggregated to {len(df)} transfer pairs")

    else:
        raise ValueError(
            f"Unrecognised ASR CSV format. Expected 'asr_percent' or "
            f"'is_jailbroken' column. Got: {list(df.columns)}"
        )

    # Map model names to geometry names
    df['source_geo'] = df['source_model'].map(NAME_MAP_ASR_TO_GEO)
    df['target_geo'] = df['target_model'].map(NAME_MAP_ASR_TO_GEO)

    unmapped = set()
    for col in ['source_model', 'target_model']:
        unmapped |= set(df[df[col].map(NAME_MAP_ASR_TO_GEO).isna()][col].unique())
    if unmapped:
        print(f"  WARNING: unmapped model names (dropped): {unmapped}")

    df = df.dropna(subset=['source_geo', 'target_geo'])
    print(f"  Final: {len(df)} pairs with mapped names")
    return df


def load_geometry_matrices(geometry_dir: Path, layer_str: str = "pct50") -> Dict[str, pd.DataFrame]:
    """Load all v2 geometry metric matrices."""

    matrices = {}

    for metric in V2_METRICS:
        csv_path = geometry_dir / f"sim_{metric}_{layer_str}.csv"
        if csv_path.exists():
            matrices[metric] = pd.read_csv(csv_path, index_col=0)
            print(f"  Loaded {metric}: {matrices[metric].shape}")
        else:
            print(f"  MISSING: {csv_path}")

    return matrices



# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def correlate_metric_with_asr(
    asr_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    metric_name: str
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int]:
    """
    Correlate a single geometry metric with ASR.
    Returns: (pearson_r, pearson_p, spearman_r, spearman_p, n_pairs)
    """

    similarities = []
    asr_values = []

    for _, row in asr_df.iterrows():
        src = row['source_geo']
        tgt = row['target_geo']
        asr = row['asr_percent']

        # Get similarity
        if src in sim_df.index and tgt in sim_df.columns:
            sim = sim_df.loc[src, tgt]
            if not np.isnan(sim) and not np.isnan(asr):
                similarities.append(sim)
                asr_values.append(asr)

    if len(similarities) < 5:
        return None, None, None, None, len(similarities)

    similarities = np.array(similarities)
    asr_values = np.array(asr_values)

    pearson_r, pearson_p = pearsonr(similarities, asr_values)
    spearman_r, spearman_p = spearmanr(similarities, asr_values)

    return pearson_r, pearson_p, spearman_r, spearman_p, len(similarities)


def create_scatter_plot(
    asr_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    pearson_r: float,
    spearman_r: float
):
    """Create a scatter plot for a single metric."""

    similarities = []
    asr_values = []
    labels = []

    for _, row in asr_df.iterrows():
        src = row['source_geo']
        tgt = row['target_geo']
        asr = row['asr_percent']

        if src in sim_df.index and tgt in sim_df.columns:
            sim = sim_df.loc[src, tgt]
            if not np.isnan(sim) and not np.isnan(asr):
                similarities.append(sim)
                asr_values.append(asr)
                labels.append(f"{src}->{tgt}")

    if len(similarities) < 3:
        return

    similarities = np.array(similarities)
    asr_values = np.array(asr_values)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(similarities, asr_values, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)

    # Trend line
    z = np.polyfit(similarities, asr_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(similarities.min(), similarities.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label='Trend line')

    ax.set_xlabel(f'Geometric Similarity ({metric_name})', fontsize=12)
    ax.set_ylabel('ASR (%)', fontsize=12)
    ax.set_title(f'{metric_name} vs Attack Success Rate\n'
                 f'Pearson r = {pearson_r:.4f}, Spearman r = {spearman_r:.4f}\n'
                 f'n = {len(similarities)} pairs',
                 fontsize=12)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"scatter_{metric_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_combined_plot(
    asr_df: pd.DataFrame,
    matrices: Dict[str, pd.DataFrame],
    results_df: pd.DataFrame,
    output_path: Path
):
    """Create a combined plot with all metrics."""

    # Get top 6 metrics by absolute Pearson r
    top_metrics = results_df.nlargest(6, 'abs_pearson')['metric'].tolist()

    n_metrics = len(top_metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(top_metrics):
        ax = axes[idx]
        sim_df = matrices[metric]

        similarities = []
        asr_values = []

        for _, row in asr_df.iterrows():
            src = row['source_geo']
            tgt = row['target_geo']
            asr = row['asr_percent']

            if src in sim_df.index and tgt in sim_df.columns:
                sim = sim_df.loc[src, tgt]
                if not np.isnan(sim) and not np.isnan(asr):
                    similarities.append(sim)
                    asr_values.append(asr)

        if len(similarities) < 3:
            continue

        similarities = np.array(similarities)
        asr_values = np.array(asr_values)

        ax.scatter(similarities, asr_values, alpha=0.5, s=30, c='steelblue')

        # Trend line
        z = np.polyfit(similarities, asr_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(similarities.min(), similarities.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)

        row_data = results_df[results_df['metric'] == metric].iloc[0]
        r = row_data['pearson_r']
        p_val = row_data['pearson_p']

        sig = ""
        if p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"

        ax.set_xlabel(f'Similarity', fontsize=10)
        ax.set_ylabel('ASR (%)', fontsize=10)
        ax.set_title(f'{metric}\nr = {r:.4f}{sig}', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(top_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Geometric Similarity vs ASR (Top 6 Metrics by |Pearson r|)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "combined_top6_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_correlation_analysis(
    asr_df: pd.DataFrame,
    matrices: Dict[str, pd.DataFrame],
    output_dir: Path
):
    """Run full correlation analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25} {'Pearson r':<12} {'p-value':<12} {'Spearman r':<12} {'n pairs':<10}")
    print("-" * 80)

    for metric_name, sim_df in matrices.items():
        pearson_r, pearson_p, spearman_r, spearman_p, n_pairs = correlate_metric_with_asr(
            asr_df, sim_df, metric_name
        )

        if pearson_r is not None:
            results.append({
                'metric': metric_name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_pairs': n_pairs,
                'abs_pearson': abs(pearson_r),
                'abs_spearman': abs(spearman_r)
            })

            # Significance markers
            sig = ""
            if pearson_p < 0.01:
                sig = "**"
            elif pearson_p < 0.05:
                sig = "*"

            print(f"{metric_name:<25} {pearson_r:>10.4f}{sig:<2} {pearson_p:<12.4f} {spearman_r:<12.4f} {n_pairs:<10}")

            # Create individual scatter plot
            create_scatter_plot(asr_df, sim_df, metric_name, output_dir, pearson_r, spearman_r)

    print("-" * 80)
    print("* p < 0.05, ** p < 0.01")

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('abs_pearson', ascending=False)
        results_df.to_csv(output_dir / "correlation_summary.csv", index=False)

        # Print best metrics
        print("\n" + "=" * 80)
        print("TOP 5 METRICS BY |PEARSON r|:")
        print("=" * 80)
        for idx, row in results_df.head(5).iterrows():
            sig = "**" if row['pearson_p'] < 0.01 else ("*" if row['pearson_p'] < 0.05 else "")
            print(f"  {row['metric']:<25} r = {row['pearson_r']:.4f}{sig}")

        # Create combined plot
        create_combined_plot(asr_df, matrices, results_df, output_dir)

        print(f"\nPlots saved to: {output_dir}")
        print(f"Summary CSV: {output_dir / 'correlation_summary.csv'}")

        return results_df

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Correlation Analysis v2')
    parser.add_argument('--asr-path', type=str, default=str(ASR_PATH))
    parser.add_argument('--geometry-dir', type=str, default=str(GEOMETRY_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--layer-str', type=str, default=LAYER_STR)
    args = parser.parse_args()

    asr_path = Path(args.asr_path)
    geometry_dir = Path(args.geometry_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CORRELATION ANALYSIS V2 - Geometric Similarity vs ASR")
    print("=" * 70)

    # 1. Load geometry matrices
    print(f"\nLoading geometry matrices from {geometry_dir} ...")
    matrices = load_geometry_matrices(geometry_dir, args.layer_str)
    if not matrices:
        print("ERROR: No geometry matrices found!")
        return
    print(f"Loaded {len(matrices)} metrics")

    # 2. Load ASR data
    print(f"\nLoading ASR data from {asr_path} ...")
    if not asr_path.exists():
        print(f"ERROR: ASR file not found at {asr_path}")
        return
    asr_df = load_asr_data(asr_path)

    # 3. Run correlation analysis
    run_correlation_analysis(asr_df, matrices, output_dir)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
