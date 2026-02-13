#!/usr/bin/env python3
"""
Correlation Analysis: Geometric Similarity vs ASR

This script correlates the geometric similarity metrics with cross-model GCG attack success rates.

For each metric, produces:
- Scatter plot with ASR on Y-axis, geometric similarity on X-axis
- Pearson and Spearman correlation coefficients
- Excludes self-attacks (source == target)

Usage:
    python correlation_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Paths - adjust these if needed
ASR_PATH = "../phase2/per_source_target_breakdown.csv"
GEOMETRY_DIR = "./geometry_output"
OUTPUT_DIR = "./correlation_plots"

# Model name mapping: ASR CSV uses different names than our geometry CSVs
NAME_MAP_ASR_TO_GEO = {
    "Llama2-7b": "llama2",
    "Llama3-8b": "llama3",
    "Vicuna-7b": "vicuna",
    "Mistral-7b": "mistral",
    "Zephyr-7b": "zephyr",
    "Hermes-2": "hermes2",
    "Starling-7b": "starling",
    "OpenChat-3.5": "openchat",
    "Gemma-7b": "gemma",
    "Phi-2": "phi2",
    "Qwen1.5-7b": "qwen",
    "Yi-6b": "yi",
    "Baichuan2-7b": "baichuan2",
    "DeepSeek-7b": "deepseek",
    "InternLM2-7b": "internlm2",
    "Falcon-7b": "falcon",
    "Solar-10.7b": "solar",
    "Orca-2-7b": "orca2",
    "NeuralChat-7b": "neuralchat",
    "StableZephyr-3b": "stablelm",
}


def load_asr_data(path: str) -> pd.DataFrame:
    """Load ASR data and filter transfer attacks only"""
    df = pd.read_csv(path)

    # Filter out self-attacks
    df_transfer = df[df['transfer_type'] == 'Transfer'].copy()

    # Map names to geometry format
    df_transfer['source'] = df_transfer['source_model'].map(NAME_MAP_ASR_TO_GEO)
    df_transfer['target'] = df_transfer['target_model'].map(NAME_MAP_ASR_TO_GEO)

    # Drop rows where mapping failed
    df_transfer = df_transfer.dropna(subset=['source', 'target'])

    return df_transfer[['source', 'target', 'asr_percent']]


def load_geometry_matrix(path: str) -> pd.DataFrame:
    """Load geometry similarity matrix"""
    return pd.read_csv(path, index_col=0)


def merge_asr_with_geometry(asr_df: pd.DataFrame, geo_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Merge ASR data with geometry similarity for each pair"""
    results = []

    for _, row in asr_df.iterrows():
        source = row['source']
        target = row['target']
        asr = row['asr_percent']

        # Get geometry similarity
        if source in geo_df.index and target in geo_df.columns:
            geo_sim = geo_df.loc[source, target]
            if not pd.isna(geo_sim):
                results.append({
                    'source': source,
                    'target': target,
                    'asr': asr,
                    'geo_sim': geo_sim,
                    'metric': metric_name
                })

    return pd.DataFrame(results)


def plot_single_correlation(merged_df: pd.DataFrame, metric_name: str, output_dir: Path) -> dict:
    """Create scatter plot for a single metric"""
    if len(merged_df) < 3:
        print(f"  Not enough data points for {metric_name}")
        return None

    x = merged_df['geo_sim'].values
    y = merged_df['asr'].values

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        print(f"  Not enough valid data points for {metric_name}")
        return None

    # Compute correlations
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label='Trend line')

    # Labels and title
    ax.set_xlabel(f'Geometric Similarity ({metric_name})', fontsize=12)
    ax.set_ylabel('Attack Success Rate (ASR %)', fontsize=12)
    ax.set_title(f'{metric_name} vs Cross-Model GCG ASR\n'
                 f'Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})\n'
                 f'Spearman r = {spearman_r:.4f} (p = {spearman_p:.4f})',
                 fontsize=14)

    # Grid
    ax.grid(True, alpha=0.3)

    # Add text box with stats
    textstr = f'n = {len(x)} pairs\nPearson r = {pearson_r:.3f}\nSpearman r = {spearman_r:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"correlation_{metric_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  {metric_name}: Pearson r = {pearson_r:.4f}, Spearman r = {spearman_r:.4f}, n = {len(x)}")
    print(f"    Saved: {output_path}")

    return {
        'metric': metric_name,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_pairs': len(x)
    }


def create_combined_plot(asr_df: pd.DataFrame, geometry_dir: Path, results_df: pd.DataFrame, output_dir: Path):
    """Create a combined figure with all metrics"""
    n_metrics = len(results_df)

    if n_metrics == 0:
        return

    # Determine grid size
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (_, row) in enumerate(results_df.iterrows()):
        metric_name = row['metric']
        ax = axes[idx]

        # Load geometry data
        geo_file = geometry_dir / f"similarity_{metric_name}_layer16.csv"
        if not geo_file.exists():
            ax.set_visible(False)
            continue

        geo_df = load_geometry_matrix(geo_file)
        merged = merge_asr_with_geometry(asr_df, geo_df, metric_name)

        if len(merged) < 3:
            ax.set_visible(False)
            continue

        x = merged['geo_sim'].values
        y = merged['asr'].values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]

        if len(x) < 3:
            ax.set_visible(False)
            continue

        ax.scatter(x, y, alpha=0.6, s=40, c='steelblue', edgecolors='white', linewidth=0.5)

        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)

        ax.set_xlabel('Geometric Similarity', fontsize=10)
        ax.set_ylabel('ASR %', fontsize=10)
        ax.set_title(f'{metric_name}\nPearson r = {row["pearson_r"]:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Geometric Similarity vs Cross-Model GCG Attack Success Rate\n(Self-attacks excluded)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / "correlation_all_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nCombined plot saved to {output_path}")


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_dir = Path(GEOMETRY_DIR)

    print("=" * 60)
    print("CORRELATION ANALYSIS: Geometric Similarity vs ASR")
    print("=" * 60)

    # Load ASR data
    print("\nLoading ASR data...")
    asr_df = load_asr_data(ASR_PATH)
    print(f"  Loaded {len(asr_df)} transfer pairs (self-attacks excluded)")

    # Find all geometry metrics
    geo_files = list(geometry_dir.glob("similarity_*.csv"))

    if not geo_files:
        print(f"\nERROR: No geometry files found in {geometry_dir}")
        print("Make sure you have run the geometric_similarity_matrix.py script first.")
        return

    print(f"\nFound {len(geo_files)} geometry metrics:")
    for f in geo_files:
        print(f"  - {f.name}")

    # Process each metric
    results = []

    print("\n" + "-" * 60)
    print("Computing correlations...")
    print("-" * 60)

    for geo_file in geo_files:
        # Extract metric name from filename
        metric_name = geo_file.stem.replace("similarity_", "").replace("_layer16", "")

        geo_df = load_geometry_matrix(geo_file)
        merged = merge_asr_with_geometry(asr_df, geo_df, metric_name)

        result = plot_single_correlation(merged, metric_name, output_dir)
        if result:
            results.append(result)

    # Create summary
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('pearson_r', ascending=False)

        print("\n" + "=" * 60)
        print("SUMMARY (sorted by Pearson r)")
        print("=" * 60)
        print(f"\n{'Metric':<20} {'Pearson r':<12} {'Spearman r':<12} {'p-value':<12} {'n pairs':<10}")
        print("-" * 66)

        for _, row in results_df.iterrows():
            p_sig = "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
            print(f"{row['metric']:<20} {row['pearson_r']:<12.4f} {row['spearman_r']:<12.4f} "
                  f"{row['pearson_p']:<12.4f} {row['n_pairs']:<10}{p_sig}")

        print("\n* p < 0.05, ** p < 0.01")

        # Save results CSV
        results_df.to_csv(output_dir / "correlation_summary.csv", index=False)
        print(f"\nSummary saved to {output_dir / 'correlation_summary.csv'}")

        # Best metric
        best = results_df.iloc[0]
        print(f"\n>>> BEST METRIC: {best['metric']} (Pearson r = {best['pearson_r']:.4f})")

        # Create combined plot
        create_combined_plot(asr_df, geometry_dir, results_df, output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
