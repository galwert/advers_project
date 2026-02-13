#!/usr/bin/env python3
"""
Per-Model Correlation Plots

For each model, creates a scatter plot with:
- X-axis: Geometric similarity (from v2 metrics)
- Y-axis: ASR (%)
- Blue points: Outgoing ASR (attacks FROM this model TO others)
- Red points: Incoming ASR (attacks FROM others TO this model)
- Circle markers (o): Different model family
- X markers (x): Same model family

Usage:
    python correlation_per_model.py
    python correlation_per_model.py --metric cka_combined
    python correlation_per_model.py --output-dir ./my_plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
ASR_PATH = Path("../phase2/per_source_target_breakdown.csv")
GEOMETRY_DIR = Path("./geometry_output_v2")
OUTPUT_DIR = Path("./correlation_plots_per_model")

# Layer string for loading files
LAYER_STR = "pct50"

# Model name mapping: ASR names -> Geometry names
NAME_MAP_ASR_TO_GEO = {
    "Llama2-7b": "llama2",
    "Llama3-8b": "llama3",
    "Vicuna-7b": "vicuna",
    "Mistral-7b": "mistral",
    "Zephyr-7b": "zephyr",
    "Hermes2-7b": "hermes2",
    "Starling-7b": "starling",
    "Openchat-7b": "openchat",
    "Gemma-7b": "gemma",
    "Phi2": "phi2",
    "Qwen1.5-7b": "qwen",
    "Yi-6b": "yi",
    "Baichuan2-7b": "baichuan2",
    "Deepseek-7b": "deepseek",
    "Internlm2-7b": "internlm2",
    "Falcon-7b": "falcon",
    "Solar-10.7b": "solar",
    "Orca2-7b": "orca2",
    "Neuralchat-7b": "neuralchat",
    "Stablelm-3b": "stablelm",
}

NAME_MAP_GEO_TO_ASR = {v: k for k, v in NAME_MAP_ASR_TO_GEO.items()}

# Model families - models that share architecture/training lineage
MODEL_FAMILIES = {
    "llama": ["llama2", "llama3", "vicuna"],
    "mistral": ["mistral", "zephyr", "hermes2", "starling", "openchat", "neuralchat", "solar"],
    "qwen": ["qwen"],
    "yi": ["yi"],
    "baichuan": ["baichuan2"],
    "deepseek": ["deepseek"],
    "internlm": ["internlm2"],
    "falcon": ["falcon"],
    "phi": ["phi2"],
    "gemma": ["gemma"],
    "orca": ["orca2"],
    "stablelm": ["stablelm"],
}

# Create reverse mapping: model -> family
MODEL_TO_FAMILY = {}
for family, models in MODEL_FAMILIES.items():
    for model in models:
        MODEL_TO_FAMILY[model] = family

# All v2 metrics
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
    """Load ASR data and create a matrix format."""
    df = pd.read_csv(asr_path)
    print(f"Loaded ASR data: {len(df)} rows")

    # Filter to transfer attacks only
    if 'transfer_type' in df.columns:
        df = df[df['transfer_type'] == 'Transfer'].copy()
        print(f"After filtering to Transfer only: {len(df)} rows")

    # Map model names to geometry format
    df['source_geo'] = df['source_model'].map(NAME_MAP_ASR_TO_GEO)
    df['target_geo'] = df['target_model'].map(NAME_MAP_ASR_TO_GEO)

    # Drop unmapped
    df = df.dropna(subset=['source_geo', 'target_geo'])
    print(f"After mapping: {len(df)} valid pairs")

    return df


def load_similarity_matrix(geometry_dir: Path, metric: str, layer_str: str = "pct50") -> Optional[pd.DataFrame]:
    """Load a single similarity matrix."""
    csv_path = geometry_dir / f"sim_{metric}_{layer_str}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
        print(f"Loaded {metric}: {df.shape}")
        return df
    else:
        print(f"WARNING: {csv_path} not found")
        return None


def get_asr_matrix(asr_df: pd.DataFrame) -> pd.DataFrame:
    """Convert ASR dataframe to matrix format."""
    # Pivot to create source x target matrix
    asr_matrix = asr_df.pivot(index='source_geo', columns='target_geo', values='asr_percent')
    return asr_matrix


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def same_family(model_a: str, model_b: str) -> bool:
    """Check if two models belong to the same family."""
    family_a = MODEL_TO_FAMILY.get(model_a, model_a)
    family_b = MODEL_TO_FAMILY.get(model_b, model_b)
    return family_a == family_b


def get_plot_data_for_model(
    current_model: str,
    asr_matrix: pd.DataFrame,
    sim_matrix: pd.DataFrame
) -> Tuple[Dict, Dict]:
    """
    Get outgoing and incoming ASR data for a specific model.

    Returns:
        outgoing: dict with 'x', 'y', 'labels', 'same_family' lists
        incoming: dict with 'x', 'y', 'labels', 'same_family' lists
    """
    outgoing = {'x': [], 'y': [], 'labels': [], 'same_family': []}
    incoming = {'x': [], 'y': [], 'labels': [], 'same_family': []}

    all_models = list(set(asr_matrix.index) & set(asr_matrix.columns) &
                      set(sim_matrix.index) & set(sim_matrix.columns))

    if current_model not in all_models:
        return outgoing, incoming

    for other_model in all_models:
        if other_model == current_model:
            continue

        # Get similarity (symmetric)
        try:
            sim_val = sim_matrix.loc[current_model, other_model]
            if pd.isna(sim_val):
                continue
        except KeyError:
            continue

        is_same_fam = same_family(current_model, other_model)

        # Outgoing: current -> other
        try:
            out_asr = asr_matrix.loc[current_model, other_model]
            if not pd.isna(out_asr):
                outgoing['x'].append(sim_val)
                outgoing['y'].append(out_asr)
                outgoing['labels'].append(f"{current_model}->{other_model}")
                outgoing['same_family'].append(is_same_fam)
        except KeyError:
            pass

        # Incoming: other -> current
        try:
            in_asr = asr_matrix.loc[other_model, current_model]
            if not pd.isna(in_asr):
                incoming['x'].append(sim_val)
                incoming['y'].append(in_asr)
                incoming['labels'].append(f"{other_model}->{current_model}")
                incoming['same_family'].append(is_same_fam)
        except KeyError:
            pass

    return outgoing, incoming


# =============================================================================
# PLOTTING
# =============================================================================

def plot_single_model(
    current_model: str,
    outgoing: Dict,
    incoming: Dict,
    metric_name: str,
    output_path: Path
):
    """Create scatter plot for a single model with incoming/outgoing ASR."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot outgoing (blue) - split by same/different family
    out_x = np.array(outgoing['x'])
    out_y = np.array(outgoing['y'])
    out_same = np.array(outgoing['same_family'])

    if len(out_x) > 0:
        # Different family (circles)
        diff_mask = ~out_same
        if diff_mask.sum() > 0:
            ax.scatter(out_x[diff_mask], out_y[diff_mask],
                      c='#3498db', marker='o', s=80, alpha=0.7,
                      label='Outgoing (diff family)', edgecolors='white', linewidth=0.5)

        # Same family (X markers)
        same_mask = out_same
        if same_mask.sum() > 0:
            ax.scatter(out_x[same_mask], out_y[same_mask],
                      c='#3498db', marker='x', s=100, alpha=0.9,
                      label='Outgoing (same family)', linewidths=2)

    # Plot incoming (red) - split by same/different family
    in_x = np.array(incoming['x'])
    in_y = np.array(incoming['y'])
    in_same = np.array(incoming['same_family'])

    if len(in_x) > 0:
        # Different family (circles)
        diff_mask = ~in_same
        if diff_mask.sum() > 0:
            ax.scatter(in_x[diff_mask], in_y[diff_mask],
                      c='#e74c3c', marker='o', s=80, alpha=0.7,
                      label='Incoming (diff family)', edgecolors='white', linewidth=0.5)

        # Same family (X markers)
        same_mask = in_same
        if same_mask.sum() > 0:
            ax.scatter(in_x[same_mask], in_y[same_mask],
                      c='#e74c3c', marker='x', s=100, alpha=0.9,
                      label='Incoming (same family)', linewidths=2)

    # Add trend lines
    if len(out_x) > 2:
        z = np.polyfit(out_x, out_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(out_x.min(), out_x.max(), 100)
        ax.plot(x_line, p(x_line), '#3498db', linestyle='--', linewidth=2, alpha=0.5)

        # Calculate correlation
        r_out, p_out = pearsonr(out_x, out_y)
    else:
        r_out, p_out = np.nan, np.nan

    if len(in_x) > 2:
        z = np.polyfit(in_x, in_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(in_x.min(), in_x.max(), 100)
        ax.plot(x_line, p(x_line), '#e74c3c', linestyle='--', linewidth=2, alpha=0.5)

        r_in, p_in = pearsonr(in_x, in_y)
    else:
        r_in, p_in = np.nan, np.nan

    # Labels and formatting
    ax.set_xlabel(f'Geometric Similarity ({metric_name})', fontsize=12)
    ax.set_ylabel('ASR (%)', fontsize=12)

    # Get model family
    family = MODEL_TO_FAMILY.get(current_model, "unknown")

    title = f'{NAME_MAP_GEO_TO_ASR.get(current_model, current_model)} ({family} family)\n'
    title += f'Outgoing r={r_out:.3f}, Incoming r={r_in:.3f}'
    ax.set_title(title, fontsize=12)

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_combined_grid(
    all_models: List[str],
    asr_matrix: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    metric_name: str,
    output_path: Path
):
    """Create a grid of all model plots."""

    n_models = len(all_models)
    cols = 4
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, current_model in enumerate(all_models):
        ax = axes[i]

        outgoing, incoming = get_plot_data_for_model(current_model, asr_matrix, sim_matrix)

        # Plot outgoing (blue)
        out_x = np.array(outgoing['x']) if outgoing['x'] else np.array([])
        out_y = np.array(outgoing['y']) if outgoing['y'] else np.array([])
        out_same = np.array(outgoing['same_family']) if outgoing['same_family'] else np.array([])

        if len(out_x) > 0:
            diff_mask = ~out_same
            if diff_mask.sum() > 0:
                ax.scatter(out_x[diff_mask], out_y[diff_mask],
                          c='#3498db', marker='o', s=40, alpha=0.7)
            same_mask = out_same
            if same_mask.sum() > 0:
                ax.scatter(out_x[same_mask], out_y[same_mask],
                          c='#3498db', marker='x', s=60, alpha=0.9, linewidths=2)

        # Plot incoming (red)
        in_x = np.array(incoming['x']) if incoming['x'] else np.array([])
        in_y = np.array(incoming['y']) if incoming['y'] else np.array([])
        in_same = np.array(incoming['same_family']) if incoming['same_family'] else np.array([])

        if len(in_x) > 0:
            diff_mask = ~in_same
            if diff_mask.sum() > 0:
                ax.scatter(in_x[diff_mask], in_y[diff_mask],
                          c='#e74c3c', marker='o', s=40, alpha=0.7)
            same_mask = in_same
            if same_mask.sum() > 0:
                ax.scatter(in_x[same_mask], in_y[same_mask],
                          c='#e74c3c', marker='x', s=60, alpha=0.9, linewidths=2)

        # Add trend lines
        if len(out_x) > 2:
            z = np.polyfit(out_x, out_y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(out_x.min(), out_x.max(), 50)
            ax.plot(x_line, p(x_line), '#3498db', linestyle='--', linewidth=1.5, alpha=0.5)

        if len(in_x) > 2:
            z = np.polyfit(in_x, in_y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(in_x.min(), in_x.max(), 50)
            ax.plot(x_line, p(x_line), '#e74c3c', linestyle='--', linewidth=1.5, alpha=0.5)

        # Format
        display_name = NAME_MAP_GEO_TO_ASR.get(current_model, current_model)
        family = MODEL_TO_FAMILY.get(current_model, "unknown")
        ax.set_title(f'{display_name}\n({family})', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Axis labels on edges only
        if i % cols == 0:
            ax.set_ylabel("ASR (%)")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Similarity")

    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=10, label='Outgoing (diff family)'),
        Line2D([0], [0], marker='x', color='#3498db', markersize=10,
               linewidth=0, label='Outgoing (same family)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='Incoming (diff family)'),
        Line2D([0], [0], marker='x', color='#e74c3c', markersize=10,
               linewidth=0, label='Incoming (same family)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, 1.02))

    plt.suptitle(f'Per-Model ASR vs Similarity ({metric_name})\n'
                 f'Blue=Outgoing, Red=Incoming | Circle=Diff Family, X=Same Family',
                 fontsize=14, y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved combined grid to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Per-Model Correlation Plots')
    parser.add_argument('--metric', type=str, default='cka_combined',
                       choices=V2_METRICS, help='Similarity metric to use')
    parser.add_argument('--all-metrics', action='store_true',
                       help='Generate plots for all metrics')
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
    print("PER-MODEL CORRELATION PLOTS")
    print("=" * 70)
    print(f"ASR data: {asr_path}")
    print(f"Geometry dir: {geometry_dir}")
    print(f"Output dir: {output_dir}")

    # Load ASR data
    print("\nLoading ASR data...")
    asr_df = load_asr_data(asr_path)
    asr_matrix = get_asr_matrix(asr_df)
    print(f"ASR matrix shape: {asr_matrix.shape}")

    # Determine which metrics to process
    metrics_to_process = V2_METRICS if args.all_metrics else [args.metric]

    for metric in metrics_to_process:
        print(f"\n{'='*50}")
        print(f"Processing metric: {metric}")
        print('='*50)

        # Load similarity matrix
        sim_matrix = load_similarity_matrix(geometry_dir, metric, args.layer_str)
        if sim_matrix is None:
            print(f"Skipping {metric} - no data")
            continue

        # Get list of all models that have both ASR and similarity data
        all_models = sorted(set(asr_matrix.index) & set(asr_matrix.columns) &
                           set(sim_matrix.index) & set(sim_matrix.columns))
        print(f"Models with data: {len(all_models)}")

        # Create output subdirectory for this metric
        metric_dir = output_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual model plots
        print("\nGenerating individual model plots...")
        for model in all_models:
            outgoing, incoming = get_plot_data_for_model(model, asr_matrix, sim_matrix)

            if len(outgoing['x']) == 0 and len(incoming['x']) == 0:
                print(f"  {model}: no data, skipping")
                continue

            output_path = metric_dir / f"{model}_correlation.png"
            plot_single_model(model, outgoing, incoming, metric, output_path)
            print(f"  {model}: saved to {output_path.name}")

        # Generate combined grid plot
        print("\nGenerating combined grid plot...")
        grid_path = metric_dir / f"combined_grid_{metric}.png"
        create_combined_grid(all_models, asr_matrix, sim_matrix, metric, grid_path)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
