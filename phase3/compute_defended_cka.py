#!/usr/bin/env python3
"""
Compute cross-layer CKA between defended Mistral (LoRA-merged) and baseline Llama-2.
Produces before/after heatmaps for the paper.

Usage:
    python compute_defended_cka.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
CACHE_DIR = Path("../phase1/embeddings_cache")
OUTPUT_DIR = Path("defended_cka_output")
ADAPTER_PATH = "layer_sweep_outputs/mistral_layer0.5/defender_v2_cka_20260223_193031"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Reuse CKA computation from phase1 script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "phase1"))
from all_layers_cka import (
    precompute_grams,
    compute_interpolated_diagonal,
)


def extract_defended_embeddings():
    """Load Mistral + LoRA adapter, extract all-layer embeddings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from datasets import load_dataset
    import requests, io

    # Check if already cached
    clean_cache = CACHE_DIR / "mistral_defended_clean_embeddings.pt"
    harm_cache = CACHE_DIR / "mistral_defended_harm_embeddings.pt"
    if clean_cache.exists() and harm_cache.exists():
        print("Defended embeddings already cached, skipping extraction.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load base model + LoRA
    print("Loading base Mistral model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))
    print(f"  Layers: {num_layers}")

    # Load datasets (same as phase1 script)
    print("Loading WikiText...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    wikitext = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) >= 50 and not text.startswith('='):
            wikitext.append(text)
        if len(wikitext) >= 520:
            break
    wikitext = wikitext[:520]

    print("Loading AdvBench...")
    ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    resp = requests.get(ADVBENCH_URL)
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.rename(columns={"goal": "prompt"})
    harmbench = df.head(520)['prompt'].astype(str).tolist()

    datasets_map = {'clean': wikitext, 'harm': harmbench}
    batch_size = 8

    for ds_name, texts in datasets_map.items():
        print(f"  Extracting {ds_name}...")
        embeddings = {layer: [] for layer in layers}

        for i in tqdm(range(0, len(texts), batch_size), desc=f"  {ds_name}", leave=False):
            batch = [str(t) for t in texts[i:i + batch_size]]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            attention_mask = inputs['attention_mask']
            seq_lengths = attention_mask.sum(dim=1) - 1
            for layer_idx in layers:
                hs = outputs.hidden_states[layer_idx + 1]
                emb = hs[torch.arange(hs.size(0)), seq_lengths]
                embeddings[layer_idx].append(emb.cpu())

        for layer_idx in layers:
            embeddings[layer_idx] = torch.cat(embeddings[layer_idx], dim=0)

        cache_path = CACHE_DIR / f"mistral_defended_{ds_name}_embeddings.pt"
        torch.save(embeddings, cache_path)
        print(f"  Saved {cache_path}")

    del model, base_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_cka_matrix(model_a_name, model_b_name, dataset, n_samples=520):
    """Compute full cross-layer CKA matrix between two models."""
    emb_a = torch.load(CACHE_DIR / f"{model_a_name}_{dataset}_embeddings.pt", map_location='cpu')
    emb_b = torch.load(CACHE_DIR / f"{model_b_name}_{dataset}_embeddings.pt", map_location='cpu')

    n = min(n_samples, emb_a[0].shape[0], emb_b[0].shape[0])

    gram_a, hsic_a = precompute_grams(emb_a, n)
    del emb_a
    gram_b, hsic_b = precompute_grams(emb_b, n)
    del emb_b
    gc.collect()

    layers_a = sorted(gram_a.keys())
    layers_b = sorted(gram_b.keys())

    cka_matrix = np.zeros((len(layers_a), len(layers_b)), dtype=np.float64)

    for i, la in enumerate(layers_a):
        Ka = gram_a[la]
        ha = hsic_a[la]
        for j, lb in enumerate(layers_b):
            Kb = gram_b[lb]
            hb = hsic_b[lb]
            hsic_ab = float(np.sum(Ka * Kb))
            denom = np.sqrt(ha * hb) + 1e-10
            cka_matrix[i, j] = np.clip(hsic_ab / denom, 0.0, 1.0)

    return cka_matrix, layers_a, layers_b


def plot_heatmap(cka_matrix, layers_a, layers_b, model_a, model_b, dataset, save_path, title=None):
    """Generate cross-layer CKA heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        cka_matrix, origin='lower', cmap='magma',
        vmin=0, vmax=1, aspect='auto',
    )
    fig.colorbar(im, ax=ax, label='CKA')

    step_a = max(1, len(layers_a) // 8)
    step_b = max(1, len(layers_b) // 8)
    ax.set_xticks(range(0, len(layers_b), step_b))
    ax.set_xticklabels([str(layers_b[i]) for i in range(0, len(layers_b), step_b)], fontsize=7)
    ax.set_yticks(range(0, len(layers_a), step_a))
    ax.set_yticklabels([str(layers_a[i]) for i in range(0, len(layers_a), step_a)], fontsize=7)

    ax.set_xlabel(f'{model_b} layer')
    ax.set_ylabel(f'{model_a} layer')

    if title is None:
        ds_label = 'WikiText (clean)' if dataset == 'clean' else 'AdvBench (harmful)'
        title = f'Cross-Layer CKA: {model_a} vs {model_b}\n{ds_label}'
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_side_by_side(before_mat, after_mat, layers_a, layers_b, dataset, save_path):
    """Side-by-side before/after heatmaps."""
    ds_label = 'AdvBench (harmful)' if dataset == 'harm' else 'WikiText (clean)'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for ax, mat, label in [(ax1, before_mat, 'Before Defense (Base Mistral)'),
                            (ax2, after_mat, 'After Defense (Defended Mistral)')]:
        im = ax.imshow(mat, origin='lower', cmap='magma', vmin=0, vmax=1, aspect='auto')

        step_a = max(1, len(layers_a) // 8)
        step_b = max(1, len(layers_b) // 8)
        ax.set_xticks(range(0, len(layers_b), step_b))
        ax.set_xticklabels([str(layers_b[i]) for i in range(0, len(layers_b), step_b)], fontsize=7)
        ax.set_yticks(range(0, len(layers_a), step_a))
        ax.set_yticklabels([str(layers_a[i]) for i in range(0, len(layers_a), step_a)], fontsize=7)

        ax.set_xlabel('Mistral layer')
        ax.set_ylabel('Llama-2 layer')
        ax.set_title(f'{label}\n{ds_label}')

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='CKA')

    # Add mean diagonal annotations
    before_diag = compute_interpolated_diagonal(before_mat, layers_a, layers_b)
    after_diag = compute_interpolated_diagonal(after_mat, layers_a, layers_b)
    ax1.text(0.02, 0.98, f'Mean diag: {before_diag:.3f}',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             va='top', color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax2.text(0.02, 0.98, f'Mean diag: {after_diag:.3f}',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             va='top', color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_three_panel(before_mat, after_mat, layers_a, layers_b, dataset, save_path):
    """Before, After, and Difference (Before - After) heatmaps side by side."""
    ds_label = 'AdvBench (harmful)' if dataset == 'harm' else 'WikiText (clean)'
    diff_mat = before_mat - after_mat

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 8))

    # Before
    im1 = ax1.imshow(before_mat, origin='lower', cmap='magma', vmin=0, vmax=1, aspect='auto')
    ax1.set_title(f'Before Defense (Base Mistral)\n{ds_label}', fontsize=13)

    # After
    im2 = ax2.imshow(after_mat, origin='lower', cmap='magma', vmin=0, vmax=1, aspect='auto')
    ax2.set_title(f'After Defense (Defended Mistral)\n{ds_label}', fontsize=13)

    # Difference (before - after): positive = similarity was reduced
    vmax_diff = max(abs(diff_mat.min()), abs(diff_mat.max()), 0.3)
    im3 = ax3.imshow(diff_mat, origin='lower', cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
    ax3.set_title(f'CKA Reduction (Before $-$ After)\n{ds_label}', fontsize=13)

    # Axis labels and ticks
    step_a = max(1, len(layers_a) // 8)
    step_b = max(1, len(layers_b) // 8)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(range(0, len(layers_b), step_b))
        ax.set_xticklabels([str(layers_b[i]) for i in range(0, len(layers_b), step_b)], fontsize=7)
        ax.set_yticks(range(0, len(layers_a), step_a))
        ax.set_yticklabels([str(layers_a[i]) for i in range(0, len(layers_a), step_a)], fontsize=7)
        ax.set_xlabel('Mistral layer')
        ax.set_ylabel('Llama-2 layer')

    # Colorbars
    fig.colorbar(im1, ax=ax1, label='CKA', fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, label='CKA', fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, label='$\Delta$CKA', fraction=0.046, pad=0.04)

    # Annotations
    before_diag = compute_interpolated_diagonal(before_mat, layers_a, layers_b)
    after_diag = compute_interpolated_diagonal(after_mat, layers_a, layers_b)
    diff_diag = before_diag - after_diag
    bbox_style = dict(boxstyle='round', facecolor='black', alpha=0.7)
    ax1.text(0.02, 0.98, f'Mean diag: {before_diag:.3f}',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             va='top', color='white', bbox=bbox_style)
    ax2.text(0.02, 0.98, f'Mean diag: {after_diag:.3f}',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             va='top', color='white', bbox=bbox_style)
    ax3.text(0.02, 0.98, f'Mean diag $\Delta$: {diff_diag:+.3f}\nMax $\Delta$: {diff_mat.max():+.3f}',
             transform=ax3.transAxes, fontsize=11, fontweight='bold',
             va='top', color='white', bbox=bbox_style)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract defended model embeddings
    print("=" * 60)
    print("Step 1: Extracting defended Mistral embeddings")
    print("=" * 60)
    extract_defended_embeddings()

    # Step 2: Compute CKA matrices
    for ds in ['harm', 'clean']:
        print(f"\n{'=' * 60}")
        print(f"Step 2: Computing CKA ({ds})")
        print(f"{'=' * 60}")

        # Before defense: llama2 vs base mistral (load from phase1 cache)
        csv_before = Path("../phase1/all_layers_cka_output/cross_layer_matrices") / f"cka_{ds}_llama2_vs_mistral.csv"
        if csv_before.exists():
            before_df = pd.read_csv(csv_before, index_col=0)
            before_mat = before_df.values
            layers_a_before = [int(c) for c in before_df.index]
            layers_b_before = [int(c) for c in before_df.columns]
            print(f"Loaded before-defense CKA from {csv_before}")
        else:
            print(f"Computing before-defense CKA...")
            before_mat, layers_a_before, layers_b_before = compute_cka_matrix('llama2', 'mistral', ds)

        # After defense: llama2 vs defended mistral
        print(f"Computing after-defense CKA...")
        after_mat, layers_a_after, layers_b_after = compute_cka_matrix('llama2', 'mistral_defended', ds)

        # Save after-defense CSV
        after_df = pd.DataFrame(
            after_mat,
            index=[str(l) for l in layers_a_after],
            columns=[str(l) for l in layers_b_after],
        )
        after_df.to_csv(OUTPUT_DIR / f"cka_{ds}_llama2_vs_mistral_defended.csv")

        # Individual heatmaps
        plot_heatmap(
            after_mat, layers_a_after, layers_b_after,
            'llama2', 'mistral_defended', ds,
            OUTPUT_DIR / f"cka_{ds}_llama2_vs_mistral_defended.png",
            title=f'Cross-Layer CKA: llama2 vs defended mistral\n{"AdvBench (harmful)" if ds == "harm" else "WikiText (clean)"}'
        )

        # Side-by-side comparison
        plot_side_by_side(
            before_mat, after_mat,
            layers_a_before, layers_b_before,
            ds,
            OUTPUT_DIR / f"cka_{ds}_before_after_defense.png"
        )

        # Three-panel: before, after, difference
        plot_three_panel(
            before_mat, after_mat,
            layers_a_before, layers_b_before,
            ds,
            OUTPUT_DIR / f"cka_{ds}_three_panel.png"
        )

        # Print diagonal summary
        before_diag = compute_interpolated_diagonal(before_mat, layers_a_before, layers_b_before)
        after_diag = compute_interpolated_diagonal(after_mat, layers_a_after, layers_b_after)
        print(f"\n  {ds.upper()} CKA mean diagonal:")
        print(f"    Before defense: {before_diag:.4f}")
        print(f"    After defense:  {after_diag:.4f}")
        print(f"    Delta:          {after_diag - before_diag:+.4f}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
