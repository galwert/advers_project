#!/usr/bin/env python3
"""Regenerate Figure 1 (left): RSA heatmap with three family boxes.

Run from the anchor-rep repo root:
    python scripts/regen_heatmap.py
Output: ./figs/heatmap_rsa_harm_clustered.{png,pdf}
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
rsa_df = pd.read_json(_REPO_ROOT / 'data' / 'similarity_matrices' / 'rsa_harm_pct50.json',
                      orient='split')

# Short display names (Rom: "make the text bigger and give them shortcuts")
DISPLAY = {
    'gemma': 'Gemma-7B', 'deepseek': 'DeepSeek-7B', 'solar': 'Solar-10.7B',
    'mistral': 'Mistral-7B', 'zephyr': 'Zephyr-7B', 'starling': 'Starling-7B',
    'openchat': 'OpenChat-3.5', 'hermes2': 'Hermes-2', 'neuralchat': 'NeuralChat-7B',
    'internlm2': 'InternLM2-7B', 'orca2': 'Orca-2-7B',
    'llama2': 'Llama-2-7B', 'vicuna': 'Vicuna-7B', 'llama3': 'Llama-3-8B',
    'yi': 'Yi-6B', 'baichuan2': 'Baichuan2-7B', 'qwen': 'Qwen-7B',
    'stablelm': 'StableLM-3B', 'falcon': 'Falcon-7B', 'phi2': 'Phi-2',
}

# Model order: Mistral block + InternLM2 right after, then Llama+Chinese+others
ordered = [
    # Mistral extended block
    'gemma', 'deepseek', 'solar', 'mistral', 'zephyr', 'starling', 'openchat', 'hermes2', 'neuralchat',
    # InternLM right after
    'internlm2',
    # Llama + Chinese + others
    'orca2', 'llama2', 'vicuna', 'yi', 'baichuan2', 'llama3', 'stablelm', 'falcon', 'phi2', 'qwen',
]
ordered = [m for m in ordered if m in rsa_df.index]

# Family boxes (smaller families)
family_boxes = [
    (3, 9),    # Mistral core: mistral, zephyr, starling, openchat, hermes2, neuralchat
    (11, 13),  # Llama: llama2, vicuna
    (13, 15),  # Chinese-1: yi, baichuan2
    (15, 16),  # Llama-3 alone (bridges both)
]
# Order so each architecturally-coherent family occupies a contiguous block.
# Three families: Mistral (7 with solar), Llama (4 with orca-2), Eastern (qwen+yi).
# Other models (deepseek, gemma, phi-2, internlm2, baichuan2, falcon, stablelm)
# do not fit a family by representational similarity (see app:family_inclusion);
# they fill the unframed regions of the heatmap.
ordered = [
    # Gemma first: highest similarity (RSA 0.41-0.43) with the Mistral cluster,
    # so placing it before Mistral lets the heatmap flow visually.
    'gemma',
    # Mistral family (7 contiguous)
    'mistral', 'zephyr', 'starling', 'openchat', 'hermes2', 'neuralchat', 'solar',
    # Unfamilied bridge
    'deepseek', 'internlm2',
    # Llama family (4 contiguous, includes Orca-2 as a Llama-2 derivative)
    'llama2', 'vicuna', 'llama3', 'orca2',
    # Eastern family (Qwen + Yi only)
    'yi', 'qwen',
    # Unfamilied tail
    'baichuan2', 'phi2', 'stablelm', 'falcon',
]
ordered = [m for m in ordered if m in rsa_df.index]

family_boxes = [
    (1, 8),    # Mistral family (7 models, contiguous)
    (10, 14),  # Llama family (4 with orca-2, contiguous)
    (14, 16),  # Eastern (Qwen + Yi only)
]

# Reorder matrix
mat = rsa_df.loc[ordered, ordered]
labels = [DISPLAY.get(m, m) for m in ordered]

# Plot using seaborn heatmap to match the original style
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(mat.values, ax=ax,
            xticklabels=labels, yticklabels=labels,
            cmap='Reds', vmin=0.0, vmax=1.0,
            annot=False,
            linewidths=0.3, linecolor='white',
            cbar_kws={'shrink': 0.8})

ax.tick_params(axis='x', rotation=90, labelsize=18)
ax.tick_params(axis='y', rotation=0, labelsize=18)

# Draw family boxes with distinct colors and labels
box_styles = [
    ((1, 8),   '#0000CC', 'Mistral family'),     # blue
    ((10, 14), '#CC0000', 'Llama family'),        # red
    ((14, 16), '#008800', 'Eastern family'),      # green (Qwen + Yi)
]
for (start, end), color, label in box_styles:
    n = end - start
    rect = Rectangle((start, start), n, n,
                      linewidth=4.5, edgecolor=color, facecolor='none',
                      clip_on=False, zorder=5, linestyle='-')
    ax.add_patch(rect)
    mid = start + n / 2
    # Label positioning: Mistral and Llama go above the box; Eastern goes below
    # so it doesn't collide with the Llama label that sits just above its box.
    if 'Eastern' in label:
        ax.text(mid, end + 0.5, label, ha='center', va='top',
                fontsize=14, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    else:
        ax.text(mid, start - 0.3, label, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

plt.tight_layout()
out = _REPO_ROOT / 'figs'
out.mkdir(exist_ok=True)
plt.savefig(out / 'heatmap_rsa_harm_clustered.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.savefig(out / 'heatmap_rsa_harm_clustered.pdf', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved to {out}")
