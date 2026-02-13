#!/usr/bin/env python3
"""
CKA vs Transfer ASR Scatter Plot (Phase 2)

Loads per-layer CKA similarity matrices (from Phase 1) and transfer attack
results, then produces a scatter plot of geometric similarity vs symmetric
transfer ASR for a chosen layer. Points are colored by same-family vs
cross-family pairs, and outliers are annotated.

Usage:
    python generate_examples_new.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
from itertools import combinations

# --- CONFIG ---
CKA_DIR = "../phase1/alignment_results_cka"
CKA_METRIC = "malicious"
TRANSFER_FILE = "final_transfer_results.csv"
SUCCESS_THRESHOLD = 2.0
TARGET_LAYER = 30

# Model family groupings for coloring
FAMILIES = {
    "Llama": ["Llama", "Vicuna", "Guanaco"],
    "Mistral": ["Mistral", "Zephyr", "Hermes", "NeuralChat", "Starling", "OpenChat", "Solar"],
    "Yi": ["Yi"],
    "Qwen": ["Qwen"],
    "Falcon": ["Falcon"],
    "Gemma": ["Gemma"],
    "Phi": ["Phi"],
    "DeepSeek": ["DeepSeek"]
}

print(f"[*] Loading Transfer Results...")
df = pd.read_csv(TRANSFER_FILE)

# Calculate Relaxed ASR
df['is_transfer'] = df['transfer_loss'] < SUCCESS_THRESHOLD
pair_asr = df.groupby(['source_model', 'target_model'])['is_transfer'].mean() * 100

models = df['source_model'].unique()
points_x = []
points_y = []
labels = []
pair_names = []  # Store names for labeling

print(f"[*] Extracting Layer {TARGET_LAYER} Data...")

for name_a, name_b in combinations(models, 2):
    # 1. Get CKA for specific layer
    path1 = f"{CKA_DIR}/{name_a}_{name_b}_{CKA_METRIC}.csv"
    path2 = f"{CKA_DIR}/{name_b}_{name_a}_{CKA_METRIC}.csv"
    final_path = path1 if os.path.exists(path1) else (path2 if os.path.exists(path2) else None)

    if not final_path: continue

    try:
        matrix = np.loadtxt(final_path, delimiter=",")
        # Ensure matrix is big enough
        if matrix.shape[0] <= TARGET_LAYER or matrix.shape[1] <= TARGET_LAYER:
            continue

        sim_score = matrix[TARGET_LAYER, TARGET_LAYER]
    except (IOError, ValueError, IndexError):
        continue

    # 2. Get Symmetric ASR
    try:
        asr_ab = pair_asr.get((name_a, name_b), np.nan)
        asr_ba = pair_asr.get((name_b, name_a), np.nan)

        if np.isnan(asr_ab) and np.isnan(asr_ba): continue
        if np.isnan(asr_ab):
            sym_asr = asr_ba
        elif np.isnan(asr_ba):
            sym_asr = asr_ab
        else:
            sym_asr = (asr_ab + asr_ba) / 2.0
    except (KeyError, TypeError):
        continue

    points_x.append(sim_score)
    points_y.append(sym_asr)
    pair_names.append(f"{name_a}\n{name_b}")  # Store name for labeling

    # Family Logic
    fam_a = next((f for f, mems in FAMILIES.items() if any(m.lower() in name_a.lower() for m in mems)), "Other")
    fam_b = next((f for f, mems in FAMILIES.items() if any(m.lower() in name_b.lower() for m in mems)), "Other")
    is_same = (fam_a == fam_b and fam_a != "Other")
    labels.append("Same Family" if is_same else "Different Families")

# --- 3. Plotting with Labels ---
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")
sns.set_context("talk")

df_plot = pd.DataFrame({
    "Similarity": points_x,
    "ASR": points_y,
    "Type": labels,
    "Pair": pair_names
})

ax = sns.scatterplot(
    data=df_plot, x="Similarity", y="ASR", hue="Type", style="Type",
    palette={"Same Family": "#1f77b4", "Different Families": "#d62728"},
    s=150, alpha=0.8
)

sns.regplot(
    data=df_plot, x="Similarity", y="ASR", scatter=False,
    color="black", line_kws={"linestyle": "--", "alpha": 0.5}, ax=ax
)

# --- DETECTIVE MODE: Label the outliers ---
# We label points that are far from the trend line to identify "Problem Pairs"
x_mean = np.mean(points_x)
y_mean = np.mean(points_y)

for i in range(len(df_plot)):
    x = df_plot.Similarity[i]
    y = df_plot.ASR[i]
    label = df_plot.Pair[i]

    # Logic: Label points that are "weird"
    # Case 1: High Sim, Low Transfer (Bottom Right) -> The "Failures"
    if x > 0.8 and y < 60:
        plt.text(x, y - 2, label, fontsize=8, color='black', ha='center', va='top')

    # Case 2: Low Sim, High Transfer (Top Left) -> The "Magic"
    # (These are ruining your correlation the most)
    if x < 0.6 and y > 80:
        plt.text(x, y + 1, label, fontsize=8, color='darkred', ha='center', va='bottom')

corr, _ = pearsonr(points_x, points_y)
plt.title(f"Layer {TARGET_LAYER} Malicious CKA vs. Relaxed Transfer\nPearson r = {corr:.2f}")
plt.xlabel(f"Geometric Similarity (Malicious CKA @ Layer {TARGET_LAYER})")
plt.ylabel(f"Symmetric Transfer ASR (%)")

plt.tight_layout()
plt.savefig(f"{CKA_DIR}/layer_{TARGET_LAYER}_detailed_plot.png")
print(f"[*] Plot saved to {CKA_DIR}/layer_{TARGET_LAYER}_detailed_plot.png")
plt.show()