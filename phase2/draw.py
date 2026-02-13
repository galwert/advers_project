#!/usr/bin/env python3
"""
Transfer Matrix Heatmap Visualization (Phase 2)

Reads the scored transfer results from llm_judge.py and produces a
multi-panel figure with:
  1. 20x20 ASR heatmap (source -> target)
  2. Sample count matrix
  3. Self-attack vs transfer-attack distribution box plot
  4. Model vulnerability bar chart rankings

Also prints summary statistics (most vulnerable, most robust, best sources).

Usage:
    python draw.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "phase2_scored_detailed_new.csv"

print("=" * 80)
print("TRANSFER MATRIX VISUALIZATION")
print("=" * 80)

# Load data
df = pd.read_csv(INPUT_CSV)
print(f"\n[+] Loaded {len(df)} rows")

# Get unique models (sorted alphabetically)
models = sorted(list(set(df['source_model'].unique()) | set(df['target_model'].unique())))
n_models = len(models)

print(f"[+] Found {n_models} models")

# Create ASR matrix: rows = source, columns = target
asr_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
count_matrix = pd.DataFrame(index=models, columns=models, dtype=int)

print("\n[*] Computing transfer matrix...")

for source in models:
    for target in models:
        subset = df[(df['source_model'] == source) & (df['target_model'] == target)]

        total = len(subset)
        success = len(subset[subset['is_jailbroken'] == True])

        count_matrix.loc[source, target] = total

        if total > 0:
            asr_matrix.loc[source, target] = (success / total) * 100
        else:
            asr_matrix.loc[source, target] = np.nan

# Convert to numeric
asr_matrix = asr_matrix.astype(float)
count_matrix = count_matrix.astype(int)

print("[+] Matrix computed")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, height_ratios=[10, 10, 3])

# ==========================================
# 1. MAIN HEATMAP: ASR Matrix
# ==========================================
ax1 = fig.add_subplot(gs[0, :])

# Create mask for missing data
mask = asr_matrix.isna()

# Custom colormap: blue (low) -> yellow -> red (high)
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Plot heatmap
sns.heatmap(
    asr_matrix,
    annot=True,
    fmt='.1f',
    cmap=cmap,
    center=asr_matrix.mean().mean(),  # Center on overall mean
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Attack Success Rate (%)'},
    linewidths=0.5,
    linecolor='gray',
    mask=mask,
    ax=ax1,
    square=True,
    annot_kws={'fontsize': 8}
)

ax1.set_title('GCG Attack Transfer Matrix (Source → Target ASR %)',
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Target Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Source Model', fontsize=12, fontweight='bold')

# Highlight diagonal (self-attacks)
for i in range(n_models):
    ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=3))

# Rotate labels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=9)

from matplotlib.patches import Rectangle
legend_elements = [Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', lw=3, label='Self-Attack (diagonal)')]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# ==========================================
# 2. SAMPLE COUNT MATRIX
# ==========================================
ax2 = fig.add_subplot(gs[1, 0])

sns.heatmap(
    count_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar_kws={'label': 'Number of Examples'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax2,
    square=True,
    annot_kws={'fontsize': 7}
)

ax2.set_title('Sample Count Matrix', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Target Model', fontsize=11)
ax2.set_ylabel('Source Model', fontsize=11)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

# ==========================================
# 3. TRANSFER vs SELF ASR COMPARISON
# ==========================================
ax3 = fig.add_subplot(gs[1, 1])

# Extract diagonal (self) and off-diagonal (transfer) ASRs
self_asrs = []
transfer_asrs = []

for i, model in enumerate(models):
    # Self ASR
    self_asr = asr_matrix.loc[model, model]
    if not np.isnan(self_asr):
        self_asrs.append(self_asr)

    # Average transfer ASR (as target)
    transfer_vals = []
    for j, source in enumerate(models):
        if i != j:  # Off-diagonal
            val = asr_matrix.loc[source, model]
            if not np.isnan(val):
                transfer_vals.append(val)

    if transfer_vals:
        transfer_asrs.append(np.mean(transfer_vals))

# Box plot
data_to_plot = [self_asrs, transfer_asrs]
bp = ax3.boxplot(data_to_plot, labels=['Self-Attack', 'Transfer-Attack'],
                 patch_artist=True, showmeans=True)

# Color boxes
colors = ['#ff6b6b', '#4ecdc4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel('Attack Success Rate (%)', fontsize=11)
ax3.set_title('Self vs Transfer Attack Distribution', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add statistics
if self_asrs:
    ax3.text(1, max(self_asrs) + 5, f'Mean: {np.mean(self_asrs):.1f}%\nMedian: {np.median(self_asrs):.1f}%',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
if transfer_asrs:
    ax3.text(2, max(transfer_asrs) + 5, f'Mean: {np.mean(transfer_asrs):.1f}%\nMedian: {np.median(transfer_asrs):.1f}%',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ==========================================
# 4. MODEL RANKINGS (Bottom row)
# ==========================================
ax4 = fig.add_subplot(gs[2, :])

# Compute rankings
model_stats = []
for model in models:
    # Self ASR
    self_asr = asr_matrix.loc[model, model] if not np.isnan(asr_matrix.loc[model, model]) else 0

    # Average as target (vulnerability)
    as_target = asr_matrix[model].dropna()
    target_asr = as_target.mean() if len(as_target) > 0 else 0

    # Average as source (attack effectiveness)
    as_source = asr_matrix.loc[model].dropna()
    source_asr = as_source.mean() if len(as_source) > 0 else 0

    # Transfer in (excluding self)
    transfer_in = asr_matrix[model].drop(model, errors='ignore').dropna()
    transfer_in_asr = transfer_in.mean() if len(transfer_in) > 0 else 0

    model_stats.append({
        'model': model,
        'self': self_asr,
        'target': target_asr,
        'source': source_asr,
        'transfer_in': transfer_in_asr
    })

df_stats = pd.DataFrame(model_stats)

# Sort by vulnerability (target ASR)
df_stats_sorted = df_stats.sort_values('target', ascending=False)

# Bar plot
x = np.arange(len(models))
width = 0.25

bars1 = ax4.bar(x - width * 1.5, df_stats_sorted['self'], width, label='Self ASR', color='#ff6b6b', alpha=0.8)
bars2 = ax4.bar(x - width * 0.5, df_stats_sorted['transfer_in'], width, label='Transfer-In ASR', color='#4ecdc4',
                alpha=0.8)
bars3 = ax4.bar(x + width * 0.5, df_stats_sorted['target'], width, label='Overall Target ASR', color='#95a5a6',
                alpha=0.8)

ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
ax4.set_ylabel('ASR (%)', fontsize=11)
ax4.set_title('Model Vulnerability Rankings (Sorted by Overall Target ASR)', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(df_stats_sorted['model'], rotation=45, ha='right', fontsize=9)
ax4.legend(fontsize=10, loc='upper right')
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, 100)

# ==========================================
# Save figure
# ==========================================
plt.suptitle(f'GCG Attack Transfer Analysis ({len(df)} examples, {n_models} models)',
             fontsize=18, fontweight='bold', y=0.995)

output_file = 'transfer_matrix_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[+] Saved visualization to: {output_file}")

# ==========================================
# Print statistics
# ==========================================
print("\n" + "=" * 80)
print("TRANSFER MATRIX STATISTICS")
print("=" * 80)

# Overall statistics
all_self = [asr_matrix.loc[m, m] for m in models if not np.isnan(asr_matrix.loc[m, m])]
all_transfer = []
for i, source in enumerate(models):
    for j, target in enumerate(models):
        if i != j:
            val = asr_matrix.loc[source, target]
            if not np.isnan(val):
                all_transfer.append(val)

print(f"\nSelf-Attack ASR:")
print(f"  Mean:   {np.mean(all_self):.2f}%")
print(f"  Median: {np.median(all_self):.2f}%")
print(f"  Min:    {np.min(all_self):.2f}%")
print(f"  Max:    {np.max(all_self):.2f}%")
print(f"  Std:    {np.std(all_self):.2f}%")

print(f"\nTransfer-Attack ASR:")
print(f"  Mean:   {np.mean(all_transfer):.2f}%")
print(f"  Median: {np.median(all_transfer):.2f}%")
print(f"  Min:    {np.min(all_transfer):.2f}%")
print(f"  Max:    {np.max(all_transfer):.2f}%")
print(f"  Std:    {np.std(all_transfer):.2f}%")

print(f"\nTransfer Effectiveness:")
print(f"  Self ASR - Transfer ASR: {np.mean(all_self) - np.mean(all_transfer):.2f}%")
if np.mean(all_self) > np.mean(all_transfer):
    print(f"  → Self attacks are MORE effective (attacks don't transfer well)")
else:
    print(f"  → Transfer attacks are MORE effective (attacks generalize well!)")

# Most vulnerable models
print(f"\n{'=' * 80}")
print("TOP 5 MOST VULNERABLE MODELS (Target ASR)")
print(f"{'=' * 80}")
top_vulnerable = df_stats.sort_values('target', ascending=False).head(5)
for _, row in top_vulnerable.iterrows():
    print(
        f"  {row['model']:<20} {row['target']:.2f}%  (Self: {row['self']:.2f}%, Transfer-In: {row['transfer_in']:.2f}%)")

# Most robust models
print(f"\n{'=' * 80}")
print("TOP 5 MOST ROBUST MODELS (Lowest Target ASR)")
print(f"{'=' * 80}")
top_robust = df_stats.sort_values('target', ascending=True).head(5)
for _, row in top_robust.iterrows():
    print(
        f"  {row['model']:<20} {row['target']:.2f}%  (Self: {row['self']:.2f}%, Transfer-In: {row['transfer_in']:.2f}%)")

# Best attack sources
print(f"\n{'=' * 80}")
print("TOP 5 BEST ATTACK SOURCES (Source ASR)")
print(f"{'=' * 80}")
top_sources = df_stats.sort_values('source', ascending=False).head(5)
for _, row in top_sources.iterrows():
    print(f"  {row['model']:<20} {row['source']:.2f}%")

# Best transfer attacks (highest off-diagonal ASR)
print(f"\n{'=' * 80}")
print("TOP 10 BEST TRANSFER ATTACKS (Source → Target)")
print(f"{'=' * 80}")

transfer_pairs = []
for i, source in enumerate(models):
    for j, target in enumerate(models):
        if i != j:  # Off-diagonal
            asr = asr_matrix.loc[source, target]
            if not np.isnan(asr):
                transfer_pairs.append({
                    'source': source,
                    'target': target,
                    'asr': asr,
                    'count': count_matrix.loc[source, target]
                })

df_transfers = pd.DataFrame(transfer_pairs).sort_values('asr', ascending=False)

print(f"\n{'Source':<20} {'Target':<20} {'ASR':<10} {'Count':<10}")
print("-" * 80)
for _, row in df_transfers.head(10).iterrows():
    print(f"{row['source']:<20} {row['target']:<20} {row['asr']:<10.2f} {row['count']:<10}")

# Save matrices to CSV
asr_matrix.to_csv('transfer_matrix_asr.csv')
count_matrix.to_csv('transfer_matrix_counts.csv')
print(f"\n[+] Saved matrices to:")
print(f"    - transfer_matrix_asr.csv")
print(f"    - transfer_matrix_counts.csv")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

plt.show()