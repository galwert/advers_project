#!/usr/bin/env python3
"""bar_final using the paper's Table 11 ASR matrix (WildGuard pipeline judge,
19 aligned LLMs, no manual verification). This is what the appendix labels
tab:asr_matrix and what fig:bar_transfer claims to summarize.

Hardcoded from phase3/overleaf/sections/appendix.tex tab:asr_matrix so the
figure does not drift if any intermediate CSV gets re-judged.

Two output paths controlled by EXCLUDE_HERMES at the top:
  - EXCLUDE_HERMES = False  → bar_final.{png,pdf}
  - EXCLUDE_HERMES = True   → bar_final_no_hermes.{png,pdf}
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ---- Paper Table 11: ASR (%) matrix, rows=source, cols=target ----
# 19 models; '--' means cell unfilled in paper (no run available); diagonal
# is self-attack (kept here for completeness; excluded in pair scan below).
COLS = ['mistral','zephyr','starling','hermes2','openchat','neuralchat','solar',
        'gemma','deepseek','llama2','vicuna','llama3','yi','baichuan2',
        'orca','falcon','phi2','qwen','stablelm']

# rows in same order as COLS
PAPER_ASR = {
 'mistral':    [69, 56, 58, 12, 56, 72, 28,  4, 30,  1, 17,  1, 18,  1, 24, 14, 0, 12, 42],
 'zephyr':     [34, 65, 45, 11, 42, 70, 20,  4,  6,  0,  9,  0, 14,  5, 32, 10, 0,  7, 25],
 'starling':   [38, 57, 75, 11, 73, 76, 22,  1, 11,  1, 13,  1, 16, 10, 24, 19, 0,  3, 22],
 'hermes2':    [34, 58,'--', 9, 46, 75, 17,  2,  6,  0, 13,  0, 17,  6, 28, 16, 0,'--',30],
 'openchat':   [32, 57,'--',13, 77, 70, 23,  0,  9,  0, 10,  0, 15,  5, 17, 22, 0,'--',25],
 'neuralchat': [25, 56,'--',13, 36, 81, 18,  4, 17,  0, 25,  0, 27,  4, 21, 14, 0,'--',36],
 'solar':      [31, 55,'--',18, 43, 74, 42,  1, 11,  0, 16,  0, 16,  3, 26, 20, 0,'--',26],
 'gemma':      [24, 47,'--',14, 40, 80, 14, 17,  7,  0, 11,  0,  8,  2, 35, 16, 0,'--',29],
 'deepseek':   [30, 61,'--',15, 45, 82, 18,  2, 39,  0, 17,  0, 14,  2, 27, 16, 0,'--',40],
 'llama2':     [34, 67, 53, 14, 57, 72, 21,  1, 17, 23,  8,  1, 16,  8, 20, 13, 0,  8, 27],
 'vicuna':     [20, 53, 39, 19, 38, 82, 13,  2,  7,  1, 17,  0, 10,  1, 29, 23, 0,  2, 22],
 'llama3':     [27, 50, 35, 10, 40, 73, 17,  1, 10,  1, 14,  1, 10,  2, 26,  9, 0,  4, 27],
 'yi':         [25, 61, 49,  9, 45, 75, 16,  1, 12,  0, 12,  0, 67,  0, 25, 14, 0,  5, 26],
 'baichuan2':  [26, 55,'--',17, 42, 73, 10,  0,  9,  0,  9,  0, 12,  0, 24, 23, 0,'--',34],
 'orca':       [25, 57, 34, 16, 40, 81, 14,  0,  5,  0,  9,  0,  7,  0, 35, 20, 0,  0, 25],
 'falcon':     [32, 57,'--',14, 35, 66, 15,  4, 11,  0, 10,  1, 13,  5, 26, 39, 0,'--',33],
 'phi2':       [19, 48, 35, 13, 28, 60, 16,  2, 10,  0, 16,  0,  9,  8, 25,  9, 0,  4, 26],
 'qwen':       [36, 67, 45,  8, 49, 69, 25,  4, 23,  0, 25,  1, 28,  4, 21, 18, 0, 54, 42],
 'stablelm':   [29, 47,'--', 9, 40, 70, 17,  1,  9,  0, 10,  0,  8,  1, 28,  9, 0,'--',50],
}

def asr(src, tgt):
    if src not in PAPER_ASR: return None
    if tgt not in COLS:      return None
    v = PAPER_ASR[src][COLS.index(tgt)]
    return None if v == '--' else float(v)

# Scenario "D" family grouping: architecturally-coherent families with high
# pairwise similarity. Gemma + Phi-2 + DeepSeek dropped (representational
# outliers: low pairwise similarity with their nominal-family peers). Orca-2
# moved to Llama family (it is a Llama-2 derivative). See app:family_inclusion
# in the appendix for full rationale.
FAMILIES = {
    'Mistral': ['mistral','zephyr','hermes2','starling','openchat','neuralchat','solar'],
    'Llama':   ['llama2','llama3','vicuna','orca'],
    'Eastern': ['qwen','yi'],
}
def fam(m):
    for f, ms in FAMILIES.items():
        if m in ms: return f
    return 'Other'

# ---- CKA from cross-layer matrices ----
# Name mapping: ASR data uses 'orca' but CKA matrix files use 'orca2'. Apply
# this mapping when looking up similarity matrices.
SIM_NAME = {'orca': 'orca2'}
def simname(m): return SIM_NAME.get(m, m)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CKA_DIR = _REPO_ROOT / 'data' / 'cross_layer_cka'
mats = {}
for p in sorted(_CKA_DIR.glob('cka_harm_*_vs_*.json')):
    parts = p.stem.replace('cka_harm_','').split('_vs_')
    if len(parts) == 2:
        mats[(parts[0], parts[1])] = pd.read_json(p, orient='split').values

def cka_diag_mean(s, t):
    s2, t2 = simname(s), simname(t)
    if (s2,t2) in mats:   M = mats[(s2,t2)]
    elif (t2,s2) in mats: M = mats[(t2,s2)]
    else: return None
    La, Lb = M.shape
    n = min(La, Lb)
    out = []
    for k in range(n):
        r = k / max(n-1, 1)
        out.append(M[min(int(r*(La-1)), La-1), min(int(r*(Lb-1)), Lb-1)])
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-hermes', action='store_true')
    ap.add_argument('--out-base', default=None,
                    help='Output filename base under imgs/. Defaults: bar_final or bar_final_no_hermes.')
    args = ap.parse_args()

    EXCLUDE = {'hermes2'} if args.exclude_hermes else set()
    out_base = args.out_base or ('bar_final_no_hermes' if args.exclude_hermes else 'bar_final')

    pairs = []
    for s in COLS:
        for t in COLS:
            if s == t: continue
            if fam(s) != fam(t) or fam(s) == 'Other': continue
            if s in EXCLUDE or t in EXCLUDE: continue
            a = asr(s, t)
            c = cka_diag_mean(s, t)
            if a is None or c is None: continue
            pairs.append({'src':s, 'tgt':t, 'asr':a, 'cka':c})
    df = pd.DataFrame(pairs)
    print(f"Same-family pairs: {len(df)}  (excluding hermes2: {args.exclude_hermes})")

    rho, pval = stats.spearmanr(df['cka'], df['asr'])
    print(f"Spearman rho={rho:.3f}, p={pval:.4f}")

    t1 = df['cka'].quantile(1/3)
    t2 = df['cka'].quantile(2/3)
    low = df[df['cka'] <= t1]
    mid = df[(df['cka'] > t1) & (df['cka'] <= t2)]
    hi  = df[df['cka'] > t2]
    means  = [low['asr'].mean(), mid['asr'].mean(), hi['asr'].mean()]
    counts = [len(low), len(mid), len(hi)]
    ratio = means[2] / means[0] if means[0] else float('nan')
    print(f"Cutoffs: t1={t1:.2f}, t2={t2:.2f}")
    print(f"Low n={counts[0]} mean={means[0]:.1f}% | Mid n={counts[1]} mean={means[1]:.1f}% | High n={counts[2]} mean={means[2]:.1f}%")
    print(f"Ratio: {ratio:.2f}x")

    plt.rcParams.update({
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'font.family':'serif','font.size':12,
    })
    labels = [f'Low\n(<{t1:.2f})', f'Medium\n({t1:.2f}–{t2:.2f})', f'High\n(>{t2:.2f})']
    colors = ['#B0C4DE', '#6495ED', '#CD3333']
    fig, ax = plt.subplots(figsize=(6, 7))
    bars = ax.bar(labels, means, color=colors, edgecolor='black', linewidth=0.8, width=0.55, zorder=3)
    for bar, val, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f'{val:.1f}%\nn={n}', ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='black')
    ax.annotate(f'{ratio:.1f}× higher',
                xy=(1.72, means[2] + 0.5), xytext=(0.85, means[2] + 16),
                fontsize=14, fontweight='bold', color='#CD3333',
                arrowprops=dict(arrowstyle='->', color='#CD3333', lw=2,
                                connectionstyle='arc3,rad=-0.25'),
                ha='center', va='bottom')
    ax.set_ylabel('Mean Transfer ASR (%)', fontsize=14)
    ax.set_xlabel('Cross-Layer CKA (harmful)', fontsize=13)
    ax.set_ylim(0, max(means) + 30)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = _REPO_ROOT / 'figs'
    out.mkdir(exist_ok=True)
    plt.savefig(out / f'{out_base}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out / f'{out_base}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_base}.{{png,pdf}}")


if __name__ == '__main__':
    main()
