# Reproducing Figure 1

Figure 1 of the paper has two panels:

- **Left** (`heatmap_rsa_harm_clustered.{png,pdf}`): pairwise representational similarity (RSA on harmful prompts, layer 50%) across 20 LLMs, with three architecturally-coherent family boxes (Mistral, Llama, Eastern).
- **Right** (`bar_final.{png,pdf}`): mean GCG transfer ASR by within-family CKA tercile.

## Required data

Both panels read from the precomputed similarity / transfer matrices that ship with this repository:

| Path | What it holds |
|---|---|
| `data/similarity_matrices/cka_harm_pct50.csv`           | 20×20 pairwise CKA on harmful prompts |
| `data/similarity_matrices/rsa_harm_pct50.csv`           | 20×20 pairwise RSA on harmful prompts |
| `data/similarity_matrices/var_explained_pct50.csv`      | 20×20 Variance Explained |
| `data/similarity_matrices/neighborhood_harm_pct50.csv`  | 20×20 Neighborhood (k-NN Jaccard, harmful prompts) |
| `data/similarity_matrices/distance_ratio_pct50.csv`     | 20×20 Distance Ratio (within/between cluster) |
| `data/cross_layer_cka/cka_harm_<X>_vs_<Y>.csv`          | Per-pair full L×L cross-layer CKA matrices (used by the bar) |

The transfer ASR values consumed by the bar are hardcoded inside `scripts/regen_bar.py` from the paper's Table 11 (`tab:asr_matrix`), so the bar does not depend on a re-judged CSV.

## Regenerating both panels

```bash
python scripts/regen_heatmap.py
python scripts/regen_bar.py --exclude-hermes
```

Outputs land in `figs/heatmap_rsa_harm_clustered.{png,pdf}` and `figs/bar_final.{png,pdf}`.

## Family grouping

Both scripts use the architecturally-coherent family grouping from the paper appendix (`app:family_inclusion`):

- **Mistral** (7): Mistral-7B, Zephyr, Hermes-2, Starling, OpenChat, NeuralChat, Solar
- **Llama** (4): Llama-2, Llama-3, Vicuna, Orca-2 (Llama-2 derivative)
- **Eastern** (2): Qwen-7B, Yi-6B

Four models are excluded from the family-level analysis: Gemma, Phi-2, DeepSeek (representational outliers; pairwise similarity within their nominal grouping is much lower than within the three families above) and Hermes-2 (directional ASR-asymmetry confound; same Mistral family but excluded from the right-panel aggregation only). All 20 models still appear in the heatmap.

## Naming gotcha

Inside the data files, the model "Orca-2" is referred to as `orca2` in `data/cross_layer_cka/` and the similarity matrices, but as `orca` in the paper's ASR matrix. The regen scripts handle this with a one-line `SIM_NAME = {'orca': 'orca2'}` mapping; do not remove it.
