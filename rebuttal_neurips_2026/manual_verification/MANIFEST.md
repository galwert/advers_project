# Rebuttal Data — Manifest

Complete archive of all logs, judge decisions, manual verdicts, and full model responses referenced in the rebuttal. Everything the rebuttal cites is reproducible from files under this tree.

## Root-level files

- `README.md` — original narrative overview (may be stale relative to this manifest)
- `MANIFEST.md` — this file
- `rebuttal_status_20260724.md` — earlier internal status snapshot

## Per-experiment archives

### 1. Multilingual robustness (xyar Q1)

- `multilingual_100/prompts.json` — 100 hand-crafted parallel EN/ES/ZH harmful prompts
- `multilingual_100/responses.json` — 1,200 raw model responses (2 defenders × 3 langs × 2 arms × 100 prompts)
- `multilingual_100/substring_summary.json` — substring-refusal proxy verdicts
- `multilingual_100/wildguard_judgments.json` — WildGuard verdicts per response
- `multilingual_100/wildguard_summary.json` — aggregate WG verdicts
- `multilingual_100_manual_truth.json` — **manual verification (via Claude) of every response**

### 2. TAP-Phi-3 auditing (xyar Q2)

- `tap_phi3_50/judged.json` — old n=50 TAP responses + judge verdicts
- `tap_phi3_50/wg_vs_original_summary.json` — WG vs paper's original judge disagreement
- `tap_phi3_100/tap_baseline_new50.json`, `tap_defended_new50.json` — additional 50 attacks
- `tap_phi3_100/wildguard_judgments.json`, `wildguard_summary.json` — WG verdicts

### 3. Refusal-direction stability (xyar W2)

- `refusal_direction_stability/summary.json` — 11-condition pairwise cosine matrix + verdict

### 4. Paper canonical adapter eval (calibration point)

- `paper_canonical_eval/{base,defended}_model_results.csv` — 2,000 raw responses per arm
- `paper_canonical_eval/judge_pipeline/{base,defended}_scored.csv` — per-row 6-stage WildGuard pipeline verdicts
- `paper_canonical_eval/judge_pipeline/{base,defended}_summary.json` — Self/Anchor/Other/Transfer aggregates

### 5. v4 rebuttal defenders (multi-anchor + LoRA sparsity — original)

Under `v5_evals/`:
- `hyperparameters.json` — all training hyperparameters for the 3 v4 defenders
- `manual_verification.json` — **full manual verification protocol + verdicts for every defended positive**
- `sparse_r128_8layers/` — full eval CSVs + judge_pipeline outputs
- `phi3+mistral_g2.0_hf/` — same
- `phi3+qwen15_g4.0_hf/` — same
- `bgr_check/` — BGR + XSTest utility numbers per config (+ baseline cache)
- `mtbench_check/` — MT-Bench utility numbers per config
- `mmlu_check/` — MMLU utility for phi3+qwen15

### 6. v5t hyperparameter sweep

Under `v5t_sweep/`, one dir per config: `sp128_g1a5_P`, `sp128_g1a15_P`, `maq_g2a5_P`, `maq_g1a5_P`, `maq_g4a15_P`. Each contains:
- `{base,defended}_model_results.csv` — full response CSVs
- `judge_pipeline/{base,defended}_scored.csv` — pipeline verdicts
- `judge_pipeline/{base,defended}_summary.json` — Self/Anchor/Other aggregates
- `utility.json` — BGR + MT-Bench numbers

### 7. v6 seed-variance retrains (reproducibility check)

Under `v6_seed_variance/`, one dir per (config × seed): `sp128_s42`, `sp128_s99`, `maq_g4_s42`, `maq_g4_s99`, `mam_s42`, `mam_s99`, `maq_g4_s99_RERUN`, `maq_g4_s7`, `mam_g1.5_s42`. Same file structure as v5t_sweep.

### 8. Alt-refusal training (xyar W2 Part B)

Under `exp13_alt_refusal/`:
- `{base,defended}_model_results.csv` — full responses
- `judge_pipeline/{base,defended}_scored.csv` — per-row verdicts
- `judge_pipeline/{base,defended}_summary.json` — aggregates

## Manual review workfiles

Under `manual_review_workfiles/`:
- `all_defended_positives.json` — every defended positive across all 10 primary configs (2,553 items total): prompt + full response + source_model, keyed by config
- `pass1_borderline.json` — output of first-pass rule-based auto-FP classifier (garbage / prompt-echo / soft-refusal / no-content filters applied); items that survived require hand review
- `v6_borderline.json` — same as above but for v6 seed-variance configs
- `auto_verification_summary.json` — count of auto-FP by rule-category, per config

## What's NOT in this archive (and where to find it)

- **Trained LoRA adapters (~330 MB each)**: `/home/wertheizer/advers_project/experiments/runs/v4/`, `v5_train/`, `v6/`, `paper_reproduce/`. Not copied because of size; adapter_config.json + training_metrics.json ARE preserved in each rebuttal_data subdirectory via the summary/utility JSON metadata.
- **Raw SLURM / direct-ssh job logs**: `/home/wertheizer/advers_project/experiments/logs/`. Preserved on disk; not copied because verbose.
- **Base models (Llama-3, Phi-3, Mistral, Qwen)**: HuggingFace cache; re-downloadable.

## How to reproduce any claim in the rebuttal

| Rebuttal claim | Data source |
|---|---|
| Multilingual Δ = 0/0/+1 pp Llama-3, −5/−10/−3 pp Mistral | `multilingual_100_manual_truth.json` |
| TAP-Phi-3 auto vs manual gap | `tap_phi3_50/wg_vs_original_summary.json` |
| Refusal cos 0.995 ± 0.002, worst 0.85 | `refusal_direction_stability/summary.json` |
| Paper canonical reproduces 1.5% raw ASR | `paper_canonical_eval/judge_pipeline/defended_summary.json` |
| Every v4/v6/v5t verified ASR | `v5_evals/manual_verification.json` |
| v4 defender BGR / MT-Bench numbers | `v5_evals/bgr_check/*.json`, `v5_evals/mtbench_check/*.json` |
| v6 seed variance table | `v6_seed_variance/*/utility.json` + `judge_pipeline/defended_summary.json` |
| alt_refusal defender ASR | `exp13_alt_refusal/judge_pipeline/defended_summary.json` |
