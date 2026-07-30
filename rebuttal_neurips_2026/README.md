# NeurIPS 2026 Rebuttal Materials — AnchorRep

This folder archives all materials produced during the NeurIPS 2026 review-response round for the AnchorRep paper.

## Structure

```
rebuttal_neurips_2026/
├── drafts/                # Rebuttal drafts (final .txt and .md ready for OpenReview)
├── manual_verification/   # Rater prompt, verdict files, verified Table 3 numbers
├── experiment_logs/       # SLURM .out logs for every rebuttal-round training and eval job
└── experiment_runs/       # Per-run configs, metrics, and CSV/JSON outputs (weights excluded)
```

## drafts/

Final rebuttal text as prepared for OpenReview posting.

- `rebuttal_openreview_final_short.txt` — plain-text source of truth
- `rebuttal_openreview_final_short.md` — GitHub-flavored markdown version (renders correctly in OpenReview)

## manual_verification/

Full manual audit of defended-arm auto-flagged responses.

- `RATER_PROMPT.txt` — verbatim criterion applied to every response (paper's `app:manual_verification` protocol)
- `full_manual_verification/verdicts/` — 1,090 items across mam, maq_g4, sp128 configs (per-item REAL/FALSE_POSITIVE with criterion)
- `table3_manual_verification/verdicts/` — Table 3 adaptive-attacks audit (qwen14b pgd/gcg, llama3 tap, phi3 tap)
- `table3_internal/verified_table3_full.json` — internal-only full manually-verified Table 3 (absolute and delta form)

Key numbers surfaced in the rebuttal:
- mam full: 8/407 REAL → 0.40% ASR
- maq_g4 full: 4/443 REAL → 0.20% ASR
- sp128 full: 5-6/240 REAL → 0.25-0.30% ASR
- TAP-Phi-3 canonical: automated +6 pp → verified -4 pp (baseline 28%, defended 24%)
- Qwen-14B PGD released canonical: 22/41 REAL → 44% verified (vs 48% baseline; -4 pp)

## experiment_logs/

SLURM `.out` logs for every rebuttal experiment. Includes training logs (defender_v2_cka), evaluation logs (GCG, PGD, PAIR, AutoDAN, TAP, MMLU, MT-Bench, OR-Bench), and multilingual runs.

## experiment_runs/

Per-run outputs for the rebuttal experiments (LoRA `.safetensors` weights intentionally excluded). Each subdirectory follows the pattern `<TIMESTAMP>_<CONFIG_NAME>/` and contains:

- `training_config.json`, `eval_config.json` — the exact hyperparameters used
- Metrics `.json` and `.csv` — per-attack ASR, BGR, benchmark scores
- Manifest / summary MD files where written

Rebuttal-round experiment groups:
- `11_two_anchor_llama3/`, `11_two_anchor_v3/` — multi-anchor training (xyar Q3, zqtf W2)
- `14_lora_sparsity/` — matched-parameter concentrated LoRA (zqtf Q3, W5)
- `exp13/`, `exp15/` — refusal-direction stability (xyar W2) and multi-depth CKA
- `v4/`, `v5_train/`, `v6/` — extended verification runs on paper configs
- `paper_reproduce/` — Qwen-1.5-14B released-canonical adapter re-run for the PGD anomaly investigation

## What is NOT here

- LoRA adapter weight files (`*.safetensors`, `*.bin`, `*.pt`) — recoverable by re-running the training scripts with the frozen configs and single seed 42
- Model caches, HuggingFace hub caches, virtual environments
