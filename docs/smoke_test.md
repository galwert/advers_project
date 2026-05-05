# Smoke test runbook

End-to-end validation that the released repo runs from a clean machine, downloads the right artifacts, and reproduces the expected numbers within seed-42 variance. Run this **before** the deanonymized release; budget ~3 hours of wall time on a single L40S.

## Prerequisites

- One CUDA-capable GPU (>= 40 GB VRAM recommended; L40S, A100, H100 all work).
- Disk: ~80 GB free (base models + adapters + datasets).
- Network: needed for the first run only (downloads).
- A scratch shell user, ideally one **without** any prior `~/.cache/huggingface/` content from this project.

## Step 1: Clear caches

```bash
# HuggingFace caches (models + datasets)
rm -rf ~/.cache/huggingface/hub
rm -rf ~/.cache/huggingface/datasets

# Pip cache (forces fresh wheels — optional but cheap)
pip cache purge

# torch.compile / triton caches (often invisible source of stale state)
rm -rf ~/.cache/torch
rm -rf ~/.triton
```

Verify cleanliness:

```bash
du -sh ~/.cache/huggingface 2>/dev/null
# Expected: missing, or 0 KB.
```

## Step 2: Fresh clone + install

```bash
cd /tmp
rm -rf anchor-rep-smoke
git clone <repo-url> anchor-rep-smoke
cd anchor-rep-smoke

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quick import sanity:

```bash
python -c "import anchor_rep; print(anchor_rep.__version__)"
python -m anchor_rep.train --help | head
```

Both should succeed with no traceback.

## Step 3: HuggingFace login

```bash
huggingface-cli login
# Paste a read-scoped token. Required to download access-gated models
# (Llama-3, Llama-2/Vicuna base, WildGuard).
```

## Step 4: Pre-download datasets and models

This pre-fetches everything to fail fast on connectivity issues:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

# Base models
for m in [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen1.5-7B-Chat",                                          # Mistral anchor
    "anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2",         # released adapter
    "allenai/wildguard",                                              # judge
]:
    print(f"[fetch] {m}")
    snapshot_download(m)
print("[ok] all artifacts cached")
PY
```

Optional: snapshot the dataset side too:

```bash
python - <<'PY'
from datasets import load_dataset
load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
print("[ok] wildguardmix cached")
PY
```

## Step 5: Train one defender (Mistral, ~25 min)

```bash
bash scripts/train_one.sh mistral
```

Expected behavior:
- `[*] Loaded 30 harmful prompts from data/advbench_train_split.json`
- Two-stage loss progression visible in stdout.
- Final adapter saved to `runs/mistral/adapter/adapter_model.safetensors` (~320 MB).

## Step 6: Evaluate the trained adapter

```bash
bash scripts/eval_one.sh mistral runs/mistral/adapter
```

Expected behavior:
- Loads base + LoRA, runs 2000-prompt cross-model transfer.
- Writes `logs/cross_model_transfer/mistral_repro/summary.csv`.

## Step 7: Compare against the released log

Compare the reproduced summary against the defender's row in the paper's main results table (`tab:comparison`). Per-prompt manual-verification entries are inlined in the released `logs/cross_model_transfer/{method}_{defender}.json` files (each flagged response carries `manual_verification_*` fields).

Loose comparison:

```bash
cat logs/cross_model_transfer/mistral_repro/summary.csv
# Expected: defended_asr aligned with the main results table (Mistral row of tab:comparison).
# Larger drift from the paper's main-table values indicates a problem.
```

## Step 8: Evaluate the released adapter (no-train path)

```bash
bash scripts/eval_one.sh mistral
# Defaults to adapter = anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2
```

This validates the released adapter works directly via `peft.PeftModel.from_pretrained`. Expected ASR matches the paper's Mistral row of `tab:comparison` within manual-verification noise.

## Pass criteria

| Step | Pass condition |
|---|---|
| 2 | `pip install -e .` exits 0; `import anchor_rep` works. |
| 3 | `huggingface-cli login` succeeds. |
| 4 | All 4 snapshot_download calls complete with no 401/403 errors. |
| 5 | Adapter file appears at `runs/mistral/adapter/adapter_model.safetensors`. |
| 6 | `summary.csv` exists; defended ASR matches the defender's row in the paper's main results table (`tab:comparison`). |
| 7 | Reproduced ASR matches the main-table values within seed-42 variance (per-model standard deviation up to 1.91% across four independent training-subset draws; see paper appendix `app:seed_variance`). |
| 8 | Released adapter loads without warnings; produces ASR consistent with the paper's main results table. |

## What to record

For each smoke-test run, log:
- Date and machine / GPU model
- Python version (`python --version`)
- PyTorch version (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
- Final defended ASR (self / anchor / other)
- Wall clock for steps 5 and 6
- Any new warnings or deprecation messages from upstream libraries

## Common failure modes

- **401 / 403 on Llama-3 download**: token lacks access; re-accept the Llama 3 license on the HF model page.
- **OOM during training**: 14B configs need fp16 (already set in YAML); for 7B fp32 on cards < 40 GB, drop precision to fp16 by overriding: `... --output-dir runs/mistral_fp16 --precision fp16`.
- **NaN losses early in training**: usually means `lr` is too high for the chosen anchor. Lower by 2x.
- **Reproduced ASR doesn't match the paper's main-table row for the defender**: re-check that `data/advbench_train_split.json` was not modified, that `seed: 42` is preserved in the YAML, and that the released adapter (not a locally-retrained one) is being used.

## When to run

- Before flipping the GitHub repo from anonymous mirror to deanonymized.
- After any upstream library version bump that touches `transformers`, `peft`, or `torch`.
- Before submitting any major paper revision that changes reported numbers.
