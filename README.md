# AnchorRep

Cross-model jailbreak defense via representation repulsion. Companion code release for the AnchorRep paper (NeurIPS 2026, anonymous submission).

AnchorRep trains a lightweight LoRA adapter that pushes the defended model's internal representations of harmful prompts away from those of a frozen anchor model, reducing cross-model adversarial transfer to under 2% across five defended models without attack-specific training.

## What's in this repo

| Path | Contents |
|---|---|
| `src/anchor_rep/` | Training code (`train.py`) and YAML config wrapper (`run_with_config.py`). |
| `eval/` | Evaluation scripts: cross-model GCG transfer, adaptive attacks, benchmarks, WildGuard judge. |
| `configs/` | Per-defender YAML configs reproducing each row of `tab:comparison`. |
| `data/` | AdvBench-30 training split (seed 42) + 10 refusal templates. |
| `attack_artifacts/` | GCG suffixes used in evaluation (AdvBench-derived for cross-model transfer; HarmBench-derived for OOD). |
| `logs/` | All evaluation logs and the manual-verification audit log. |
| `scripts/` | Reproduction shell scripts. |
| `docs/` | Reproducibility recipe, config reference, usage guidelines. |

Companion HuggingFace Collection (5 trained adapters): `https://huggingface.co/anonsubmission12345`.

## Install

```bash
git clone <repo-url>
cd anchor-rep
pip install -e .
```

CUDA-capable GPU is required for training and most eval scripts. Adapter training fits on a single L40S (48 GB).

## Quick start

Four common workflows are wrapped in single shell commands:

### 1. Train only

Train a defender from scratch with paper hyperparameters:

```bash
bash scripts/train_one.sh mistral
```

Or with a custom hyperparameter config (copy `configs/custom.yaml.example` to `configs/custom.yaml` and edit):

```bash
bash scripts/train_one.sh custom
```

Or override individual flags on the command line:

```bash
bash scripts/train_one.sh mistral --gamma 0.5 --stage2_steps 400
```

### 2. Eval only (no training)

Evaluate the released HuggingFace adapter for a defender (default):

```bash
bash scripts/eval_only.sh mistral
```

Evaluate a locally-trained adapter:

```bash
bash scripts/eval_only.sh mistral runs/mistral/adapter
```

Evaluate any HuggingFace adapter compatible with the base model:

```bash
bash scripts/eval_only.sh mistral some-user/SomeOtherAdapter
```

### 3. Train and evaluate, end-to-end

```bash
bash scripts/train_and_eval.sh mistral
bash scripts/train_and_eval.sh custom
```

This runs `train_one.sh` followed immediately by `eval_one.sh` on the freshly-trained adapter.

### 4. Reproduce the full main results table

```bash
bash scripts/reproduce_main.sh
```

Loops over all 5 defenders and produces per-row outputs that aggregate into `tab:comparison`.

**More examples:** [`docs/usage_examples.md`](docs/usage_examples.md) is a copy-paste cookbook covering every common workflow (eval-only, train-only, train+eval, custom configs, CLI overrides, full reproduction, anchor caching, batch sweeps).

Step-by-step recipes mapped to each paper table: [`docs/reproducibility.md`](docs/reproducibility.md).
Full YAML key reference: [`docs/config_reference.md`](docs/config_reference.md).

## How the defense works

1. CKA repulsion against a frozen anchor model on harmful prompts.
2. Refusal-direction projection (Arditi et al. 2024).
3. Coherency loss preserving benign hidden states.
4. KL preservation on borderline benign prompts (XSTest safe subset).
5. LM loss on harmful prompts (refusal targets).

Implemented as a single LoRA adapter applied at the mid-stream residual layer (50% of model depth). All five losses combine into one objective minimized over 200 to 600 gradient steps. Full method: paper Section 2.

## Data and training protocol

Training uses 30 prompts sampled uniformly at random from the 520-prompt AdvBench pool with seed 42 (predetermined split, disjoint from the held-out evaluation pool). The exact 30 prompts are in `data/advbench_train_split.json`. Borderline prompts come from XSTest (200 safe prompts), benign prompts from WikiText-2.

## Licensing

- **Code** (`src/`, `eval/`, `scripts/`, `pyproject.toml`): MIT License (see `LICENSE-CODE`).
- **Data artifacts** (`configs/`, `data/`, `attack_artifacts/`, `logs/`, model card text): CC BY 4.0 (see `LICENSE-DATA`).

External assets (datasets, base models, anchor models, judge classifier) are cited at point of use; see the paper appendix Asset Licenses table for full attribution.

## Defensive-research use only

This release is intended for defensive security research on open-weight LLMs. The released GCG suffixes are derived from publicly available attack methods on public prompts and represent a routine extension of existing practice (`attack_artifacts/README.md`). Do not deploy adapters to production without independent safety validation.

## Citation

```bibtex
@inproceedings{anchorrep2026,
  title={AnchorRep: Defending LLMs Against Cross-Model Adversarial Transfer via Representation Repulsion},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```
