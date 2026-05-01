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

Train a defender adapter from scratch (Mistral example):

```bash
python -m anchor_rep.run_with_config \
    --config configs/mistral.yaml \
    --output-dir runs/mistral
```

Evaluate a pre-trained adapter from the HuggingFace Collection:

```bash
python eval/cross_model_transfer.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2 \
    --suffixes-csv attack_artifacts/advbench_suffixes_all_models.csv \
    --output-dir logs/cross_model_transfer/mistral_repro
```

Step-by-step recipes mapped to paper tables: `docs/reproducibility.md`.

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
