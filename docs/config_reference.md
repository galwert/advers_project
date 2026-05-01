# YAML config reference

Each `configs/{defender}.yaml` is the single source of truth for a defender's training run. The fields below are translated to `train.py` CLI flags by `src/anchor_rep/run_with_config.py`.

## Top-level keys

```yaml
defender:        # which model to harden
anchor:          # which model to repel against
losses:          # five loss weights
training:        # optimizer, schedule, scope, layer
lora:            # LoRA rank/alpha
borderline:      # KL preservation source
data:            # paths to training data
```

## Defender

| Key | Type | Description |
|---|---|---|
| `name` | string | Short alias used by `train.py` (e.g., `mistral`, `llama3`, `vicuna`, `qwen14b`, `phi3`). |
| `base_model` | string | Hugging Face model id. Documented for clarity; the alias resolves to this internally. |
| `precision` | string | `fp32`, `fp16`, or `4bit`. <=8B models use fp32; 14B models use fp16. |

## Anchor

| Key | Type | Description |
|---|---|---|
| `name` | string | Short alias for the anchor model. |
| `precision` | string | Anchor precision. fp16 is fine because CKA is scale-invariant. |

## Losses

All loss weights are paper-tuned per defender. See paper Methodology for exact roles.

| Key | Type | Description |
|---|---|---|
| `alpha` | float | Refusal-direction weight. Lower = less refusal. Range: 0.10 to 0.15. |
| `beta` | float | Coherency weight (benign output preservation). Always 1.0 in paper. |
| `gamma` | float | CKA repulsion weight. Most model-sensitive parameter (range 0.3 to 2.0). |
| `delta` | float | LM loss weight (next-token preservation on harmful prompts). |
| `epsilon` | float | KL-divergence weight (logit preservation on benign prompts). |

## Training

| Key | Type | Description |
|---|---|---|
| `cka_scope` | string | Prompts contributing to CKA repulsion: `harmful_only` (default) or `all`. Mistral uses `all`. |
| `target_layer_pct` | float | Where in the residual stream CKA is computed. Always 0.5 (mid-stream) in paper. |
| `stage2_steps` | int | Optimizer steps. 200 for most defenders; 600 for Mistral. |
| `stage2_lr` | float | Learning rate. Typical: 5e-5 for Llama-3, 7.5e-5 for Mistral, 2e-4 for Vicuna/Phi-3, 2e-5 for Qwen-14B. |
| `seed` | int | Random seed for data shuffling and LoRA initialization. Always 42. |

## LoRA

| Key | Type | Description |
|---|---|---|
| `rank` | int | LoRA rank. Always 32. |
| `alpha` | int | LoRA alpha. Always 64. |

LoRA target modules are the standard set for the supported architectures (q, k, v, o, up, down, gate projections). Configurable in `train.py` if needed.

## Borderline

| Key | Type | Description |
|---|---|---|
| `source` | string | Borderline prompt source. `xstest` is the paper default. Other choices: `wildguard`, `orbench`, `falsereject`. |
| `count` | int | Number of borderline prompts. 200 in paper. |

## Data

| Key | Type | Description |
|---|---|---|
| `harmful_prompts_file` | string (path) | JSON file with harmful training prompts. Default: `data/advbench_train_split.json` (30 AdvBench prompts, seed 42). |
| `gcg_data_path` | string (path) | GCG suffixes CSV. Default: `attack_artifacts/advbench_suffixes_all_models.csv`. |

## Overriding from the command line

Any extra arguments after `--output-dir` are passed verbatim to `train.py`, allowing one-off overrides without editing the YAML:

```bash
python -m anchor_rep.run_with_config \
    --config configs/mistral.yaml \
    --output-dir runs/mistral_g05 \
    --gamma 0.5
```

## Adding a new defender

1. Copy an existing YAML (the closest base model is a good starting point).
2. Update `defender.name`, `defender.base_model`, and `defender.precision`.
3. Pick an anchor model with high cross-family representational distance (paper Appendix `app:hyperparams:anchor`).
4. Tune `gamma`: start at 1.0 and adjust based on initial defender-anchor CKA. Use BGR as an early-stop signal.
5. Run a short hyperparameter sweep over `alpha`, `epsilon`, `delta`. Paper Appendix `app:hyperparams` documents the search ranges.
