# Usage examples

Every common AnchorRep workflow as a copy-paste-runnable example. Run from the repo root after `pip install -e .`.

## Cheat sheet

| Goal | Command |
|---|---|
| Eval a released adapter | `bash scripts/eval_only.sh mistral` |
| Train one defender from paper config | `bash scripts/train_one.sh mistral` |
| Train + immediately eval | `bash scripts/train_and_eval.sh mistral` |
| Train with custom hyperparameters | edit `configs/custom.yaml`, then `bash scripts/train_one.sh custom` |
| Override one knob on the command line | `bash scripts/train_one.sh mistral --gamma 0.5` |
| Reproduce the full main results table | `bash scripts/reproduce_main.sh` |

---

## 1. Eval a released adapter (no GPU training)

The 5 paper-pick adapters live on HuggingFace under `anonsubmission12345/AnchorRep-*`. To download one and evaluate it on the 2,000-prompt cross-model GCG transfer suite:

```bash
bash scripts/eval_only.sh mistral
```

What this does:
1. Downloads `anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2` from HuggingFace.
2. Loads it as a PEFT adapter on top of `mistralai/Mistral-7B-Instruct-v0.2`.
3. Generates responses for 2,000 prompt+suffix pairs from `attack_artifacts/advbench_suffixes_all_models.json`.
4. Computes refusal rate and ASR.
5. Writes results to `logs/cross_model_transfer/mistral_repro/{summary.csv, defended_model_results.csv}`.

Time: ~25 minutes on a single L40S.

You can also evaluate a different adapter:

```bash
# Evaluate the most recently trained local adapter (under runs/mistral/)
bash scripts/eval_only.sh mistral --latest

# Evaluate a specific local adapter directory
bash scripts/eval_only.sh mistral runs/mistral/defender_v2_cka_20260505_120000

# Evaluate any HuggingFace adapter compatible with the Mistral base
bash scripts/eval_only.sh mistral some-other-user/SomeOtherAdapter
```

The defender base model and `target_model` label come from `configs/mistral.yaml` (`defender.base_model`, `eval.defender_label`); the released anonymous adapter shipped with the paper is the `eval.hf_adapter` field of the same YAML and is the default when no second arg is given.

---

## 2. Train one defender from paper config

Each `configs/{defender}.yaml` reproduces the corresponding row of `tab:comparison`:

```bash
bash scripts/train_one.sh mistral
```

What this does:
1. Loads `configs/mistral.yaml` (γ=0.7, scope=all, 600 steps, lr=7.5e-5, anchor=Qwen-1.5-7B-Chat).
2. Loads the 30 AdvBench training prompts from `data/advbench_train_split.json`.
3. Loads `mistralai/Mistral-7B-Instruct-v0.2` as the defender and `Qwen/Qwen1.5-7B-Chat` as the frozen anchor.
4. Trains a LoRA adapter via two-stage CKA repulsion + auxiliary preservation losses.
5. Saves the adapter to `runs/mistral/defender_v2_cka_<timestamp>/`.

Time: ~30 minutes on a single L40S for Mistral (600 steps); 15-20 minutes for the other defenders (200 steps).

---

## 3. Train and evaluate end-to-end

```bash
bash scripts/train_and_eval.sh mistral
```

What this does:
1. Runs `scripts/train_one.sh mistral` (steps as above).
2. Auto-locates the freshly-saved adapter under `runs/mistral/`.
3. Runs `scripts/eval_one.sh mistral <fresh-adapter-path>` against the 2,000-prompt suite.

Time: ~50 minutes on a single L40S for Mistral.

Output files after completion:
```
runs/mistral/defender_v2_cka_<timestamp>/
    adapter_model.safetensors           ; the trained LoRA delta
    adapter_config.json                 ; PEFT config
    training_metrics.json               ; final loss values, hyperparameters
logs/cross_model_transfer/mistral_repro/
    base_model_results.csv              ; baseline (undefended) per-prompt outputs
    defended_model_results.csv          ; defended per-prompt outputs
    summary.csv                         ; aggregated ASR / refusal-rate / improvement
```

---

## 4. Train with custom hyperparameters

Step 1: copy the template.

```bash
cp configs/custom.yaml.example configs/custom.yaml
```

Step 2: edit `configs/custom.yaml`. For example, to try a Llama-3-anchored Mistral with stronger refusal direction:

```yaml
defender:
  name: mistral
  base_model: mistralai/Mistral-7B-Instruct-v0.2
  precision: fp32

anchor:
  name: llama3                    # changed from qwen
  precision: fp16

losses:
  alpha:   0.20                   # bumped from 0.15
  beta:    1.0
  gamma:   1.0                    # bumped from 0.7
  delta:   0.04
  epsilon: 0.8

training:
  alignment: cka
  cka_scope: harmful_only         # changed from all
  target_layer_pct: 0.5
  stage2_steps: 300               # 200 to 600 typical
  stage2_lr: 5.0e-5
  seed: 42
```

Step 3: train.

```bash
bash scripts/train_one.sh custom
# or
bash scripts/train_and_eval.sh custom
```

Adapter is saved under `runs/custom/`.

---

## 5. CLI overrides (no YAML edit needed)

Any flag exposed by `anchor_rep.train` can be passed on the command line. Extras after the config name are forwarded verbatim:

```bash
# Same as configs/mistral.yaml but with γ=0.5 instead of 0.7
bash scripts/train_one.sh mistral --gamma 0.5

# Same plus a different number of steps and disable LM loss
bash scripts/train_one.sh mistral --gamma 0.5 --stage2_steps 400 --no_lm_loss

# Use the multi-layer CKA mode
bash scripts/train_one.sh phi3 --target_layers 0.25,0.5,0.75 --layer_weights 0.3,1.0,0.3

# Run a debiased-CKA ablation
bash scripts/train_one.sh llama3 --debiased_cka
```

Useful flag families (full list: `python -m anchor_rep.train --help`):

| Flag(s) | Effect |
|---|---|
| `--alpha 0.15 --beta 1.0 --gamma 0.7 --delta 0.04 --epsilon 0.8` | the five core loss weights |
| `--cka_scope all` / `--cka_scope harmful_only` | which prompts contribute to the CKA gradient |
| `--target_layer_pct 0.5` | mid-stream is the paper choice |
| `--target_layers 0.25,0.5,0.75 --layer_weights 0.3,1.0,0.3` | multi-layer CKA mode |
| `--use_gcg_training --n_gcg_samples 100` | include GCG-suffixed prompts in the training mix |
| `--borderline_source xstest --n_borderline 200` | borderline preservation set |
| `--lora_r 32 --target_layer_pct 0.5` | LoRA rank and target layer |
| `--debiased_cka` | switch to the debiased HSIC estimator |
| `--coeff_schedule` | front-load defense / back-load retain (CB-style) |
| `--save_anchor_cache <path>` / `--load_anchor_cache <path>` | reuse pre-computed anchor hidden states |

---

## 6. Reproduce the full `tab:comparison`

```bash
bash scripts/reproduce_main.sh
```

What this does: loops over all 5 defenders and runs `eval_one.sh` against the released adapters. ~4-6 hours on a single L40S.

Per-defender outputs land in `logs/cross_model_transfer/{defender}_repro/`. Compare against the paper-original outputs already in `logs/cross_model_transfer/{defender}_*.json`.

---

## 7. Common operational patterns

### Quick smoke test (no full training)

Verify the install is correct without burning a GPU on training:

```bash
python -c "import anchor_rep; print(anchor_rep.__version__)"
python -m anchor_rep.train --help | head
python -m anchor_rep.run_with_config --config configs/mistral.yaml --output-dir /tmp/dryrun --stage2_steps 1
```

The last line runs only one optimizer step, useful as a smoke test on a small GPU.

### Resume / re-train after a config tweak

Saved adapters and training metadata go in `runs/<config_name>/`. To start fresh after editing a YAML, just delete the previous run:

```bash
rm -rf runs/mistral
bash scripts/train_one.sh mistral
```

### Run multiple custom configs in sequence

```bash
for cfg in custom_a custom_b custom_c; do
    cp configs/${cfg}.yaml configs/custom.yaml
    bash scripts/train_and_eval.sh custom
    mv runs/custom runs/${cfg}_run
done
```

### Just compute the anchor cache (one-time, then reuse across runs)

```bash
python -m anchor_rep.train --defender mistral --anchor qwen \
    --save_anchor_cache /scratch/anchor_cache.pt
# subsequent runs:
bash scripts/train_one.sh mistral --load_anchor_cache /scratch/anchor_cache.pt
```

Saves ~2 minutes per training run if you're sweeping hyperparameters with the same defender/anchor pair.

---

## 8. Inspecting results

Every run writes structured output:

```bash
# Look at the trained adapter's final hyperparameters and loss values
cat runs/mistral/defender_v2_cka_*/training_metrics.json | jq

# Aggregate cross-model transfer summary
cat logs/cross_model_transfer/mistral_repro/summary.csv

# Per-source-model breakdown of which attacks succeeded
cat logs/cross_model_transfer/mistral_per_source_summary.json | jq
```

---

## See also

- `docs/reproducibility.md` — step-by-step paper-table recipes.
- `docs/config_reference.md` — YAML key-by-key reference.
- `docs/usage_guidelines.md` — defensive-research-only release policy.
- `docs/smoke_test.md` — runbook for end-to-end fresh-machine validation.
