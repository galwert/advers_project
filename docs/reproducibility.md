# Reproducibility recipe

Step-by-step recipes for reproducing each table in the AnchorRep paper. All commands run from the repo root.

## 0. Environment

```bash
pip install -e .
```

Verify a CUDA GPU is visible:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count())"
```

## 1. Reproduce a single row of `tab:comparison`

Each `configs/{defender}.yaml` reproduces the corresponding row of the main results table. Mistral example:

```bash
python -m anchor_rep.run_with_config \
    --config configs/mistral.yaml \
    --output-dir runs/mistral
```

Output: `runs/mistral/defender_v2_cka_<TIMESTAMP>/adapter_model.safetensors` (~320 MB). Each train run lands in a fresh timestamp-suffixed subdirectory under `runs/<config>/`, so multiple runs accumulate side by side; pass `--latest` to `scripts/eval_one.sh` to evaluate the most recent one.

Approximate runtime on a single L40S (48 GB): 15 to 25 minutes per defender.

## 2. Skip training, evaluate the released adapters

The 5 paper-pick adapters are already published in the companion HuggingFace Collection. To evaluate one defender end-to-end (generation + WildGuard pipeline + Self/Anchor/Other breakdown):

```bash
bash scripts/eval_one.sh mistral
```

`eval_one.sh` runs three steps:

1. `eval/cross_model_transfer.py` --- generates the 2,000 prompt $\times$ defender responses on both baseline and defended sides; saves `defended_model_results.csv`.
2. `eval/judge_pipeline.py` --- scores each defended response through the canonical WildGuard pipeline (Stages 0--5 from Appendix `app:judge`: degenerate-output / gibberish / quality / refusal / auto-classification / WildGuard tiebreaker); saves `defended_judged.csv` with an `is_jailbroken` column.
3. `eval/aggregate_asr.py` --- breaks the verified ASR down by attack-source role (Self / Anchor / Other) and writes `defended_judged_summary.json`.

The `is_jailbroken` column matches the verdict reported in `tab:comparison`. The simple refusal-keyword stat printed by step 1 is just a quick sanity check.

To call the steps independently or with a non-default adapter, see `scripts/eval_one.sh`.

## 3. Reproduce the full `tab:comparison`

```bash
bash scripts/reproduce_main.sh
```

This loops over all 5 defenders, runs cross-model transfer, then aggregates the results into a CSV that mirrors `tab:comparison`. Total runtime: 4 to 6 hours on one L40S.

## 4. Reproduce `tab:adaptive-attack`

```bash
python eval/adaptive_attacks_basic.py    --adapter anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2
python eval/adaptive_attacks_advanced.py --adapter anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2
```

## 5. Reproduce `tab:harmbench`

`eval/cross_model_transfer.py` is schema-agnostic — pass the
HarmBench bundle as `--suffixes` and it will treat each entry's `goal`
as the prompt and `source` as the attack-source label, mirroring the
AdvBench (`prompt` + `model`) flow:

```bash
for tgt in llama3 mistral vicuna qwen14b phi3; do
    python eval/cross_model_transfer.py \
        --base-model "$(yq .defender.base_model configs/${tgt}.yaml)" \
        --adapter "anonsubmission12345/AnchorRep-${tgt}" \
        --suffixes attack_artifacts/harmbench_suffixes_all_sources.json \
        --output-dir "logs/cross_model_transfer/harmbench_${tgt}"
done
```

The 500-entry bundle drives baseline + defended generation in one pass per
target; the `source_model` column in the output preserves the per-source
breakdown so you can recover the 5×5 cell numbers for `tab:harmbench`.
(Adjust the `--adapter` argument to the exact HF repo name; see
`configs/{tgt}.yaml`.)

## 6. Reproduce benchmarks (MT-Bench, MMLU, OR-Bench, XSTest, FalseReject, BGR)

```bash
python eval/benchmarks.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2 \
    --output-dir logs/benchmarks/mistral_repro
```

## 7. Manual verification protocol

The paper applies a manual verification step over automated WildGuard judgments (paper Appendix `app:manual_verification`). The verified verdict and overturn rationale for every flagged response is inlined in the released `logs/cross_model_transfer/{method}_{defender}.json` files via the `manual_verification_*` entry fields.

## Validation suite

After running the recipes above, verify your numbers against `logs/cross_model_transfer/` (paper-original outputs) using a per-row comparison. ASR should match within seed-42 variance (per-model standard deviation up to 1.91% across four independent training-subset draws; see paper appendix `app:seed_variance`).

## Reproduction tips

- The defended model uses greedy decoding (`do_sample=False`) for deterministic generation.
- Training uses a fixed random seed (42) for data shuffling and LoRA initialization.
- 14B defenders use fp16 throughout; <=8B defenders use fp32 for training.
- Anchor models can run in fp16 freely; CKA is correlation-based and scale-invariant.
