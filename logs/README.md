# Evaluation logs

All evaluation logs released alongside the AnchorRep paper. Use these to verify reported numbers, audit manual-verification decisions, or seed follow-up analyses.

## Contents

### `mmlu/`
- 10 files: `mmlu_{baseline,cb,crl,repbend,rmu}{,_mistral}_manual.json`. Per-method MMLU scoring with manual disambiguation of edge cases. Backs the MMLU column in `tab:comparison`.

### `harmbench_transfer/`
- 10 JSON files: `transfer_to_<tgt>_{bl,def}.json` for the 5 target × {baseline, defended} HarmBench cross-model GCG transfer evaluation reported in `tab:harmbench`. Each file aggregates 500 attacks (5 sources × 100 prompts) on a single target+side, with full per-prompt `source`/`goal`/`suffix`/`success`/`response` traces and a `by_source` summary. See `harmbench_transfer/README.md` for the schema and the relationship between the per-prompt automated `success` flags and the manually-verified ASR reported in the paper.

### `cross_model_transfer/`
- `anchorrep_{defender}.json` (AnchorRep, one per defender: `llama3`, `mistral`, `vicuna`, `qwen14b`, `phi3`) and `{method}_{defender}.json` (retrained competing defenses on Llama-3 and Mistral: `circuit_breakers`, `repbend`, `rmu`, `crl`): per-prompt response trace for both baseline and defended sides of the 2,020-prompt cross-model GCG evaluation, plus the manual-verification verdict for every flagged response. All files share the same JSON schema (top-level keys: `defender`, `adapter_path`, `baseline_responses`, `baseline_metrics`, `defended_responses`, `defended_metrics`, `_provenance_note`, `_manual_verification_summary`; entry keys: `prompt`, `suffix`, `response`, `reason`, `attack_success`, `compliance`, `coherence`, plus `manual_verification_*` on flagged entries). The final manual judgment for each entry lives in this file---there is no separate audit file. Each flagged response carries:
  - `manual_verification_inspected` (bool): true if the entry was reviewed manually.
  - `manual_verification_flipped` (bool): true if the manual judgment overrode the automated label.
  - `manual_verification_criterion`: `non_actionable_hallucination` | `satire` | `refusal_pivot` for FP overturns; `judge_missed_jailbreak` for FN recoveries; null when inspection confirmed the automated label.
  - `manual_verification_evidence`: short rationale.
  - `manual_verification_audit_kind`: `FP` (drawn from the flagged-success pool) or `FN` (drawn from the refused-response pool).
  
  The top-level `_manual_verification_summary` block reports aggregate audit counts so the verified ASR (the numbers reported in `tab:comparison` and `tab:baseline_calibration`) can be re-derived directly from this file. Three AnchorRep defenders (Mistral, Vicuna, Phi-3) retain the full 2,020-prompt response trace; Llama-3 and Qwen-14B retain the full set of flagged-success responses on both sides plus the full defended trace where it was preserved (see the `provenance` block inside each file for what was retained vs. what was discarded by the original eval pipeline). The eight competing-defense files retain the full 2,020-prompt response trace on both sides.
- `{defender}_per_source_summary.json`: aggregated ASR by source model.
- `{defender}_eval.json`: per-prompt automated judge verdicts.

### `adaptive_attacks/`
- Per-defender raw outputs from the adaptive-attack suite (GCG-self, Embedding PGD, PAIR, AutoDAN, TAP). Filenames follow `{defender}_adp_{attack}_{bl,def}.json` where `bl` is baseline (undefended) and `def` is defended.

### `benchmarks/`
- Per-defender benchmark outputs:
  - `{defender}_bench.json`: MT-Bench, OR-Bench, XSTest scores plus BGR.
  - `{defender}_falsereject.json`: FalseReject evaluation, full per-prompt traces.
    Top-level keys: `model`, `adapter_path`, `n_prompts`, `refusal_keywords`,
    `summary` (`baseline_refusals`, `baseline_rate`, `defended_refusals`,
    `defended_rate`, `delta`), `baseline` (list of 500 records), `defended`
    (list of 500 records). Each record is `{prompt, response, is_refusal}`.
    Backs the FalseReject column in `tab:comparison` (`tab:falsereject`).
    Headline numbers from this run:

    | Defender   | Baseline rate | Defended rate | Δ (pp) |
    |------------|--------------:|--------------:|-------:|
    | Llama-3    |         37.2% |         15.0% | −22.2  |
    | Mistral    |         18.0% |          8.0% | −10.0  |
    | Vicuna     |         22.6% |         35.4% | +12.8  |
    | Qwen-14B   |         34.6% |         35.6% |  +1.0  |
    | Phi-3      |         29.6% |         25.2% |  −4.4  |

## License

All logs in this directory are released under CC BY 4.0 (see `LICENSE-DATA` in repo root).
