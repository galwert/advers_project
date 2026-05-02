# Evaluation logs

All evaluation logs released alongside the AnchorRep paper. Use these to verify reported numbers, audit manual-verification decisions, or seed follow-up analyses.

## Contents

### `audit/`
- `transfer_asr_verified.json`: HarmBench 100-prompt cross-model GCG transfer audit. Each entry includes prompt, response, automated WildGuard verdict, manual verdict, and overturn rationale. Source of the bolded ASR numbers in `tab:harmbench`.

### `mmlu/`
- 10 files: `mmlu_{baseline,cb,crl,repbend,rmu}{,_mistral}_manual.json`. Per-method MMLU scoring with manual disambiguation of edge cases. Backs the MMLU column in `tab:comparison`.

### `cross_model_transfer/`
- `{defender}_responses_full2k.json` (AnchorRep, one per defender, all five) and `{method}_retrained_{defender}_responses_full2k.json` (retrained competing defenses: CB, RepBend, RMU, CRL on Llama-3 and Mistral): per-prompt response trace for both baseline and defended sides of the 2,020-prompt cross-model GCG evaluation, plus the manual-verification verdict for every flagged response. All files share the same JSON schema (top-level keys: `defender`, `adapter_path`, `baseline_responses`, `baseline_metrics`, `defended_responses`, `defended_metrics`, `_provenance_note`, `_manual_verification_summary`; entry keys: `prompt`, `suffix`, `response`, `reason`, `attack_success`, `compliance`, `coherence`, plus `manual_verification_*` on flagged entries). The final manual judgment for each entry lives in this file---there is no separate audit file. Each flagged response carries:
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
  - `{defender}_falsereject.json`: FalseReject evaluation.

## Audit-log structure

```jsonc
{
  "description": "HarmBench 100-prompt GCG transfer ASR, manually verified",
  "date": "2026-03-25",
  "prompts_per_pair": 100,
  "sources": ["llama3", "mistral", "vicuna", "qwen14b", "phi3"],
  "targets": ["llama3", "mistral", "vicuna", "qwen14b", "phi3"],
  "harmful_prompt_indices": [...],
  "per_pair": {
    "llama3->mistral": [
      {
        "prompt": "...",
        "response": "...",
        "automated_verdict": "harmful",
        "manual_verdict": "safe",
        "overturn_category": "non-actionable hallucination",
        "overturn_rationale": "..."
      }
    ]
  },
  "per_target_summary": {...}
}
```

## License

All logs in this directory are released under CC BY 4.0 (see `LICENSE-DATA` in repo root).
