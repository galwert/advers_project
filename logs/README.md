# Evaluation logs

All evaluation logs released alongside the AnchorRep paper. Use these to verify reported numbers, audit manual-verification decisions, or seed follow-up analyses.

## Contents

### `audit/`
- `transfer_asr_verified.json`: HarmBench 100-prompt cross-model GCG transfer audit. Each entry includes prompt, response, automated WildGuard verdict, manual verdict, and overturn rationale. Source of the bolded ASR numbers in `tab:harmbench`.

### `mmlu/`
- 10 files: `mmlu_{baseline,cb,crl,repbend,rmu}{,_mistral}_manual.json`. Per-method MMLU scoring with manual disambiguation of edge cases. Backs the MMLU column in `tab:comparison`.

### `cross_model_transfer/`
- Per-defender raw outputs from the 2000-prompt cross-model transfer evaluation:
  - `{defender}_responses_full2k.json`: full prompt -> response trace for the defended model.
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
