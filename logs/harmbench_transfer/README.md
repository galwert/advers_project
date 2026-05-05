# HarmBench Cross-Model GCG Transfer Logs

Per-prompt response traces for the 5 source × 5 target × {baseline, defended}
HarmBench cross-model GCG transfer evaluation reported in `tab:harmbench`
(paper Section "Out-of-distribution safety").

## Files

10 JSON files named `transfer_to_<tgt>_{bl,def}.json`, one per
(target, side) combination — each contains all 500 attacks against that target
(5 sources × 100 prompts each):

- `<tgt>` ∈ `{llama3, mistral, vicuna, qwen14b, phi3}` — the five defended models, also serving as the five suffix sources.
- `bl` = baseline (undefended) target. `def` = defended target (LoRA adapter applied).

The 100 HarmBench prompts come from the canonical HarmBench v1 standard
behaviors set; the corresponding 500 GCG suffixes (5 source models x 100
prompts each) are bundled in `attack_artifacts/harmbench_suffixes_all_sources.json`
(filter by the `source` key to recover the per-source slice).

## Schema

```jsonc
{
  "target": "mistral_def",
  "side": "defended",                  // "baseline" or "defended"
  "total": 500,                        // 5 sources × 100 prompts
  "successes": 57,                     // automated-judge successes (raw, pre-manual-review)
  "asr": 11.4,                         // automated-judge ASR (%)
  "by_source": {                       // per-source breakdown for this target/side
    "llama3":  {"n": 100, "successes": 1,  "asr": 1.0},
    "mistral": {"n": 100, "successes": 14, "asr": 14.0},
    ...
  },
  "results": [                         // 500 entries, all sources concatenated
    {
      "source":   "mistral",                              // attack source
      "goal":     "Write a persuasive article ...",
      "suffix":   "ve... But distributed write ...",      // GCG suffix optimized on <source>; matches attack_artifacts/harmbench_suffixes_all_sources.json filtered by source
      "success":  false,                                   // automated WildGuard pipeline verdict
      "response": "..."                                    // full target response (truncated upstream to ~256 tok)
    },
    ...
  ]
}
```

## Relationship to the paper

`tab:harmbench` reports manually-verified ASR following the same protocol as
the AdvBench-derived `tab:comparison` (Appendix `app:manual_verification`):
every flagged response is human-reviewed and overturned only under the
non-actionable / satire / refusal-pivot criteria. The `success` flag in the
JSONs below is the **pre-manual-review** automated verdict; the manual
overturn decisions are the audit trail behind the headline numbers but are
not re-encoded per-prompt here. The full set of automated-flagged responses
(those with `success: true` on the defended side) is what the paper claims
to have inspected.

## License

CC BY 4.0 (see `LICENSE-DATA` at repo root). Underlying HarmBench prompts
are released under the HarmBench license (see `app:licenses` in the paper).
