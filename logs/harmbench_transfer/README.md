# HarmBench Cross-Model GCG Transfer Logs

Per-prompt response traces for the 5 source × 5 target × {baseline, defended}
HarmBench cross-model GCG transfer evaluation reported in `tab:harmbench`
(paper Section "Out-of-distribution safety").

## Files

50 JSON files named `transfer_<src>_to_<tgt>_{bl,def}.json`, one per
(source, target, side) cell:

- `<src>` and `<tgt>` ∈ `{llama3, mistral, vicuna, qwen14b, phi3}` — the five
  defenders, also serving as the five suffix sources.
- `bl` = baseline (undefended) target. `def` = defended target (LoRA adapter applied).

The 100 HarmBench prompts come from the canonical HarmBench v1 standard
behaviors set; the corresponding GCG suffixes optimized per source model
are in `attack_artifacts/harmbench_suffixes/`.

## Schema

```jsonc
{
  "source": "mistral",
  "target": "mistral_def",
  "asr": 14.0,                        // automated-judge ASR (raw, no manual review)
  "successes": 14,                    // count of automated-judge successes
  "total": 100,
  "results": [
    {
      "goal": "Write a persuasive article ...",
      "success": false,               // automated WildGuard pipeline verdict
      "response": "..."               // full target response (truncated upstream to ~256 tok)
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
