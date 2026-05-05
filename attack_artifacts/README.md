# Attack artifacts

This directory contains the GCG suffixes used to evaluate AnchorRep.

## Contents

| File | Description |
|---|---|
| `advbench_suffixes_all_models.json` | 2000 GCG suffixes (20 source models x 100 prompts each), used for the cross-model transfer evaluation in `tab:comparison`. JSON list-of-records format; per-record keys: `model_index, model, example_index, prompt, target, suffix, loss`. |
| `harmbench_suffixes/gcg_suffixes_{defender}_100.json` | 5 per-source HarmBench-derived suffix files (100 each), used for the HarmBench out-of-distribution evaluation in `tab:harmbench`. |

## Defensive-research-only policy

These artifacts are released **only to support reproducibility of the AnchorRep paper's evaluation results**. The terms below apply:

1. **Use solely for defensive research.** Do not use these suffixes to attack third-party deployed systems. Doing so may violate the law in many jurisdictions.
2. **Do not redistribute as a standalone attack toolkit.** If you redistribute the artifacts, do so as part of a defensive research context (e.g., evaluation of a new defense), with the same usage notice.
3. **Cite the AnchorRep paper and the original GCG paper** (Zou et al., 2023) when using these artifacts.

## Dual-use note

The suffixes are derived from a publicly known attack method (GCG; Zou et al. 2023) applied to publicly available prompts (AdvBench, HarmBench). They represent a routine extension of existing public practice, not a new attack capability. Any researcher with the public method and prompts can regenerate equivalent artifacts.

We release them because:
- One-step reproduction of the paper's evaluation requires the exact suffixes used (GCG is stochastic; regeneration would not produce bit-identical outputs).
- Future defensive research building on AnchorRep needs comparable evaluation artifacts.
- The marginal offensive uplift over what is already public is negligible.

## Provenance

- AdvBench-derived suffixes were optimized using the standard public GCG procedure on prompts drawn from `https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv`.
- HarmBench-derived suffixes were optimized using the same public procedure on prompts drawn from `https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv`.
- The 20 source models are listed in the paper appendix `tab:model_key` and `tab:licenses`.

## Reporting concerns

If you believe a specific suffix in this collection enables an attack capability that is not already trivially obtainable from the public GCG method on public prompts, please open a discussion on the paper's OpenReview page (after the anonymous review period concludes). The release will be revised if a non-trivial novel capability is identified.
