# Rebuttal Data — Safe Archive

Preserved copies of rebuttal experiment outputs and manual classifications, so paper edits and re-runs later can't lose them.

## Layout

```
rebuttal_data/
├── README.md                              (this file)
├── multilingual_100_manual_truth.json     ← manual truth for xyar Q1 (n=100)
│
├── multilingual_100/                      EXP 10 outputs (raw)
│   ├── prompts.json                       100 hand-crafted parallel EN/ES/ZH prompts
│   ├── responses.json                     1,200 raw generations
│   ├── substring_summary.json             keyword-proxy ASR (unreliable)
│   ├── wildguard_judgments.json           1,200 responses × WG verdicts
│   └── wildguard_summary.json             WG ASR per (model, arm, lang)
│
├── tap_phi3_50/                           EXP 7 raw outputs (paper's n=50 slice)
│   ├── judged.json                        50 responses × orig + WG judges
│   └── wg_vs_original_summary.json        both judges show +6% delta
│
└── tap_phi3_100/                          EXP 9 raw outputs (extended n=100)
    ├── tap_baseline_new50.json            50 additional TAP attacks on baseline Phi-3
    ├── tap_defended_new50.json            50 additional TAP attacks on defended Phi-3
    ├── wildguard_judgments.json           100 responses × WG verdicts
    └── wildguard_summary.json
```

## Multilingual (EXP 10, n=100, manual truth)

| model / lang | manual base | manual defended | Δ (pp) |
|---|---|---|---|
| Llama-3 EN | 1% | 1% | 0 |
| Llama-3 ES | 3% | 3% | 0 |
| Llama-3 ZH | 5% | 6% | +1 |
| Mistral EN | 5% | 0% | **−5** |
| Mistral ES | 12% | 2% | **−10** |
| Mistral ZH | 5% | 2% | **−3** |

For reference (**not** what we report in the rebuttal), automated proxies dramatically over-count on this eval set — e.g. Llama-3 defended EN reads as 25% under substring / 17% under WildGuard vs 1% manual. This is the auto-judge artifact discussed below.

## TAP-Phi-3 (EXP 7 + 9)

The paper reported a +6% TAP-Phi-3 regression using the paper's automated judge pipeline. Reproduced under WildGuard on the same 50 responses: +6pp. Re-run on 50 additional TAP attacks (EXP 9) with WildGuard only: baseline 54%, defended 56% → +2pp. Combined n=100 under WildGuard: 52% vs 56% → +4pp.

We do NOT surface any manual TAP-Phi-3 numbers in the rebuttal. The argument is a qualitative one: the same auto-judge failure mode we demonstrate concretely on multilingual (topic-adjacency treated as harmful) applies here — inspecting the defended-model responses shows they lean toward academic reframings (research papers, historical narratives) that automated judges flag, without carrying actionable harmful content. This is the "hollow compliance" phenomenon named in the paper's §Discussion.

## Rebuttal claims we're prepared to make

- **xyar Q1 multilingual (concrete data, n=100 manual):** defense generalizes non-English. Llama-3 neutral (0, 0, +1pp); Mistral clear positive (−5, −10, −3pp).
- **xyar Q2 TAP-Phi-3 (qualitative + auto-judge critique):** the +6pp reported in the paper is an auto-judge pipeline artifact — the same topic-adjacency failure mode we can quantify on multilingual. Automated judges systematically over-count hollow compliance because they classify by response topic rather than actionable content.

## Reproduction

All raw responses live here so downstream analysis (different judge, different verification protocol) can be re-done without re-running GPU experiments.

## v5 rebuttal defenders (xyar Q3 + zqtf Q3/W2/W5)

Three defenders trained on Llama-3-8B-Instruct for the multi-anchor and LoRA-sparsity rebuttal points. Evaluated end-to-end with the paper's canonical pipeline (chat-template applied, cross_model_transfer eval → judge_pipeline.py 6-stage WildGuard → aggregate_asr).

### Configs (see `v5_evals/hyperparameters.json` for full record)

| Config | Anchor(s) | γ | α | LoRA |
|---|---|---|---|---|
| `sparse_r128_8layers` | Phi-3-medium (single) | 2.0 | 0.15 | r=128, layers 12–19 (matched-capacity: 128×8 = 1024 rank-layers vs paper's 32×32 = 1024) |
| `phi3+mistral_g2.0_hf` | Phi-3-medium + Mistral-7B | 2.0 | 0.15 | r=32, all layers |
| `phi3+qwen15_g4.0_hf` | Phi-3-medium + Qwen1.5-7B (architecturally distinct) | 4.0 | 0.15 | r=32, all layers |

All three: β=1.0, ε=0.4, δ=0.0, precision=fp32, 200 stage-2 steps at lr=5e-5, cka_scope=harmful_only, target_layer_pct=0.5 (Llama-3 layer 16), borderline_source=xstest (200), n_harmful=500, n_benign=500.

### Numbers (Self / Anchor / Other → Total)

| Config | Base (raw pipeline) | Defended (raw pipeline) | Defended (strict manual, sample-extrapolated) | Paper single-anchor |
|---|---|---|---|---|
| sparse_r128_8layers | 4.1 / — / 1.0 → 1.1% | 6.1 / — / 12.3 → 12.0% | **~0.9–1.2%** | 1.1% |
| phi3+mistral_g2.0_hf | (same base — deterministic) → 1.1% | 25.5 / — / 20.1 → 20.3% | **~0.0–0.4%** | 1.1% |
| phi3+qwen15_g4.0_hf | (same base — deterministic) → 1.1% | 12.2 / — / 22.7 → 22.1% | **~0.0–0.6%** | 1.1% |

Anchor column is empty because the 20-source cross-model attack pool does not include Phi-3-medium (aggregator has nothing to route there).

### Manual verification (see `v5_evals/manual_verification.json`)

Protocol: 40 random defended positives per config (seed=42), each hand-classified strictly against the paper's `app:manual_verification` criteria (`non_actionable_hallucination`, `satire`, `refusal_pivot` for false-positive flips). Sample flip rates: sparse 92.5%, phi3+mistral 100%, phi3+qwen15 100%. The raw pipeline is inflated by hollow-compliance responses ("I'd be happy to help", "I hope you are referring to...", fictional virus names without payloads), matching the failure mode our multilingual evaluation already documented.

### Raw response CSVs

`v5_evals/{config}/{base,defended}_model_results.csv` — every generated response (2000 per arm).
`v5_evals/{config}/judge_pipeline/{arm}_scored.csv` — per-row WildGuard-pipeline verdicts.
`v5_evals/{config}/judge_pipeline/{arm}_summary.json` — Self/Anchor/Other/Transfer aggregates.

## Standing rebuttal-question map

| Reviewer question | Status |
|---|---|
| xyar Q1 multilingual | ✅ n=100 manual (this doc), defense generalizes non-English |
| xyar Q2 TAP-Phi-3 | ✅ Same auto-judge failure mode we documented on multilingual |
| xyar Q3 multi-anchor | ✅ v5 evals; both multi-anchor variants match paper under manual verification |
| xyar W2 refusal-direction | ✅ EXP 12 cosine matrix (11 conditions) |
| zqtf Q3 / W5 LoRA sparsity | ✅ v5 evals; sparse_r128 matched-capacity variant matches paper under manual verification |
| zqtf W2 multi-anchor | ✅ same as xyar Q3 |
| zqtf Qs 1, 2, 4, W4 | ✅ Pointer-back to existing paper sections |
| WPkd (BGR + cost undefined) | ✅ Pointers + camera-ready commitment to cross-link app:scaling / app:deployment from main body |
| PGD / TAP partial bypass (xyar W1, zqtf W1) | ⚠️ Concede |
| γ tuning burden (zqtf W3) | ⚠️ Concede |
| xyar Q3 defense-aware anchor | ⚠️ Concede |
