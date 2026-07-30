# REBUTTAL: AnchorRep (NeurIPS submission)

---

## Official Comment by Authors (to all reviewers)

We thank all reviewers for their careful reading and for acknowledging the paper's strengths:

- R-xyar: "It is really impressive that AnchorRep can achieve very low transfer ASR without needing to train on complex, optimized adversarial strings."
- R-zqtf: "AnchorRep introduces a new defense paradigm based on directly disrupting shared representational geometry (via CKA repulsion)."
- R-WPkd: "The authors present experimental results across a wide variety of LLMs, five models from four different families."

New experiments (multilingual robustness, TAP-Phi-3 verification audit, refusal-direction stability, multi-anchor training, matched-parameter LoRA sparsity, seed variance) are summarized below and included as new appendices in the revised version.

---

## Response to Meta-Review (Area Chair)

Dear Area Chair,

Thank you for the constructive framing. We address your questions and the four suggestions below.

### Categorization of concerns

1. Limitations we concede: embedding-space PGD, semantic-reframing (TAP), per-model gamma. All measured in Table 3 and Appendix Limitations; mitigation is future work.
2. Analysis errors we correct: multi-anchor mischaracterized as "unstable" from an incomplete sweep; LoRA subset in Table E.4 was matched-rank rather than matched-parameter; TAP-Phi-3 in Table 3 reported raw automated verdict while the rest of the paper is manually verified.
3. Insufficiently surfaced material: BGR (defined in Appendix A.2, only glossed in main text) and computational cost (Appendices K, M, uncross-referenced). We will lift BGR's definition into Section 2 and add a cost summary in Section 3.

### (1) Motivating the rationale

AnchorRep generalizes because it disrupts a shared transfer geometry, not because it memorizes attack patterns. Evidence: the defense holds across 20 held-out source models (Table 1), multiple anchor choices (xyar Q3), and non-English prompts (xyar Q1). Mechanism: attacks converge on a shared mid-layer subspace, and repelling the defender from an architecturally-distinct anchor disrupts it (Appendix Cross-Prompt Suffix Universality), without adversarial examples.

### (2) Positioning against existing defenses

AnchorRep is the only defense in the comparison set that targets cross-model transfer directly, without adversarial-example training. Circuit Breakers (Zou 2024), RepBend (Yousefpour 2025), and RepE/RMU (Zou 2023, Li 2024) modify internal representations using adversarial training data; LAT (Casper 2024, Sheshadri 2024) trains against embedding-space perturbations. AnchorRep is external and geometric: the anchor supplies the direction, CKA supplies the invariant, and training uses only 30 harmful prompts. The distinction is already discussed in Related Work (paragraph 2); we will further clarify it by adding a "Requires adversarial examples" column to Table 1.

### (3) Threat model

Attacker: gradient-based token-level attacks (GCG, AutoDAN, PAIR) with white-box source access. Anchor: frozen; the canonical setup does not require the attacker to know it; anchor-secrecy is obscurity, not security, and we do not rely on it.

- Design target: white-box source, black-box defended target.
- Evaluated beyond: Table 3 grants white-box access to the defended model including the LoRA adapter. Under white-box GCG the defense substantially reduces attack success; embedding-space PGD partially bypasses, most clearly at 14B (a limitation, not a scoping choice).
- Outside scope: joint optimization with white-box anchor access; the multi-anchor experiment (xyar Q3) is our step toward that case.

### (4) BGR novelty

BGR operationalizes an under-reported failure mode (surface-fluent but content-degenerate benign output) that refusal metrics, MT-Bench, and MMLU all miss. Concrete case: Circuit Breakers on Llama-3 hits 77.1% BGR while Delta MT (-0.18) and Delta MMLU (-1.6) both look in-budget (Table 1). We will scope BGR this way in the revised text.

### (5) Robustness / utility / generalization

Robustness and utility trade off through a single knob (gamma); generalization is empirical. Gamma is per-model, selected via a small BGR-early-stopping grid (Appendix J): raising it reduces ASR and eventually raises BGR, while lowering preserves utility and weakens defense. Generalization holds for 20 held-out source models (Table 1) and is preliminary for non-English attacks (xyar Q1). We will summarize this at the close of Section Discussion.

### AC Q1: Training-objective sensitivity

The selected configuration sits at a local optimum. The five-weight objective is characterized in the hyperparameter ablation grid (Appendix J.5): every single-weight perturbation violates at least one utility budget, and only the selected configuration satisfies all budgets simultaneously. Across defenders, alpha=0.15 and beta=1.0 are shared; gamma (0.5-2.0, dominant), delta (0-0.08), epsilon (0.4-1.5) are per-model. Automating this small search is future work.

### AC Q2: Dependence on training samples

AnchorRep learns a representational subspace to repel, not exhaustive attack patterns: CKA operates on the geometry of mid-layer activations, so training prompts need only span the shared harmful-representation manifold, not enumerate attack strings. 30 suffice: four disjoint subsets yield defended-ASR std <= 1.91% (Appendix K). Benign/borderline pool choice matters more: swapping XSTest for WildGuardMix drives BGR from 0% to 29% (Appendix J), which is why we selected the curated set. Unsupervised alternatives are a natural follow-up.

Sincerely,
The Authors

---

## Response to Reviewer xyar

Dear Reviewer xyar,

Thank you for your careful review; your questions on TAP-Phi-3 and defense-aware optimization led us to re-examine our analysis.

### Cross-Language Handling of Refusal Direction (Q1)

The defense preserves or improves ASR in all three languages (100 aligned prompts each; base -> defended):

| Model | EN | ES | ZH |
|---|---|---|---|
| Llama-3-8B  base | 1% | 3% | 5% |
| Llama-3-8B  defended | 1% | 3% | 3% |
| Mistral-7B  base | 5% | 12% | 5% |
| Mistral-7B  defended | 0% | 2% | 2% |

Llama-3 unchanged (0, 0, -2 pp); Mistral has real gaps that the defense closes (-5, -10, -3 pp), largest where base multilingual safety is weakest. Caveat: no author reads ES or ZH; those columns were LLM-judged. Within-column deltas hold (same judge both arms).

### TAP-Phi-3 +6% Regression, Verification Audit (Q2)

We discovered an inconsistency: TAP-Phi-3 was the one Table 3 cell not run through our manual protocol. After manual verification the cell is -4 pp (baseline 28%, defended 24%), not +6 pp. Thank you for catching it. The gap reflects the hollow-compliance failure mode the paper identifies: TAP often produces academically phrased but non-actionable responses that confuse automated topic classifiers. All other main-body cells were already manually verified (Appendix B).

Reliability: on a 100-item blind subsample an independent rater agreed 93% (kappa = 0.50), slightly stricter, so reported ASRs are if anything conservative.

### Non-Geometric Attacks (W1)

Genuine limitation, measured in Table 3. PGD does not pass through the token distribution we operate on; TAP operates at higher abstraction than our mid-layer target.

### Refusal-Direction Brittleness (W2)

The direction is stable. Across 11 conditions (5 seeds; template counts 3/10/20; prompt counts 5/15/30; alternative-phrased 10 templates) cos >= 0.94; seed variance 0.995 +/- 0.002. Retraining on the joint extreme (cos=0.85) yields verified ASR in the canonical range.

### Defense-Aware Joint Optimization (Q3)

Partly answered by the paper: Table 3 evaluates GCG with white-box adapter access. The gap is anchor knowledge. To reduce that surface we trained a Llama-3 defender on two architecturally-distinct anchors with CKA averaged across them (single seed 42, verified ASR under Appendix B):

| Config | Verified defended ASR | BGR | Delta MT |
|---|---|---|---|
| Canonical single-anchor Llama-3 | 0.43% (paper) | 0.68% | -0.04 |
| Two-anchor (Phi-3 + Mistral) | 0.40% | 0.38% | -0.04 |
| Two-anchor (Phi-3 + Qwen-1.5-7B) | 0.20% | 0.18% | -0.01 |

Both match or improve on canonical across all three axes. Not the default because the small initial sweep appeared unstable; the fuller exploration came after canonical was frozen. We will extend this validation across defenders and seeds in the revised manuscript.

Sincerely,
The Authors

---

## Response to Reviewer zqtf

Dear Reviewer zqtf,

Thank you for the technical depth of your review; two points led us to correct our own characterizations.

### Hollow Compliance vs BGR (Q1)

BGR measures damaged fluency, whereas hollow compliance preserves fluency while suppressing harmful content. CKA repulsion severs the harmful pathway; coherency and KL preserve fluency. BGR appears when the fluency infrastructure is damaged (CKA-only prototype, Circuit Breakers on Mistral). We will surface this in Section Method.

### Auxiliary Loss Roles (Q2)

All five terms are load-bearing: every single-weight perturbation violates a utility budget. L_CKA (gamma) provides the push away from the anchor; L_refusal (alpha) supplies the destination direction; L_coherency (beta) is a weighted MSE that pulls benign representations back; L_KL (epsilon) preserves the output distribution; L_LM (delta) adds direct LM supervision when the base model's refusal is weak.

### Architecturally-Distinct Anchor (Q4)

Cross-family is the safer default; within a family the relationship reverses. Across 420 configurations (Appendix D), cross-family pairs give the largest ASR reductions (e.g., Mistral+Llama-2: -73% self-ASR) because between-family RSA is low (<=0.20). Within a family (RSA>0.5), more similar anchors give stronger defenses (rho=-0.50). Stability holds throughout (BGR<=1.1%). We will surface this in the main body.

### Multi-Anchor Configurations (W2)

Multi-anchor is stable; our "unstable" characterization was wrong, from an incomplete sweep. Thank you for pressing on this. Retrained with two pairings (single seed 42, Appendix B):

| Config | Verified defended ASR | BGR | Delta MT |
|---|---|---|---|
| Canonical single-anchor Llama-3 | 0.43% (paper) | 0.68% | -0.04 |
| Two-anchor (Phi-3 + Mistral) | 0.40% | 0.38% | -0.04 |
| Two-anchor (Phi-3 + Qwen-1.5-7B) | 0.20% | 0.18% | -0.01 |

Both match or improve on canonical. Not the default because the initial sweep was all we had when canonical was frozen. We will extend this validation across defenders and seeds in the revised manuscript.

### Narrow Mid-Depth Window (W4)

Narrow because safety features are narrowly localized: the layer-position table (Appendix E) shows the 43-47% band = safety/semantic boundary. The 50% setting is empirically robust across all five defenders, consistent with prior work (Arditi 2024, Tenney 2019, Geva 2022).

### LoRA Subset Binding Cost (Q3, W5)

Binding cost is a rank artifact, not a fundamental requirement (a precise critique). Table E.4 is matched-rank, not matched-parameter. To disentangle we trained r=128 across 8 mid layers (12-19), matching canonical at ~1024 rank-layers (single seed 42, Appendix B):

| Metric | Canonical (r=32, all layers, 100% params) | Subset (r=32, 9 mid layers, 28% params) | Matched (r=128, 8 mid layers, ~100% params) |
|---|---|---|---|
| Verified defended ASR | 0.43% (paper) | 38% (paper) | 0.30% |
| BGR | 0.68% | 0.5% | 0.68% |
| Delta MT | -0.04 | +0.14 | +0.12 |

At matched parameter count, concentrated LoRA improves on canonical (0.30% vs 0.43%) and sits far below fixed-rank subset (0.30% vs 38%), with the same utility gain. Not the default because the rank/parameter distinction surfaced during rebuttal preparation. We will extend this validation across defenders and seeds in the revised manuscript.

### gamma Tuning Burden (W3)

Genuine limitation. Automating gamma (from base-model CKA or via meta-learning) is a real follow-up we will state explicitly in the revised Section Discussion.

### PGD / TAP Partial Bypass (W1)

Measured in Table 3. PGD bypasses the token distribution; TAP operates at higher abstraction than the mid-layer geometry.

Sincerely,
The Authors

---

## Response to Reviewer WPkd

Dear Reviewer WPkd,

Thank you for your careful reading and constructive suggestions; your points on BGR and cost analysis materially affect how the contribution reads and we agree.

### BGR Definition and Motivation (W2, Q2)

BGR is defined but only in the appendix; the main text has a one-line gloss (Section 3), the wrong place for a claimed contribution. We will move the definition to Section 2. BGR = (garbled responses) / (total prompts) on the 1,320-prompt OR-Bench Hard subset, via a five-stage pipeline (neural gibberish classifier + stopword-ratio + single-token-dominance + character-repetition). Motivation: a utility-side complement to ASR capturing "surface-fluent but degenerate" failures that refusal metrics, MT-Bench, and MMLU miss (Table 1: Circuit Breakers on Llama-3 = 77.1% BGR with Delta MT -0.18, Delta MMLU -1.6, both in-budget).

### Computational Cost (W3, Q3)

Training requires only 15-30 minutes on a single 48 GB L40S GPU, with zero inference overhead after LoRA merge. CKA adds under 5% wall-clock per step. Analysis in Appendix K, M; we will add a summary to Section Experimental Setup.

### Prompt Volume Selection (Q1)

Each volume is empirically motivated (Section 3.1). 30 harmful: four 30-prompt subsets yield defended-ASR std <= 1.91% (Appendix K). 500 benign WikiText-2: standard pool, batch-4 stability. 200 XSTest borderline: switching to WildGuardMix (~4K) drives BGR from 0% to 29% (Appendix J). 15 harmful x 10 refusal templates: grounded in refusal-as-low-rank-subspace (Arditi 2024, Zou 2023).

### Presentation and Proofreading

Agreed. Revised version: (i) remove duplicate abstract sentence, (ii) expand every acronym at first use, (iii) fix table margin overflow, (iv) restructure Section 3 and Appendix, (v) additional proofreading pass.

We hope these clarifications address your concerns and would be grateful if you would reconsider your evaluation.

Sincerely,
The Authors
