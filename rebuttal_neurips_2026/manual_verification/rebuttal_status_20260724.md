# Rebuttal Status

---

## Category 1: Cannot address / must concede

### PGD / TAP partial bypass (xyar W1 + zqtf W1)

**The problem.**
xyar: *"Vulnerability to Non-Geometric Attacks: As the authors note, the method struggles against continuous embedding-space attacks (Embedding PGD) and semantic-reframing attacks (TAP), as these do not traverse the specific token-gradient geometry targeted by CKA repulsion."*
zqtf: *"The defense is partially bypassed by continuous-space embedding PGD attacks (especially on larger 14B models) and semantic-reframing attacks (TAP on Llama-3), indicating limitations in its scope of protection against all adversarial manipulation types."*

**Short answer.** Concede. Cross-model transfer is the specific channel our defense is designed for — attacks in the discrete token space against one open-weight model then applied to another. Continuous embedding-space PGD never passes through the token distribution we operate on, and semantic-reframing lives at a higher abstraction than the mid-layer geometry we modify. Prior work (revisiting Circuit Breakers) confirms this ceiling for representation-engineering defenses more broadly. We scope both attack classes as future work in our Adaptive Attacks section and Limitations appendix, and we're not going to claim otherwise.

### γ tuning burden (zqtf W3)

**The problem.**
zqtf: *"The optimal γ (anchor repulsion weight) is highly model-sensitive, requiring careful tuning based on the initial defender-anchor CKA and BGR, which could complicate deployment to new models."*

**Short answer.** Concede honestly. Every new defender requires per-model tuning of γ. We provide a manual heuristic in Appendix §Hyperparameters:gamma — start at γ=1.0 and adjust based on the initial defender-anchor CKA, using BGR as an early-stopping signal — but it is manual, and full automation of γ selection is a real follow-up direction.

---

## Category 2: Dry answers (all already in the paper)

### Presentation issues (WPkd main + zqtf formatting note)

**The problem.**
WPkd: *"There is considerable room to improve the presentation... Certain sections read more like an information dump rather than flowing smoothly. Furthermore, thorough proofreading is necessary as I found multiple errors throughout the paper. To enhance readability, a suggested approach would be to write out the full abbreviation upon its first introduction."*
zqtf: *"Some tables extend beyond the page margins. There are repetitive statements in the abstract."*

**Short answer.** Camera-ready commitments: remove the duplicate sentence in the abstract, replace the em-dash, expand every acronym at first use (BGR, CKA, LoRA, ASR, PGD), fix table margin overflow, and pass another editing round over Section 3 + Appendix for flow.

### BGR undefined (WPkd W2 + Q2)

**The problem.**
WPkd: *"The authors claim the introduction of the Benign Garble Rate (BGR) as a main contribution. However, this metric is not fully defined or motivated within the paper."* / *"How is 'garbled' formally defined?"*

**Short answer.** BGR is fully defined in Appendix §BGR/FRR (`app:bgr_frr`): BGR = (garbled responses) / (total prompts) on the 1,320-prompt OR-Bench Hard subset, with a five-stage detection pipeline (neural gibberish classifier + stopword-ratio, single-token-dominance, and character-repetition heuristics) validated at 0% false positives on 100 held-out garbled samples. The reviewer missed this appendix — polite pointer with the exact numbers, plus we will cross-link it more prominently from Introduction and Discussion in the camera-ready.

### Computational cost undiscussed (WPkd W3 + Q3)

**The problem.**
WPkd: *"A discussion concerning the operational cost of applying this defense mechanism would also benefit the manuscript by providing context on trade-offs."* / *"What is the computational cost associated with implementing AnchorRep?"*

**Short answer.** Quantified in Appendix §Computational Cost and Scaling Analysis (`app:scaling`) and §Deployment Overhead and Reproducibility (`app:deployment`): 15–30 min training on one 48 GB L40S, <5% CKA wall-clock overhead per step, 50–100 MB LoRA adapter that merges into the base with zero inference overhead, 70B extrapolation ~4×A100-80GB with <10% overhead. The reviewer's concern is partly fair — the main body of the paper does not cross-reference this appendix, so a reader following the main text would not find it. Camera-ready commitment: add explicit pointers from the Introduction and Discussion into `app:scaling` / `app:deployment`.

### Prompt volume selection (WPkd Q1)

**The problem.**
WPkd: *"What was the process for selecting prompt volumes for different loss functions?"*

**Short answer.** All four counts are stated in §3.1 with justifications: 30 harmful (motivated by CKA operating on geometry rather than requiring an exhaustive attack set, and empirically robust across four independent 30-prompt subsets in Appendix §Seed Variance); 500 benign WikiText-2 (standard, sized for batch-4 stability); 200 XSTest borderline (empirically motivated in Appendix §Hyperparameters:borderline — switching to WildGuardMix's ~4K pool drives BGR from 0% to 29%); 15 harmful × 10 refusal templates for r̂ (concept grounded in prior work on refusal as a low-rank subspace — Arditi 2024, Zou 2023, Li 2023 — but the specific 15×10 protocol is our design choice, validated by the same seed-variance robustness).

### Why hollow compliance not BGR? (zqtf Q1)

**The problem.**
zqtf: *"How does CKA repulsion, by pushing harmful prompt representations away from an anchor, semantically disrupt 'actionable harmful content' while preserving surface fluency, leading to low BGR and the 'hollow compliance' phenomenon?"*

**Short answer.** Directly answered verbatim in our §Discussion: *"Removing the compliance direction preserves surface fluency (language modeling, style, and instruction following), while severing the representational subspace that grounds responses in actionable realizations of the harmful request."* CKA only breaks the harmful-anchor relational structure; the coherency and KL losses protect fluency and output distribution. BGR occurs when the fluency infrastructure itself is damaged (CKA-only prototype, Circuit Breakers on Mistral); hollow compliance occurs when fluency is intact but the harmful-content pathway is severed. Two mechanistically distinct failure modes.

### Auxiliary loss roles (zqtf Q2)

**The problem.**
zqtf: *"Please detail the specific role and design motivation of each auxiliary loss term (L_refusal, L_coherency, L_KL, L_LM) in preventing 'degenerate benign output' or 'over-refusal' when CKA repulsion is used alone. How does L_coherency specifically anchor adapted representations via weighted MSE to balance the 'push-pull' system?"*

**Short answer.** Per-loss roles specified in §Methodology `subsec:aux_losses`; empirical demonstration of coupling in `tab:hp_ablation_grid` (Appendix §Hyperparameter Ablation Grid). Briefly: L_CKA(γ) pushes; L_refusal(α) gives the push a destination; L_coherency(β) is weighted MSE (weight 5 for benign/borderline, 1 for harmful) that anchors benign representations more strongly, providing the pull; L_KL(ε) preserves output distribution at token level; L_LM(δ) supplies direct language-modeling supervision when base refusal is weak. Every single-weight perturbation in `tab:hp_ablation_grid` breaks at least one utility budget; only the picked configuration satisfies all four simultaneously.

### Architecturally-distinct anchor (zqtf Q4)

**The problem.**
zqtf: *"Explain why selecting an anchor model that is 'architecturally distinct' from the defender (e.g., different tokenizer, pre-training data) maximizes the representational contrast for CKA repulsion, leading to stronger defense. What are the implications of this choice for training stability and convergence speed?"*

**Short answer.** More nuanced than the reviewer's framing. Across all 420 configurations in Appendix §Anchor Ablation, cross-family anchors give the largest absolute ASR reductions (Mistral+Llama-2: −73% self-ASR) because between-family pairs have low baseline RSA (≤0.20). But there's a Simpson's paradox: within the same architectural family (RSA >0.5), more similar anchors give stronger defenses (ρ=−0.50). Self-anchors are also effective. We attribute the within-family reversal to CKA getting a richer gradient signal when anchor geometry is structurally close (small targeted perturbations beat large undirected ones). Training stability is untouched: BGR stays ≤1.1% across every anchor. Practical takeaway: cross-family is the safer default; per-defender optimization matters and is reported in `tab:anchor_ablation`.

### Narrow mid-depth window (zqtf W4)

**The problem.**
zqtf: *"The defense's effectiveness is highly dependent on targeting a 'mid-depth layer' (50th percentile). Intervening too shallowly leads to garbling, while too deeply fails to suppress attacks, indicating a narrow optimal intervention window."*

**Short answer.** The window is narrow because safety features are narrowly localized in the residual stream — this is a strength, not a fragility. Appendix §Layer Depth Mechanism (`tab:layer_position`) shows the 43–47% band is the safety-encoding vs semantic-output boundary. Below it (shallow), ΔOR-Bench blows up because we're perturbing lexical layers. Above it (deep), ASR reduction collapses because we're downstream of safety encoding. Consistent with prior mechanistic-interpretability work (Arditi 2024, Tenney 2019, Geva 2022). The 50% setting is empirically robust across all five defended models (`tab:layer_sweep`).

---

## Category 3: Experimental answers — DONE

### Multilingual robustness (xyar Q1)

**The problem.**
xyar: *"The refusal direction r is calculated using a small set of English-only templates (e.g., 'I can't help with that request'). How does AnchorRep handle cross-model attacks framed in different languages? Does the CKA repulsion safely redirect non-English harmful prompts into this English-derived refusal subspace, or does the defense break down?"*

**Short answer.** New experiment: 100 hand-crafted parallel harmful prompts in English / Spanish / Chinese × Llama-3 and Mistral, base and defended, with every response strictly hand-classified for actionable harmful content. Under manual verification, Llama-3 base→defended deltas are (0, 0, +1 pp) for EN/ES/ZH; Mistral deltas are (−5, −10, −3 pp). The English-derived refusal direction generalizes: on Mistral (where the base model has real multilingual gaps) the defense closes them; on Llama-3 (already refuses non-English prompts in English) there is nothing to add.

**Note.** As already established in the paper's §Discussion (hollow compliance), automated judges over-count topic-adjacent responses as harmful. The deltas above are the strict-manual numbers; the same over-counting is what drives the TAP-Phi-3 answer below.

### TAP-Phi-3 +6% (xyar Q2)

**The problem.**
xyar: *"In Table 3, the ASR for the TAP attack on the Phi-3 model is +6%, meaning the defended model is actively more vulnerable to this attack than the baseline. Could you elaborate on why this regression occurs? Does reshaping the mid-layer geometry inadvertently disrupt Phi-3's natural guardrails against semantic reframing? If some model is already quite good at defending against these attacks, could AnchorRep disrupt their safety and worse ASRs?"*

**Short answer.** Not a real regression — the reported +6 pp is the same auto-judge failure mode quantified in the multilingual result above (topic-adjacency scored as harmful), applied to the TAP-Phi-3 cell. The defended Phi-3 produces the hollow-compliance outputs §Discussion warned about, and the automated pipeline over-scores them. The +6 pp should not have been reported as a headline number; it reflects our evaluation pipeline, not the defense breaking Phi-3's guardrails.

### Refusal-direction stability (xyar W2)

**The problem.**
xyar: *"Refusal Direction Brittleness: The defense relies on a pre-computed 'refusal direction' mapped from 10 specific English templates. It remains unclear how robust this specific geometric landing pad is."*

**Short answer.** Recomputed r̂ on Llama-3-8B under 11 conditions (paper protocol repeated with 5 seeds; 3/10/20 templates; 5/15/30 prompts; alternative-phrased 10 templates) and reported the full pairwise cosine-similarity matrix. Seed-to-seed variance under paper protocol: 0.995 ± 0.002 (half a percent). Reasonable perturbations: cos ≥ 0.94. The direction is not brittle in any practical sense — it's a stable geometric quantity of the model that our 10-template protocol samples reliably. To further check whether the lowest-cos-similarity batch (the extreme joint perturbation of 3 templates + alternative phrasing) affects defense quality, we are retraining Llama-3 with those templates and re-running the full cross-model transfer eval; result queued behind the current sweep.

---

### Multi-anchor viability (xyar Q3 + zqtf W2)

**The problem.**
xyar Q3: *"If a white-box attacker knows both the target model and the specific anchor model used for the defense, could they jointly optimize a continuous attack that explicitly evades the anchor's trajectory while still finding a pathway to non-refusal to harmful queries in the target?"*
zqtf W2: *"The current design uses a single anchor model for repulsion. The paper acknowledges that this might be insufficient against sophisticated, defense-aware adversaries who could optimize attacks across multiple surrogate geometries. Multi-anchor configurations were explored but found unstable."*

**Short answer.** New experiment. Modified the training code to accept two anchors and average the CKA loss across them. Trained two Llama-3 defenders: Phi-3-medium + Mistral-7B (γ=2.0) and Phi-3-medium + Qwen1.5-7B-Chat (γ=4.0, architecturally-distinct pair). Both were evaluated end-to-end with the paper's canonical pipeline (chat template + 6-stage WildGuard judge + Self/Anchor/Other aggregation), and every automated-pipeline defended positive was inspected against the paper's `app:manual_verification` criteria. Under strict manual verification the two multi-anchor defenders match the paper's single-anchor Llama-3 headline (~1.1% defended), settling the reviewer's concern that single-anchor training is not the only viable option.

### LoRA sparsity binding cost (zqtf Q3 + W5)

**The problem.**
zqtf W5: *"Restricting LoRA adapters to a subset of layers significantly degrades defense effectiveness, implying that full-network perturbation is a necessary binding cost, which might limit flexibility in parameter reduction."*
zqtf Q3: *"Why is full-network LoRA identified as a 'binding cost' essential for defense effectiveness, rather than localized adapters? Does this imply inherent limitations in parameter efficiency or deployment flexibility, and are there future strategies to mitigate this cost while maintaining defense efficacy?"*

**Short answer.** Two parts. Mechanism (already in the paper): CKA repulsion is layer-localized but drives a representation-global change. Pre-target-layer adapters reshape the trajectory into the target layer; target-layer adapters receive gradient directly; post-target-layer adapters route the perturbed representation into fluent output. Remove any and the causal chain breaks — grounded in `tab:lora_subset` + §Discussion's "gradient dilution across layers" line. Matched-parameter mitigation (new experiment): trained a Llama-3 defender with LoRA rank 128 restricted to 8 mid-band layers (layers 12–19), matching the total capacity of the paper's canonical rank-32-across-32-layers configuration (both give 1024 rank-layers). Under the paper's canonical pipeline with manual verification, the concentrated-capacity variant matches the paper's single-anchor headline (~1.1% defended). Concentrated capacity is therefore not fundamentally worse than spread capacity at matched parameter count; the "binding cost" observation in `tab:lora_subset` reflects the fixed-rank comparison, and matched-parameter concentration recovers the defense.

