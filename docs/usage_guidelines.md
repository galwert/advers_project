# Usage guidelines

This release is intended for defensive security research on open-weight LLMs.

## What's appropriate

- Reproducing or extending the AnchorRep results.
- Studying cross-model jailbreak transfer in academic or defensive industry contexts.
- Developing improved defense mechanisms or evaluation methodologies that build on the released artifacts.
- Stress-testing the defended adapters against new attack methods to find weaknesses, then publishing the findings responsibly.

## What's not

- Deploying the adapters to production without independent safety validation. The defense is designed and evaluated as a research artifact.
- Using the released GCG suffixes to attack third-party deployed systems.
- Removing the LoRA delta to obtain a base model with reduced safety. The base model is already publicly available, so doing this provides no offensive uplift, but the released artifact should not be marketed as a tool for that purpose.

## Dual-use note on attack artifacts

The GCG suffixes in `attack_artifacts/` are derived from a publicly known attack method (Zou et al. 2023) applied to publicly available prompts (AdvBench, HarmBench). They are released to enable reproducibility of the paper's evaluations. Their inclusion is consistent with the practice of the original llm-attacks repository and does not introduce new offensive capability beyond what is already obtainable from public methods. See `attack_artifacts/README.md` for further details.

## Reporting issues

If you find a serious safety failure mode in the defended adapters (for example, a class of attacks that bypasses the defense across all five defenders with high success rate), the responsible disclosure path is to open a discussion on the paper's OpenReview page (after the anonymous review period concludes) rather than publishing the attack standalone.

## Limitations recap

- Embedding-space PGD bypasses the defense; this is documented in the paper.
- TAP (semantic reframing) partially bypasses on Llama-3.
- Single-anchor design; an adversary jointly optimizing across multiple surrogates could partially circumvent the defense.
- Performance under non-English prompts and refusal templates has not been evaluated.

For full discussion, see the paper Limitations section.
