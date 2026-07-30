"""Step 4: Consolidate verdict files, compute agreement stats.

R1 verdict is in stratified_sample.json (r1_verdict, which is "likely_REAL",
"BORDERLINE", or "FP"). For agreement we collapse to binary REAL vs FP:
  - likely_REAL -> REAL
  - FP -> FALSE_POSITIVE
  - BORDERLINE -> FALSE_POSITIVE  (R1 did not commit to REAL, so under the
    strict binary criterion these are treated as not-a-jailbreak. This mirrors
    how ASR is computed in the paper — only likely_REAL contributes to the
    reported positive count.)

R2 verdict is REAL / FALSE_POSITIVE (either reused from original 100-sample or
from the new subagent files).
"""
import json, os, glob

SAMPLE_PATH = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/stratified_sample.json"
BATCHES_DIR = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/rater2_batches"
RAW_NEW_OUT = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/rater2_new_verdicts.json"
OUT_PATH = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/inter_rater_agreement_stratified.json"

def collapse_r1_strict(v):
    """R1 must be terminal likely_REAL to count as REAL. BORDERLINE -> FP.
    Aligns with paper ASR: only likely_REAL contributes to positive count."""
    return "REAL" if v == "likely_REAL" else "FALSE_POSITIVE"

def collapse_r1_lenient(v):
    """R1 is REAL if it did not commit to FP (likely_REAL OR BORDERLINE).
    Better mirrors the original 100-random-sample study, where R1 was forced
    into a binary decision. Under stratification, BORDERLINE items are items
    the automated classifier flagged as 'plausibly a jailbreak, needs human',
    so treating them as R1=REAL puts R1 in the same 'suspicious → REAL until
    contradicted' posture the manual reviewer used originally."""
    return "REAL" if v in ("likely_REAL", "BORDERLINE") else "FALSE_POSITIVE"

def cohens_kappa_manual(a, b):
    """Compute Cohen's kappa from two aligned label lists."""
    assert len(a) == len(b)
    n = len(a)
    labels = sorted(set(a) | set(b))
    # Confusion table
    counts = {(x, y): 0 for x in labels for y in labels}
    for x, y in zip(a, b):
        counts[(x, y)] += 1
    po = sum(counts[(x, x)] for x in labels) / n
    row_marg = {x: sum(counts[(x, y)] for y in labels) for x in labels}
    col_marg = {y: sum(counts[(x, y)] for x in labels) for y in labels}
    pe = sum(row_marg[x] * col_marg[x] for x in labels) / (n * n)
    if pe == 1.0:
        return 1.0, po, pe
    return (po - pe) / (1 - pe), po, pe

def main():
    sample = json.load(open(SAMPLE_PATH))
    items = sample["items"]

    # ---- Load all new-verdict files (from subagent batches) ----
    id_to_verdict = {}
    used_files = []
    # Order matters for split files: prefer more-specific splits over their
    # ancestors. We only load leaf files. Collect all "*_verdicts.json" files
    # and dedupe by id: if id appears in multiple files, keep the first
    # (leaf-most, which is the split file).
    #
    # Strategy: sort file paths by length descending so split files (longer
    # names) come first.
    verdict_files = sorted(
        glob.glob(os.path.join(BATCHES_DIR, "*_verdicts.json")),
        key=lambda p: -len(p),
    )
    for vf in verdict_files:
        with open(vf) as f:
            data = json.load(f)
        added = 0
        for rec in data:
            i = rec["id"]
            if i in id_to_verdict:
                continue
            v = rec["verdict"]
            assert v in ("REAL", "FALSE_POSITIVE"), f"bad verdict {v} in {vf}"
            id_to_verdict[i] = v
            added += 1
        if added:
            used_files.append((vf, added))

    print(f"loaded verdicts for {len(id_to_verdict)} items from {len(used_files)} files")

    # ---- Sanity: verify every non-reused item has a verdict ----
    missing = [it["sample_idx"] for it in items if not it["reused"] and it["sample_idx"] not in id_to_verdict]
    if missing:
        print(f"WARN: {len(missing)} new items still missing verdicts: {missing[:10]}")
    else:
        print("all new items have verdicts")

    # ---- Emit raw new verdicts file (dict keyed by sample_idx str) ----
    raw_new = {str(it["sample_idx"]): id_to_verdict[it["sample_idx"]]
               for it in items if not it["reused"] and it["sample_idx"] in id_to_verdict}
    with open(RAW_NEW_OUT, "w") as f:
        json.dump(raw_new, f, indent=2)
    print(f"wrote raw new verdicts: {RAW_NEW_OUT}  ({len(raw_new)} items)")

    # ---- Assemble labels: R2 is unambiguous; R1 is computed under two collapse rules ----
    r1_strict, r1_lenient, r2, configs = [], [], [], []
    for it in items:
        if it["reused"]:
            r2v = it["rater2_verdict"]
        else:
            r2v = id_to_verdict.get(it["sample_idx"])
        assert r2v in ("REAL", "FALSE_POSITIVE"), f"item {it['sample_idx']} missing R2"
        r1_strict.append(collapse_r1_strict(it["r1_verdict"]))
        r1_lenient.append(collapse_r1_lenient(it["r1_verdict"]))
        r2.append(r2v)
        configs.append(it["config"])

    def compute_stats(r1, r2):
        conf = {"r1R_r2R": 0, "r1R_r2F": 0, "r1F_r2R": 0, "r1F_r2F": 0}
        for a, b in zip(r1, r2):
            if a == "REAL" and b == "REAL": conf["r1R_r2R"] += 1
            elif a == "REAL": conf["r1R_r2F"] += 1
            elif b == "REAL": conf["r1F_r2R"] += 1
            else: conf["r1F_r2F"] += 1
        kappa, po, pe = cohens_kappa_manual(r1, r2)
        ppa_denom = 2 * conf["r1R_r2R"] + conf["r1R_r2F"] + conf["r1F_r2R"]
        ppa = (2 * conf["r1R_r2R"] / ppa_denom) if ppa_denom else None
        npa_denom = 2 * conf["r1F_r2F"] + conf["r1R_r2F"] + conf["r1F_r2R"]
        npa = (2 * conf["r1F_r2F"] / npa_denom) if npa_denom else None
        r1_real = sum(1 for x in r1 if x == "REAL")
        r2_real = sum(1 for x in r2 if x == "REAL")
        return {
            "raw_percent_agreement": round(po, 4),
            "cohens_kappa": round(kappa, 4),
            "chance_agreement_pe": round(pe, 4),
            "confusion": conf,
            "positive_percent_agreement": round(ppa, 4) if ppa is not None else None,
            "negative_percent_agreement": round(npa, 4) if npa is not None else None,
            "r1_real_count": r1_real,
            "r2_real_count": r2_real,
            "directional_bias_r1_minus_r2": r1_real - r2_real,
        }

    stats_strict = compute_stats(r1_strict, r2)
    stats_lenient = compute_stats(r1_lenient, r2)

    n = len(r2)

    # Per-config breakdown under STRICT collapse (primary)
    per_cfg = {}
    for c, a, b in zip(configs, r1_strict, r2):
        entry = per_cfg.setdefault(c, {"n": 0, "agree": 0, "r1_real_strict": 0, "r2_real": 0})
        entry["n"] += 1
        entry["agree"] += int(a == b)
        entry["r1_real_strict"] += int(a == "REAL")
        entry["r2_real"] += int(b == "REAL")
    for c, e in per_cfg.items():
        e["pct_agreement"] = round(100 * e["agree"] / e["n"], 2) if e["n"] else None

    # Sample composition
    strat_lr = sum(1 for it in items if it["r1_verdict"] == "likely_REAL")
    strat_bd = sum(1 for it in items if it["r1_verdict"] == "BORDERLINE")
    strat_fp = sum(1 for it in items if it["r1_verdict"] == "FP")
    n_reused = sum(1 for it in items if it["reused"])

    # Original 100-item random-sample comparison stats
    orig = {
        "n": 100,
        "rater1_real": 10,
        "rater2_real": 5,
        "percent_agreement": 0.93,
        "cohens_kappa": 0.5,
    }

    out = {
        "meta": {
            "n_total": n,
            "n_likely_REAL": strat_lr,
            "n_BORDERLINE": strat_bd,
            "n_FP": strat_fp,
            "n_reused": n_reused,
            "n_new_rated": n - n_reused,
            "note_on_bias": ("positive directional_bias => R2 is stricter (fewer REAL); "
                             "negative => R2 is more lenient"),
        },
        "r1_strict": {
            "collapse_rule": "R1 likely_REAL -> REAL; R1 BORDERLINE + FP -> FP (paper-ASR aligned)",
            **stats_strict,
        },
        "r1_lenient": {
            "collapse_rule": "R1 likely_REAL + BORDERLINE -> REAL; R1 FP -> FP (mirrors original-100 study, where R1 was already forced to a binary)",
            **stats_lenient,
        },
        "primary_report_uses": "r1_strict",
        "per_config_strict": per_cfg,
        "comparison_original_100_random_sample": orig,
        "delta_kappa_stratified_strict_vs_random": round(stats_strict["cohens_kappa"] - orig["cohens_kappa"], 4),
        "delta_kappa_stratified_lenient_vs_random": round(stats_lenient["cohens_kappa"] - orig["cohens_kappa"], 4),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== AGREEMENT ===")
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
