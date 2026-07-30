"""Step 2: Build stratified sample.

- Include ALL likely_REAL (17 items)
- Include a random subset of BORDERLINE to bring total to ~200 (with seed=17)
- Reuse verdicts for items overlapping the original 100-item random sample
"""
import json, random

R1_PATH = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/r1_full_verdicts.json"
OLD_SAMPLE = "/home/wertheizer/advers_project/rebuttal_data/v5_evals/rater2_sample.json"
OLD_VERDICTS = "/home/wertheizer/advers_project/rebuttal_data/v5_evals/rater2_verdicts.json"
OLD_LINKAGE = "/home/wertheizer/advers_project/rebuttal_data/v5_evals/rater2_linkage.json"

OUT_PATH = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/stratified_sample.json"

TARGET_N = 200
SEED = 17

def main():
    r1 = json.load(open(R1_PATH))
    old_sample = json.load(open(OLD_SAMPLE))
    old_verdicts = json.load(open(OLD_VERDICTS))
    old_linkage = json.load(open(OLD_LINKAGE))

    # Build (cfg, pool_idx) -> R1 record
    pool = {}   # (cfg, idx) -> record
    for cfg, items in r1.items():
        for it in items:
            pool[(cfg, it["idx"])] = it

    # Build response-key lookup to map old sample items to (cfg, idx)
    resp_to_ci = {}
    for cfg, items in r1.items():
        for it in items:
            k = (it["prompt"][:200], it["response"][:200])
            resp_to_ci.setdefault(k, []).append((cfg, it["idx"]))

    # Determine which pool items are already rated
    old_rated = {}   # (cfg, idx) -> {"verdict": ..., "orig_idx": int}
    for s_it in old_sample:
        k = (s_it["prompt"][:200], s_it["response"][:200])
        hits = resp_to_ci.get(k, [])
        expected_cfg = old_linkage[str(s_it["idx"])]
        chosen = None
        for c, i in hits:
            if c == expected_cfg:
                chosen = (c, i)
                break
        if chosen is None and hits:
            chosen = hits[0]
        if chosen is None:
            print(f"WARN: no pool match for old sample idx={s_it['idx']}")
            continue
        old_rated[chosen] = {
            "verdict": old_verdicts[str(s_it["idx"])],
            "orig_idx": s_it["idx"],
        }
    print(f"Old rated items mapped into pool: {len(old_rated)} / {len(old_sample)}")

    # Bucket by R1 verdict
    likely_real = [(c, i) for (c, i), it in pool.items() if it["r1_verdict"] == "likely_REAL"]
    borderline = [(c, i) for (c, i), it in pool.items() if it["r1_verdict"] == "BORDERLINE"]
    fp = [(c, i) for (c, i), it in pool.items() if it["r1_verdict"] == "FP"]
    print(f"pool sizes: likely_REAL={len(likely_real)}  BORDERLINE={len(borderline)}  FP={len(fp)}")

    # Old-rated items broken down by R1 verdict
    old_lr = [k for k in old_rated if pool[k]["r1_verdict"] == "likely_REAL"]
    old_bd = [k for k in old_rated if pool[k]["r1_verdict"] == "BORDERLINE"]
    old_fp = [k for k in old_rated if pool[k]["r1_verdict"] == "FP"]
    print(f"  of which already rated: LR={len(old_lr)}  BD={len(old_bd)}  FP={len(old_fp)}")

    # Case: LR + BD > 220 (348 here). Take all LR + subset of BD to reach TARGET_N.
    # Ensure all old-rated BD are included.
    lr_total = len(likely_real)
    bd_needed = TARGET_N - lr_total  # 200 - 17 = 183
    if bd_needed < 0:
        bd_needed = 0

    rng = random.Random(SEED)
    # start with old-rated BD (guaranteed to include), then randomly fill up to bd_needed
    old_bd_set = set(old_bd)
    remaining_bd = [k for k in borderline if k not in old_bd_set]
    rng.shuffle(remaining_bd)
    take_random = max(0, bd_needed - len(old_bd_set))
    bd_selected = list(old_bd_set) + remaining_bd[:take_random]
    # Sort for determinism of the output
    bd_selected.sort()

    # Selected likely_REAL: always all of them (already includes any old-rated LR)
    lr_selected = sorted(likely_real)

    # We must also preserve any old-rated FP items that fell outside strat sample?
    # Spec: "reuse verdicts for items overlapping the original 100-item random sample"
    # The stratified sample is defined as LR + BD subset + (spec-optional FP). Old-rated
    # FP items outside stratum are simply not part of the new sample. We treat overlap
    # only within the strat sample.

    selected = lr_selected + bd_selected

    # Build the sample records
    out = []
    reused = 0
    new_items = 0
    for (cfg, idx) in selected:
        rec = pool[(cfg, idx)]
        entry = {
            "sample_idx": len(out),
            "config": cfg,
            "pool_idx": idx,
            "prompt": rec["prompt"],
            "response": rec["response"],
            "source_model": rec["source_model"],
            "r1_verdict": rec["r1_verdict"],
            "r1_reason": rec["r1_reason"],
        }
        if (cfg, idx) in old_rated:
            entry["reused"] = True
            entry["rater2_verdict"] = old_rated[(cfg, idx)]["verdict"]
            entry["reused_from_orig_idx"] = old_rated[(cfg, idx)]["orig_idx"]
            reused += 1
        else:
            entry["reused"] = False
            new_items += 1
        out.append(entry)

    print(f"\nStratified sample composition:")
    print(f"  n_likely_REAL: {len(lr_selected)}")
    print(f"  n_BORDERLINE:  {len(bd_selected)}")
    print(f"  n_FP:          0")
    print(f"  total:         {len(out)}")
    print(f"  reused (from orig 100): {reused}")
    print(f"  new items to rate:      {new_items}")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "meta": {
                "seed": SEED,
                "target_n": TARGET_N,
                "n_likely_REAL": len(lr_selected),
                "n_BORDERLINE": len(bd_selected),
                "n_FP": 0,
                "n_reused": reused,
                "n_new": new_items,
            },
            "items": out,
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
