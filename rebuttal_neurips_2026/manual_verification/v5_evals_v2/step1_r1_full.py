"""Step 1: Apply R1 (two-pass classifier) to ALL defended positives.

R1 = manual_verify_all.score (pass1) THEN manual_verify_pass2.score2 (pass2)
Only invoke pass2 if pass1 returns BORDERLINE. Terminal verdicts from pass1:
  - FP -> FP (reason from pass1)
  - BORDERLINE -> feed into pass2, which yields FP / likely_REAL / BORDERLINE
  - likely_REAL is not produced by pass1 (it returns FP or BORDERLINE only)
"""
import json, os, sys

sys.path.insert(0, "/home/wertheizer/advers_project/experiments")
from manual_verify_all import score as score1
from manual_verify_pass2 import score2 as score2_fn

IN_PATH = "/home/wertheizer/advers_project/rebuttal_data/manual_review_workfiles/all_defended_positives.json"
OUT_PATH = "/home/wertheizer/advers_project/rebuttal_data/v5_evals_v2/r1_full_verdicts.json"

def r1(prompt, response):
    v1, reason1 = score1(response, prompt)
    if v1 == "FP":
        return "FP", reason1
    # v1 must be BORDERLINE — run pass2
    v2, reason2 = score2_fn(response, prompt, reason1)
    return v2, reason2

def main():
    with open(IN_PATH) as f:
        data = json.load(f)
    out = {}
    totals = {"FP": 0, "BORDERLINE": 0, "likely_REAL": 0}
    print(f"{'config':40s}  total   FP  BORD  likely_REAL")
    for cfg, items in data.items():
        cfg_out = []
        counts = {"FP": 0, "BORDERLINE": 0, "likely_REAL": 0}
        for idx, it in enumerate(items):
            v, reason = r1(it["prompt"], it["response"])
            cfg_out.append({
                "idx": idx,
                "prompt": it["prompt"],
                "response": it["response"],
                "source_model": it.get("source_model", ""),
                "r1_verdict": v,
                "r1_reason": reason,
            })
            counts[v] += 1
            totals[v] += 1
        out[cfg] = cfg_out
        print(f"{cfg:40s}  {len(items):5d}  {counts['FP']:4d}  {counts['BORDERLINE']:4d}  {counts['likely_REAL']:4d}")
    print(f"{'GRAND TOTAL':40s}  {sum(len(v) for v in data.values()):5d}  {totals['FP']:4d}  {totals['BORDERLINE']:4d}  {totals['likely_REAL']:4d}")
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved: {OUT_PATH}")

    # Also print a couple of examples per verdict for sanity
    print("\n--- SAMPLES ---")
    for target in ["likely_REAL", "BORDERLINE", "FP"]:
        for cfg, items in out.items():
            picks = [it for it in items if it["r1_verdict"] == target][:1]
            for it in picks:
                print(f"[{target}] cfg={cfg} idx={it['idx']} reason={it['r1_reason']}")
                print(f"  prompt: {it['prompt'][:120]}...")
                print(f"  response[:200]: {it['response'][:200]}...")
                break
            if picks:
                break

if __name__ == "__main__":
    main()
