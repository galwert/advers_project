#!/usr/bin/env python3
"""Run evaluate_v2 multiple times with different random seeds to estimate variance.
Reports mean +/- std for ASR, BGR, BRR, PPL across seeds."""

import argparse, json, os, subprocess, sys, statistics

def run_eval(seed, defender, adapter_path, anchor, precision, output_prefix):
    """Run evaluate_v2.py with a specific seed and return the results dict."""
    out_json = f"{output_prefix}_seed{seed}.json"
    cmd = [
        sys.executable, "evaluate_v2.py",
        "--adapter_path", adapter_path,
        "--defender", defender,
        "--anchor", anchor,
        "--precision", precision,
        "--cka_per_group",
        "--seed", str(seed),
        "--output_json", out_json,
    ]
    print(f"\n[Seed {seed}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[Seed {seed}] WARNING: eval returned {result.returncode}")
        return None

    if os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defender", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_prefix = args.output.replace(".json", "")
    all_results = []

    for seed in args.seeds:
        r = run_eval(seed, args.defender, args.adapter_path, args.anchor, args.precision, output_prefix)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("ERROR: No successful runs")
        sys.exit(1)

    # Aggregate
    metrics = {}
    for key in ["asr_self", "asr_anchor", "asr_other", "bgr", "brr", "ppl"]:
        vals = [r["defended"].get(key, 0) for r in all_results if "defended" in r]
        if vals:
            metrics[key] = {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "values": vals,
            }

    summary = {
        "n_seeds": len(all_results),
        "seeds": args.seeds[:len(all_results)],
        "metrics": metrics,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== VARIANCE SUMMARY ({len(all_results)} seeds) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}  (range: {v['min']:.4f} - {v['max']:.4f})")
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
