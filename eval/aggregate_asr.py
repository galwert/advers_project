#!/usr/bin/env python3
"""Aggregate Self / Anchor / Other / Transfer ASR from a judged-pipeline CSV.

Reads the output of `eval/judge_pipeline.py` (column `is_jailbroken`) and
breaks the verified ASR down by attack-source role for one defender.

Outputs a small JSON next to the input CSV (suffix `_summary.json`) and
prints a paper-comparable table to stdout.

Usage:
    python eval/aggregate_asr.py \
        --input  logs/cross_model_transfer/mistral_repro/defended_judged.csv \
        --defender mistral
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


# Per-defender attack source labels in the GCG corpus + canonical anchor pick.
# The anchor name must exactly match the `source_model` column produced by
# eval/cross_model_transfer.py (which is whatever appears in the suffixes CSV).
DEFENDER_INFO = {
    "llama3":  {"self": "Llama3-8b",  "anchor": "Phi-3-medium"},
    "mistral": {"self": "Mistral-7b", "anchor": "Qwen1.5-7b"},
    "vicuna":  {"self": "Vicuna-7b",  "anchor": "Qwen1.5-7b"},
    "qwen14b": {"self": "Qwen1.5-14b","anchor": "Llama3-8b"},
    "phi3":    {"self": "Phi-3-medium","anchor": "Llama3-8b"},
}


def is_jb(row) -> bool:
    """Map the various truthy spellings of `is_jailbroken` to a bool."""
    v = (row.get("is_jailbroken") or row.get("jailbroken") or row.get("verdict") or "").strip().lower()
    return v in ("true", "1", "yes", "unsafe", "harmful")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Scored CSV from judge_pipeline.py")
    ap.add_argument("--defender", required=True, choices=list(DEFENDER_INFO.keys()))
    ap.add_argument("--summary", default=None, help="Output JSON summary (default: alongside input)")
    args = ap.parse_args()

    info = DEFENDER_INFO[args.defender]
    self_src, anchor_src = info["self"], info["anchor"]

    groups = defaultdict(list)
    sources = defaultdict(int)
    with open(args.input) as f:
        reader = csv.DictReader(f)
        # Backwards-compat: older runs of cross_model_transfer.py wrote a single
        # "model" column that holds the source-of-attack identifier (no separate
        # source_model / target_model columns). Detect that case and read from
        # "model" instead.
        cols = reader.fieldnames or []
        has_source = "source_model" in cols
        has_legacy_model = (not has_source) and ("model" in cols)
        if not has_source and not has_legacy_model:
            print(f"[error] judged CSV has neither 'source_model' nor 'model' column. "
                  f"Columns seen: {cols}", file=sys.stderr)
            return 2
        if has_legacy_model:
            print(f"[*] No 'source_model' column found; falling back to 'model' "
                  f"column (older cross_model_transfer.py output format).")
        for row in reader:
            src = (row.get("source_model") if has_source else row.get("model", "")).strip()
            sources[src] += 1
            jb = is_jb(row)
            if src == self_src:
                groups["self"].append(jb)
            elif src == anchor_src:
                groups["anchor"].append(jb)
            else:
                groups["other"].append(jb)

    def pct(g):
        rows = groups[g]
        return (sum(rows) / len(rows) * 100) if rows else 0.0, sum(rows), len(rows)

    self_pct, self_jb, self_n     = pct("self")
    anchor_pct, anchor_jb, anchor_n = pct("anchor")
    other_pct, other_jb, other_n   = pct("other")

    transfer_jb = anchor_jb + other_jb
    transfer_n  = anchor_n + other_n
    transfer_pct = (transfer_jb / transfer_n * 100) if transfer_n else 0.0

    total_jb = self_jb + transfer_jb
    total_n  = self_n + transfer_n
    total_pct = (total_jb / total_n * 100) if total_n else 0.0

    print()
    print("=" * 60)
    print(f"  Verified ASR — defender: {args.defender}")
    print(f"  Self source: {self_src}, Anchor source: {anchor_src}")
    print("=" * 60)
    print(f"  Self     : {self_jb:>4} / {self_n:<4} = {self_pct:5.2f}%")
    print(f"  Anchor   : {anchor_jb:>4} / {anchor_n:<4} = {anchor_pct:5.2f}%")
    print(f"  Other    : {other_jb:>4} / {other_n:<4} = {other_pct:5.2f}%")
    print(f"  Transfer : {transfer_jb:>4} / {transfer_n:<4} = {transfer_pct:5.2f}%   (Anchor + Other)")
    print(f"  Total    : {total_jb:>4} / {total_n:<4} = {total_pct:5.2f}%   (Self + Transfer)")
    print("=" * 60)
    if not groups["self"]:
        print(f"  [warn] no rows matched Self source ({self_src!r}). Sources seen: {sorted(sources.keys())}")
    if not groups["anchor"]:
        print(f"  [warn] no rows matched Anchor source ({anchor_src!r}).")

    out = args.summary or os.path.splitext(args.input)[0] + "_summary.json"
    with open(out, "w") as f:
        json.dump({
            "defender": args.defender,
            "self_source": self_src,
            "anchor_source": anchor_src,
            "self":     {"jb": self_jb,    "n": self_n,    "asr": self_pct},
            "anchor":   {"jb": anchor_jb,  "n": anchor_n,  "asr": anchor_pct},
            "other":    {"jb": other_jb,   "n": other_n,   "asr": other_pct},
            "transfer": {"jb": transfer_jb,"n": transfer_n,"asr": transfer_pct},
            "total":    {"jb": total_jb,   "n": total_n,   "asr": total_pct},
        }, f, indent=2)
    print(f"  [+] Summary written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
