#!/usr/bin/env python3
"""
Layer Sweep Experiment: Finding Optimal Defense Layer

Sweeps target_layer_pct across multiple values (including the CKA-peak layer)
and compares defense quality (ASR, BRR, OR-Bench, MT-Bench) at each layer.

Usage:
    # Full sweep for one model pair
    python layer_sweep_experiment.py --anchor llama2 --defender vicuna --precision fp32

    # Just evaluate existing adapters (skip training)
    python layer_sweep_experiment.py --anchor llama2 --defender vicuna --eval-only

    # Custom layer percentiles
    python layer_sweep_experiment.py --anchor llama2 --defender vicuna \
        --layers 0.25 0.5 0.625 0.75

    # Resume (skips layers whose adapter already exists)
    python layer_sweep_experiment.py --anchor llama2 --defender vicuna --resume
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKA_DIR = os.path.join(SCRIPT_DIR, "..", "phase1", "all_layers_cka_output", "cross_layer_matrices")

DEFAULT_LAYERS = [0.25, 0.375, 0.5, 0.625, 0.75]


# ---------------------------------------------------------------------------
# CKA peak detection
# ---------------------------------------------------------------------------

def find_peak_diagonal_layer(anchor: str, defender: str, cka_dir: str = CKA_DIR) -> float:
    """Read the harmful cross-layer CKA matrix and return the layer_pct of the diagonal peak.

    Tries both orderings (anchor_vs_defender and defender_vs_anchor) since the
    matrix files may exist in either direction.
    """
    candidates = [
        os.path.join(cka_dir, f"cka_harm_{anchor}_vs_{defender}.csv"),
        os.path.join(cka_dir, f"cka_harm_{defender}_vs_{anchor}.csv"),
    ]

    df = None
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            break

    if df is None:
        print(f"[!] No CKA matrix found for {anchor} vs {defender}, tried:")
        for p in candidates:
            print(f"    {p}")
        print("[!] Falling back to default 0.5")
        return 0.5

    # Extract diagonal — the matrix may not be square when model depths differ,
    # so take min(rows, cols) diagonal elements.
    n = min(df.shape[0], df.shape[1])
    diag = np.array([df.iloc[i, i] for i in range(n)])

    peak_idx = int(np.argmax(diag))
    peak_pct = peak_idx / (n - 1) if n > 1 else 0.5

    print(f"[*] CKA diagonal peak: layer {peak_idx}/{n-1} = {peak_pct:.4f} "
          f"(CKA={diag[peak_idx]:.4f})")
    return round(peak_pct, 4)


def get_diagonal_values(anchor: str, defender: str, cka_dir: str = CKA_DIR) -> dict:
    """Return {layer_pct: cka_value} for the full diagonal."""
    candidates = [
        os.path.join(cka_dir, f"cka_harm_{anchor}_vs_{defender}.csv"),
        os.path.join(cka_dir, f"cka_harm_{defender}_vs_{anchor}.csv"),
    ]

    df = None
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            break

    if df is None:
        return {}

    n = min(df.shape[0], df.shape[1])
    result = {}
    for i in range(n):
        pct = round(i / (n - 1), 4) if n > 1 else 0.5
        result[pct] = float(df.iloc[i, i])
    return result


def lookup_cka_diag(layer_pct: float, diag_values: dict) -> float:
    """Find the closest diagonal CKA value for a given layer_pct."""
    if not diag_values:
        return float("nan")
    closest_pct = min(diag_values.keys(), key=lambda p: abs(p - layer_pct))
    return diag_values[closest_pct]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_layer(layer_pct: float, anchor: str, defender: str,
                output_dir: str, precision: str, extra_args: list) -> str:
    """Run two_stage_defense_v2.py for a single layer_pct. Returns adapter dir."""
    adapter_dir = os.path.join(output_dir, f"{defender}_layer{layer_pct}")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "two_stage_defense_v2.py"),
        "--anchor", anchor,
        "--defender", defender,
        "--alignment", "cka",
        "--target_layer_pct", str(layer_pct),
        "--gamma", "0.5",
        "--beta", "1.0",
        "--epsilon", "1.0",
        "--alpha", "0.15",
        "--delta", "0.06",
        "--zeta", "0",
        "--stage2_steps", "300",
        "--stage2_lr", "2e-4",
        "--precision", precision,
        "--output_dir", adapter_dir,
        "--cka_scope", "harmful_only",
        "--use_borderline",
        "--n_borderline", "200",
        "--borderline_source", "xstest",
    ] + extra_args

    print(f"\n{'='*70}")
    print(f"TRAINING layer_pct={layer_pct}")
    print(f"Output: {adapter_dir}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"[!] Training failed for layer_pct={layer_pct} (exit code {result.returncode})")
    return adapter_dir


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def find_adapter_path(adapter_dir: str) -> str:
    """Find the actual adapter sub-directory inside the output dir.

    two_stage_defense_v2.py creates a timestamped subdirectory like
    defender_v2_cka_20260210/. Return the first subdirectory containing
    adapter_config.json, or the dir itself.
    """
    # Check if adapter_config.json is directly in adapter_dir
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        return adapter_dir

    # Search one level deep
    if os.path.isdir(adapter_dir):
        for entry in sorted(os.listdir(adapter_dir)):
            subdir = os.path.join(adapter_dir, entry)
            if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "adapter_config.json")):
                return subdir

    return adapter_dir


def run_security_eval(adapter_dir: str, anchor: str, defender: str,
                      precision: str, result_path: str) -> dict:
    """Run evaluate_v2.py and return parsed JSON results."""
    adapter_path = find_adapter_path(adapter_dir)

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "evaluate_v2.py"),
        "--adapter_path", adapter_path,
        "--defender", defender,
        "--anchor", anchor,
        "--precision", precision,
        "--output_json", result_path,
    ]

    print(f"\n[*] Security eval: {adapter_path}")
    print(f"    Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"[!] Security eval failed (exit code {result.returncode})")
        return {}

    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return {}


def run_benchmark_eval(adapter_dir: str, defender: str,
                       precision: str, result_path: str) -> dict:
    """Run benchmark_eval.py and return parsed JSON results."""
    adapter_path = find_adapter_path(adapter_dir)

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "benchmark_eval.py"),
        "--defender", defender,
        "--adapter_path", adapter_path,
        "--precision", precision,
        "--output_json", result_path,
        "--skip_mmlu",
        "--no_baseline",
    ]

    print(f"\n[*] Benchmark eval: {adapter_path}")
    print(f"    Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"[!] Benchmark eval failed (exit code {result.returncode})")
        return {}

    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------

def extract_security_metrics(data: dict) -> dict:
    """Pull ASR/BRR/BGR from evaluate_v2.py JSON output."""
    defended = data.get("defended", {})
    return {
        "asr_self": defended.get("asr_self", float("nan")),
        "asr_anchor": defended.get("asr_anchor", float("nan")),
        "asr_other": defended.get("asr_other", float("nan")),
        "brr": defended.get("brr", float("nan")),
        "bgr": defended.get("bgr", float("nan")),
    }


def extract_benchmark_metrics(data: dict) -> dict:
    """Pull OR-Bench / MT-Bench from benchmark_eval.py JSON output."""
    defended = data.get("defended", {})
    return {
        "orbench_refusal": defended.get("orbench_refusal_rate", float("nan")),
        "mtbench_avg": defended.get("mtbench_score", float("nan")),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(layers: list, labels: dict, output_dir: str,
                  anchor: str, defender: str, diag_values: dict) -> pd.DataFrame:
    """Collect all result JSONs into a summary DataFrame."""
    rows = []
    for pct in layers:
        label = labels.get(pct, str(pct))
        adapter_dir = os.path.join(output_dir, f"{defender}_layer{pct}")

        sec_path = os.path.join(adapter_dir, "security_eval.json")
        bench_path = os.path.join(adapter_dir, "benchmark_eval.json")

        sec_metrics = {}
        bench_metrics = {}

        if os.path.exists(sec_path):
            with open(sec_path) as f:
                sec_metrics = extract_security_metrics(json.load(f))

        if os.path.exists(bench_path):
            with open(bench_path) as f:
                bench_metrics = extract_benchmark_metrics(json.load(f))

        row = {
            "layer_pct": pct,
            "label": label,
            "cka_diag": lookup_cka_diag(pct, diag_values),
        }
        row.update(sec_metrics)
        row.update(bench_metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """Pretty-print the summary table."""
    print("\n" + "=" * 100)
    print("LAYER SWEEP SUMMARY")
    print("=" * 100)

    # Format for display
    cols = ["layer_pct", "label", "cka_diag", "asr_self", "asr_anchor",
            "asr_other", "brr", "bgr", "orbench_refusal", "mtbench_avg"]
    available = [c for c in cols if c in df.columns]
    display = df[available].copy()

    # Format percentages
    pct_cols = ["asr_self", "asr_anchor", "asr_other", "brr", "bgr", "orbench_refusal"]
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) and not np.isnan(x) else "N/A"
            )

    for col in ["cka_diag", "mtbench_avg"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and not np.isnan(x) else "N/A"
            )

    display["layer_pct"] = display["layer_pct"].apply(lambda x: f"{x:.4f}")

    print(display.to_string(index=False))
    print()


def plot_results(df: pd.DataFrame, output_path: str, cka_peak_pct: float):
    """Generate a multi-panel plot of sweep results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = df["layer_pct"].values

    # Panel 1: ASR + BGR vs layer_pct
    ax = axes[0]
    for col, label, marker in [("asr_self", "ASR (self)", "o-"),
                                ("asr_anchor", "ASR (anchor)", "o-"),
                                ("asr_other", "ASR (other)", "o-"),
                                ("bgr", "BGR (garble)", "x--")]:
        if col in df.columns:
            vals = df[col].values
            mask = ~np.isnan(vals)
            if mask.any():
                ax.plot(x[mask], vals[mask] * 100, marker, label=label)
    ax.axvline(cka_peak_pct, color="red", linestyle="--", alpha=0.6, label=f"CKA peak ({cka_peak_pct:.3f})")
    ax.set_xlabel("Target Layer %")
    ax.set_ylabel("Rate (%)")
    ax.set_title("ASR & Garble Rate (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: OR-Bench refusal rate
    ax = axes[1]
    if "orbench_refusal" in df.columns:
        vals = df["orbench_refusal"].values
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(x[mask], vals[mask] * 100, "s-", color="orange", label="OR-Bench refusal")
    ax.axvline(cka_peak_pct, color="red", linestyle="--", alpha=0.6, label=f"CKA peak ({cka_peak_pct:.3f})")
    ax.set_xlabel("Target Layer %")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("OR-Bench Refusal (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: MT-Bench score
    ax = axes[2]
    if "mtbench_avg" in df.columns:
        vals = df["mtbench_avg"].values
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(x[mask], vals[mask], "D-", color="green", label="MT-Bench avg")
    ax.axvline(cka_peak_pct, color="red", linestyle="--", alpha=0.6, label=f"CKA peak ({cka_peak_pct:.3f})")
    ax.set_xlabel("Target Layer %")
    ax.set_ylabel("Score (1-10)")
    ax.set_title("MT-Bench (higher is better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Layer Sweep Results", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer Sweep Experiment: sweep target_layer_pct to find optimal defense layer")

    parser.add_argument("--anchor", type=str, required=True, help="Anchor model name (e.g. llama2)")
    parser.add_argument("--defender", type=str, required=True, help="Defender model name (e.g. vicuna)")
    parser.add_argument("--layers", type=float, nargs="+", default=None,
                        help="Custom layer percentiles to sweep (default: 0.25 0.375 0.5 0.625 0.75 + CKA peak)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["4bit", "fp16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="./layer_sweep_outputs",
                        help="Base output directory")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--resume", action="store_true",
                        help="Skip layers whose adapter/results already exist")
    parser.add_argument("--skip_cka_peak", action="store_true",
                        help="Don't add the CKA-peak layer to the sweep")
    parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args passed through to two_stage_defense_v2.py")

    args = parser.parse_args()

    anchor = args.anchor.lower()
    defender = args.defender.lower()
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # --- Determine sweep layers ---
    cka_peak_pct = find_peak_diagonal_layer(anchor, defender)
    diag_values = get_diagonal_values(anchor, defender)

    if args.layers is not None:
        layers = sorted(set(args.layers))
    else:
        layers = sorted(set(DEFAULT_LAYERS))

    # Add CKA peak if not already present and not skipped
    if not args.skip_cka_peak and cka_peak_pct not in layers:
        layers.append(cka_peak_pct)
        layers.sort()

    # Build labels
    labels = {}
    for pct in layers:
        if pct == cka_peak_pct:
            labels[pct] = f"cka_peak ({pct:.4f})"
        else:
            labels[pct] = str(pct)

    print(f"\n[*] Sweep layers: {layers}")
    print(f"[*] CKA peak layer: {cka_peak_pct}")
    print(f"[*] Output dir: {output_dir}")

    # --- Training phase ---
    if not args.eval_only:
        for pct in layers:
            adapter_dir = os.path.join(output_dir, f"{defender}_layer{pct}")
            if args.resume and os.path.isdir(adapter_dir) and find_adapter_path(adapter_dir) != adapter_dir:
                print(f"\n[*] Skipping training for layer_pct={pct} (adapter exists: {adapter_dir})")
                continue
            train_layer(pct, anchor, defender, output_dir, args.precision, args.extra_train_args)

    # --- Evaluation phase ---
    for pct in layers:
        adapter_dir = os.path.join(output_dir, f"{defender}_layer{pct}")
        sec_path = os.path.join(adapter_dir, "security_eval.json")
        bench_path = os.path.join(adapter_dir, "benchmark_eval.json")

        if not os.path.isdir(adapter_dir):
            print(f"\n[!] No adapter dir for layer_pct={pct}: {adapter_dir}")
            continue

        # Security eval
        if args.resume and os.path.exists(sec_path):
            print(f"[*] Skipping security eval for layer_pct={pct} (exists)")
        else:
            run_security_eval(adapter_dir, anchor, defender, args.precision, sec_path)

        # Benchmark eval
        if args.resume and os.path.exists(bench_path):
            print(f"[*] Skipping benchmark eval for layer_pct={pct} (exists)")
        else:
            run_benchmark_eval(adapter_dir, defender, args.precision, bench_path)

    # --- Summary ---
    summary_df = build_summary(layers, labels, output_dir, anchor, defender, diag_values)
    print_summary(summary_df)

    # Save CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[+] Summary CSV saved to {csv_path}")

    # Save plot
    plot_path = os.path.join(output_dir, "layer_sweep_results.png")
    plot_results(summary_df, plot_path, cka_peak_pct)


if __name__ == "__main__":
    main()
