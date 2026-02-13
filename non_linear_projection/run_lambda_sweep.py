"""
Lambda Sweep for Non-Linear Projection Defense

Sweeps lambda from 0.1 to 0.9 using pre-trained projectors, trains a LoRA
defense adapter for each, evaluates (harmful ASR via llm_judge + benign BRR/PPL/TDR),
and picks the best lambda.

Each step runs as a subprocess to guarantee GPU memory cleanup.

Usage:
    # Full run (all 9 lambdas)
    python run_lambda_sweep.py

    # Single lambda
    python run_lambda_sweep.py --lambdas 0.3

    # Custom hyperparams
    python run_lambda_sweep.py --alpha 1.0 --beta 0.01 --gamma 0.5 --delta 0.1

    # Dry run
    python run_lambda_sweep.py --dry_run
"""

import subprocess
import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List


# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_defense_v2.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "evaluate_defense.py")
PROJECTOR_DIR = os.path.join(SCRIPT_DIR, "models_with_different_lambdas")

DEFAULT_LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Default training hyperparameters
DEFAULT_HYPERPARAMS = {
    "alpha": 1.0,
    "beta": 0.01,
    "gamma": 0.5,
    "delta": 0.1,
    "train_steps": 250,
    "lr": 5e-5,
    "batch_size": 4,
}

# Models
DEFENDED_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEFENDED_TYPE = "llama2"
ANCHOR_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ANCHOR_TYPE = "llama3"
PROJECTOR_LAYER = 16

# CSV columns
CSV_COLUMNS = [
    "lambda", "projector_path", "adapter_path",
    "asr_self", "asr_anchor", "asr_other",
    "brr", "ppl", "tdr", "cka_score",
    "training_time_sec", "eval_time_sec",
    "status",
]


# ==========================================
# UTILITIES
# ==========================================
def load_completed_lambdas(csv_path: str, retry_failed: bool = False) -> set:
    """Load already-completed lambdas from CSV."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "")
            lam = row.get("lambda", "")
            if status == "OK":
                completed.add(lam)
            elif not retry_failed and status:
                completed.add(lam)
    return completed


def init_csv(csv_path: str):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_row(csv_path: str, row: Dict[str, Any]):
    """Append a single row to the CSV."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def run_command(cmd: list, description: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output in real-time."""
    print(f"\n{'='*60}", flush=True)
    print(f"[CMD] {description}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=SCRIPT_DIR,
        env=env,
        bufsize=1,
    )

    stdout_lines = []
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
            stdout_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise

    return subprocess.CompletedProcess(
        cmd, proc.returncode,
        stdout=''.join(stdout_lines),
        stderr='',
    )


# ==========================================
# CORE: TRAIN + EVALUATE ONE LAMBDA
# ==========================================
def run_lambda(
    lam: float,
    csv_path: str,
    results_dir: str,
    hyperparams: dict,
    gcg_data_path: str,
    dry_run: bool = False,
):
    """Train and evaluate defense for one lambda value."""
    lam_str = f"{lam:.1f}"
    projector_path = os.path.join(PROJECTOR_DIR, f"projector_lam_{lam_str}.pt")
    lambda_dir = os.path.join(results_dir, f"lambda_{lam_str}")

    print(f"\n{'#'*70}")
    print(f"# LAMBDA = {lam_str}")
    print(f"# Projector: {projector_path}")
    print(f"{'#'*70}")

    if not os.path.exists(projector_path):
        print(f"[!] Projector not found: {projector_path}")
        row = {col: "" for col in CSV_COLUMNS}
        row["lambda"] = lam_str
        row["projector_path"] = projector_path
        row["status"] = "PROJECTOR_NOT_FOUND"
        append_row(csv_path, row)
        return

    row = {col: "" for col in CSV_COLUMNS}
    row["lambda"] = lam_str
    row["projector_path"] = projector_path

    os.makedirs(lambda_dir, exist_ok=True)
    adapter_output_dir = lambda_dir

    if dry_run:
        train_cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--projector_path", projector_path,
            "--defended_model", DEFENDED_MODEL,
            "--anchor_model", ANCHOR_MODEL,
            "--defended_type", DEFENDED_TYPE,
            "--anchor_type", ANCHOR_TYPE,
            "--projector_layer", str(PROJECTOR_LAYER),
            "--alpha", str(hyperparams["alpha"]),
            "--beta", str(hyperparams["beta"]),
            "--gamma", str(hyperparams["gamma"]),
            "--delta", str(hyperparams["delta"]),
            "--train_steps", str(hyperparams["train_steps"]),
            "--lr", str(hyperparams["lr"]),
            "--batch_size", str(hyperparams["batch_size"]),
            "--output_dir", adapter_output_dir,
        ]
        print(f"  [DRY RUN] Train: {' '.join(train_cmd)}")
        print(f"  [DRY RUN] Eval would follow")
        return

    # --- STEP 1: TRAIN ---
    train_cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--projector_path", projector_path,
        "--defended_model", DEFENDED_MODEL,
        "--anchor_model", ANCHOR_MODEL,
        "--defended_type", DEFENDED_TYPE,
        "--anchor_type", ANCHOR_TYPE,
        "--projector_layer", str(PROJECTOR_LAYER),
        "--alpha", str(hyperparams["alpha"]),
        "--beta", str(hyperparams["beta"]),
        "--gamma", str(hyperparams["gamma"]),
        "--delta", str(hyperparams["delta"]),
        "--train_steps", str(hyperparams["train_steps"]),
        "--lr", str(hyperparams["lr"]),
        "--batch_size", str(hyperparams["batch_size"]),
        "--output_dir", adapter_output_dir,
    ]

    t_train_start = time.time()
    try:
        result = run_command(train_cmd, f"Train defense (lambda={lam_str})", timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed (rc={result.returncode})")

        adapter_path = os.path.join(adapter_output_dir, "defense_adapter_final")
        if not os.path.exists(adapter_path):
            raise RuntimeError(f"Adapter not found at {adapter_path}")

        row["adapter_path"] = adapter_path
        row["training_time_sec"] = round(time.time() - t_train_start, 1)
    except Exception as e:
        print(f"[!] Training error for lambda={lam_str}: {e}")
        row["status"] = f"TRAIN_FAILED: {str(e)[:100]}"
        row["training_time_sec"] = round(time.time() - t_train_start, 1)
        append_row(csv_path, row)
        return

    # --- STEP 2: EVALUATE ---
    eval_cmd = [
        sys.executable, EVAL_SCRIPT,
        "--skip_training",
        "--adapter_path", adapter_path,
        "--target_model", DEFENDED_MODEL,
        "--target_type", DEFENDED_TYPE,
        "--anchor-model", ANCHOR_TYPE,
        "--defender-model", DEFENDED_TYPE,
        "--projector_path", projector_path,
        "--gcg_data_path", gcg_data_path,
        "--output_dir", lambda_dir,
    ]

    t_eval_start = time.time()
    try:
        result = run_command(eval_cmd, f"Evaluate defense (lambda={lam_str})", timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(f"Evaluation failed (rc={result.returncode})")

        # Parse results from the JSON output
        results_json = os.path.join(lambda_dir, "results.json")
        if os.path.exists(results_json):
            with open(results_json, 'r') as f:
                eval_data = json.load(f)

            defended = eval_data.get("defended", {})
            row["asr_self"] = defended.get("asr_self", "")
            row["asr_anchor"] = defended.get("asr_anchor", "")
            row["asr_other"] = defended.get("asr_other", "")
            row["brr"] = defended.get("brr", "")
            row["ppl"] = defended.get("ppl", "")
            row["tdr"] = defended.get("tdr", "")
            row["cka_score"] = defended.get("cka_score", "")
        else:
            raise RuntimeError("results.json not found after evaluation")

        row["eval_time_sec"] = round(time.time() - t_eval_start, 1)
        row["status"] = "OK"

    except Exception as e:
        print(f"[!] Evaluation error for lambda={lam_str}: {e}")
        row["status"] = f"EVAL_FAILED: {str(e)[:100]}"
        row["eval_time_sec"] = round(time.time() - t_eval_start, 1)

    append_row(csv_path, row)

    total_time = round(time.time() - t_train_start, 1)
    print(f"\n[+] Lambda {lam_str} completed in {total_time}s — status: {row['status']}")


# ==========================================
# SUMMARY
# ==========================================
def print_summary(csv_path: str):
    """Read CSV and print comparison table with best lambda recommendation."""
    if not os.path.exists(csv_path):
        print("[!] No results CSV found.")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    ok_df = df[df['status'] == 'OK'].copy()

    if len(ok_df) == 0:
        print("[!] No successful runs found.")
        n_fail = len(df[df['status'] != 'OK'])
        print(f"    Failed: {n_fail}")
        return

    print(f"\n{'='*90}")
    print("LAMBDA SWEEP RESULTS")
    print(f"{'='*90}")

    # Convert numeric columns
    for col in ['asr_self', 'asr_anchor', 'asr_other', 'brr', 'ppl', 'tdr', 'cka_score']:
        ok_df[col] = pd.to_numeric(ok_df[col], errors='coerce')

    # Print table
    header = f"{'Lambda':>8} {'ASR-Self':>10} {'ASR-Anchor':>12} {'ASR-Other':>11} {'BRR':>8} {'PPL':>8} {'TDR':>8} {'CKA':>8}"
    print(header)
    print("-" * len(header))

    for _, row in ok_df.iterrows():
        print(
            f"{row['lambda']:>8} "
            f"{row['asr_self']*100:>9.1f}% "
            f"{row['asr_anchor']*100:>11.1f}% "
            f"{row['asr_other']*100:>10.1f}% "
            f"{row['brr']*100:>7.1f}% "
            f"{row['ppl']:>8.2f} "
            f"{row['tdr']:>8.4f} "
            f"{row['cka_score']:>8.4f}"
        )

    # Compute composite score: lower ASR + lower BRR + reasonable PPL
    # Score = avg_asr + brr + ppl_penalty (lower is better)
    ok_df['avg_asr'] = ok_df[['asr_self', 'asr_anchor', 'asr_other']].mean(axis=1)
    ppl_median = ok_df['ppl'].median()
    ok_df['ppl_penalty'] = (ok_df['ppl'] / ppl_median - 1).clip(lower=0) * 0.1
    ok_df['composite_score'] = ok_df['avg_asr'] + ok_df['brr'] + ok_df['ppl_penalty']

    best_idx = ok_df['composite_score'].idxmin()
    best_row = ok_df.loc[best_idx]

    print(f"\n{'='*90}")
    print(f"BEST LAMBDA: {best_row['lambda']}")
    print(f"  Avg ASR: {best_row['avg_asr']*100:.1f}%")
    print(f"  BRR:     {best_row['brr']*100:.1f}%")
    print(f"  PPL:     {best_row['ppl']:.2f}")
    print(f"  TDR:     {best_row['tdr']:.4f}")
    print(f"  CKA:     {best_row['cka_score']:.4f}")
    print(f"  Composite score: {best_row['composite_score']:.4f} (lower is better)")
    print(f"  Adapter: {best_row['adapter_path']}")
    print(f"{'='*90}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Lambda Sweep for Non-Linear Projection Defense")

    parser.add_argument("--lambdas", type=float, nargs="+", default=None,
                        help="Lambda values to sweep (default: 0.1 to 0.9)")
    parser.add_argument("--results_dir", type=str, default="./lambda_sweep_results",
                        help="Directory for outputs")
    parser.add_argument("--output_csv", type=str, default="lambda_sweep_summary.csv",
                        help="Output CSV filename (relative to results_dir)")
    parser.add_argument("--gcg_data_path", type=str,
                        default="../outputs/advbench_suffixes_all_models_fixed.csv",
                        help="Path to GCG attack data")

    # Training hyperparameters
    parser.add_argument("--alpha", type=float, default=DEFAULT_HYPERPARAMS["alpha"],
                        help="Refusal direction weight")
    parser.add_argument("--beta", type=float, default=DEFAULT_HYPERPARAMS["beta"],
                        help="Coherency weight")
    parser.add_argument("--gamma", type=float, default=DEFAULT_HYPERPARAMS["gamma"],
                        help="Anchor repulsion weight")
    parser.add_argument("--delta", type=float, default=DEFAULT_HYPERPARAMS["delta"],
                        help="LM loss weight")
    parser.add_argument("--train_steps", type=int, default=DEFAULT_HYPERPARAMS["train_steps"],
                        help="Training steps")
    parser.add_argument("--lr", type=float, default=DEFAULT_HYPERPARAMS["lr"],
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_HYPERPARAMS["batch_size"],
                        help="Batch size")

    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Re-run previously failed lambdas")
    parser.add_argument("--summary_only", action="store_true",
                        help="Only print summary from existing CSV")

    args = parser.parse_args()

    lambdas = args.lambdas if args.lambdas else DEFAULT_LAMBDAS
    results_dir = os.path.join(SCRIPT_DIR, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, args.output_csv)

    if args.summary_only:
        print_summary(csv_path)
        return

    hyperparams = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    # Auto-resume
    completed = set()
    if not args.dry_run and os.path.exists(csv_path):
        completed = load_completed_lambdas(csv_path, retry_failed=args.retry_failed)
        if completed:
            print(f"[*] Auto-resume: {len(completed)} lambdas already in CSV (skipping)")

    lambdas_to_run = [l for l in lambdas if f"{l:.1f}" not in completed]

    print(f"\n{'='*70}")
    print(f"LAMBDA SWEEP FOR NON-LINEAR PROJECTION DEFENSE")
    print(f"{'='*70}")
    print(f"Lambdas: {lambdas}")
    print(f"Already completed: {len(completed)}")
    print(f"Lambdas to run: {len(lambdas_to_run)} — {lambdas_to_run}")
    print(f"Defended: {DEFENDED_MODEL} ({DEFENDED_TYPE})")
    print(f"Anchor: {ANCHOR_MODEL} ({ANCHOR_TYPE})")
    print(f"Projector layer: {PROJECTOR_LAYER}")
    print(f"Hyperparams: {hyperparams}")
    print(f"Results dir: {results_dir}")
    print(f"CSV: {csv_path}")
    print(f"GCG data: {args.gcg_data_path}")
    print(f"{'='*70}")

    if not args.dry_run:
        init_csv(csv_path)

    for i, lam in enumerate(lambdas_to_run):
        print(f"\n\n{'*'*70}")
        print(f"* LAMBDA {i+1}/{len(lambdas_to_run)}: {lam:.1f}")
        print(f"{'*'*70}")

        run_lambda(
            lam=lam,
            csv_path=csv_path,
            results_dir=results_dir,
            hyperparams=hyperparams,
            gcg_data_path=args.gcg_data_path,
            dry_run=args.dry_run,
        )

    # Print summary
    if not args.dry_run:
        print_summary(csv_path)

    print(f"\n{'='*70}")
    print("LAMBDA SWEEP COMPLETE")
    print(f"{'='*70}")
    if not args.dry_run:
        print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
