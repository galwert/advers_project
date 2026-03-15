#!/usr/bin/env python3
"""
All-Pairs Cross-Model Defense Experiment Orchestrator (Phase 3)

Runs the full train+eval pipeline (two_stage_defense_v2.py + evaluate_v2.py)
for every (anchor, defender) pair across 10 models. Each pair is launched as a
subprocess to ensure full GPU memory cleanup between runs. Results are collected
in a CSV with resume support via file-level locking.

Supports multi-GPU parallelism: each worker takes a shard of pairs, with
per-pair CSV locking to prevent duplicate work.

Usage:
    python run_all_pairs_experiment.py                                    # Full 90-pair run (auto-resume)
    python run_all_pairs_experiment.py --gpu 0 --worker_id 0 --num_workers 3  # Multi-GPU shard
    python run_all_pairs_experiment.py --dry_run                          # Print commands only
    python run_all_pairs_experiment.py --anchor llama2 --defender vicuna  # Single pair
    python run_all_pairs_experiment.py --retry_failed                     # Re-run failed pairs
"""

import subprocess
import os
import sys
import json
import csv
import time
import fcntl
import shutil
import argparse
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any


# ==========================================
# CONFIGURATION
# ==========================================
MODELS = ["llama2", "llama3", "vicuna", "mistral", "zephyr", "yi", "qwen", "starling", "phi2", "orca"]

# Best hyperparameters from user's experiments
# alpha reduced 0.15→0.08 and epsilon increased 1.0→2.0 to reduce over-refusal on OR-Bench/XSTest
HYPERPARAMS = {
    "alpha": 0.08,
    "beta": 1.0,
    "gamma": 0.3,
    "delta": 0.06,
    "epsilon": 2.5,
    "stage2_steps": 200,
    "alignment": "cka",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "two_stage_defense_v2.py")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "evaluate_v2.py")

CSV_COLUMNS = [
    "anchor", "defender", "is_self_defense",
    "baseline_asr_self", "baseline_asr_anchor", "baseline_asr_other",
    "baseline_brr", "baseline_ppl", "baseline_tdr", "baseline_cka",
    "defended_asr_self", "defended_asr_anchor", "defended_asr_other",
    "defended_brr", "defended_ppl", "defended_tdr", "defended_cka",
    "delta_asr_self", "delta_asr_anchor", "delta_asr_other", "delta_brr",
    "adapter_path", "adapter_timestamp", "training_time_sec", "eval_time_sec",
    "sample_succeeded_attacks", "sample_refused_benign", "status",
]


# ==========================================
# UTILITIES
# ==========================================
def load_completed_pairs(csv_path: str, retry_failed: bool = False) -> set:
    """Load already-completed (anchor, defender) pairs from CSV.

    By default skips only OK pairs. With retry_failed=False (default),
    also skips failed pairs so they aren't re-run. Use --retry_failed
    to re-attempt pairs that previously failed.
    """
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "")
            if status == "OK":
                completed.add((row["anchor"], row["defender"]))
            elif not retry_failed and status:
                completed.add((row["anchor"], row["defender"]))
        fcntl.flock(f, fcntl.LOCK_UN)
    return completed


def init_csv(csv_path: str):
    """Create CSV with header if it doesn't exist (race-safe)."""
    if os.path.exists(csv_path):
        return
    try:
        fd = os.open(csv_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o644)
        with os.fdopen(fd, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
    except FileExistsError:
        pass  # Another worker created it first


def append_row(csv_path: str, row: Dict[str, Any]):
    """Append a single row to the CSV with file locking."""
    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def run_command(cmd: list, description: str, timeout: int = 7200, gpu: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output in real-time."""
    print(f"\n{'='*60}", flush=True)
    print(f"[CMD] {description}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)

    # Stream output in real-time while also capturing it
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

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


def extract_adapter_path(stdout: str) -> Optional[str]:
    """Extract adapter path from training script stdout."""
    for line in stdout.split('\n'):
        if "Adapter saved to:" in line:
            path = line.split("Adapter saved to:")[-1].strip()
            if os.path.exists(path):
                return path
    # Fallback: look for the path pattern in output
    for line in reversed(stdout.split('\n')):
        line = line.strip()
        if line.startswith("./two_stage_outputs_v2/") or line.startswith("two_stage_outputs_v2/"):
            if os.path.exists(os.path.join(SCRIPT_DIR, line)):
                return os.path.join(SCRIPT_DIR, line)
    return None


def format_examples_for_csv(examples: dict, key: str, max_items: int = 3) -> str:
    """Format example list to a short string for CSV."""
    items = examples.get(key, [])
    if not items:
        return ""
    summaries = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            prompt = item.get("prompt", "")[:60]
            response = item.get("response", "")[:80]
            summaries.append(f"Q:{prompt}|A:{response}")
        else:
            summaries.append(str(item)[:100])
    # Sanitize: remove newlines and carriage returns that break CSV parsing
    result = " ||| ".join(summaries)
    return result.replace("\n", " ").replace("\r", " ").replace('"', "'")


# ==========================================
# CORE: RUN ONE PAIR
# ==========================================
def run_pair(
    anchor: str,
    defender: str,
    csv_path: str,
    gcg_data_path: str,
    output_dir: str,
    n_eval: int,
    dry_run: bool = False,
    gpu: Optional[str] = None,
    low_memory: bool = False,
    seed: int = 42,
    baseline_cache_dir: Optional[str] = None,
    verbose: bool = False,
    precision: str = "4bit",
    cka_scope: Optional[str] = None,
):
    """Train and evaluate defense for one (anchor, defender) pair."""
    is_self = anchor == defender
    gpu_tag = f" [GPU {gpu}]" if gpu is not None else ""
    pair_label = f"{anchor}->{defender}" + (" (SELF)" if is_self else "")
    print(f"\n{'#'*70}")
    print(f"# PAIR: {pair_label}{gpu_tag}")
    print(f"{'#'*70}")

    row = {col: "" for col in CSV_COLUMNS}
    row["anchor"] = anchor
    row["defender"] = defender
    row["is_self_defense"] = is_self

    # Extra flags shared across train/eval commands
    extra_precision = ["--precision", precision] if precision != "4bit" else []
    extra_cka_scope = ["--cka_scope", cka_scope] if cka_scope else []

    if dry_run:
        train_cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--anchor", anchor, "--defender", defender,
            "--alignment", HYPERPARAMS["alignment"],
            "--alpha", str(HYPERPARAMS["alpha"]),
            "--beta", str(HYPERPARAMS["beta"]),
            "--gamma", str(HYPERPARAMS["gamma"]),
            "--delta", str(HYPERPARAMS["delta"]),
            "--epsilon", str(HYPERPARAMS["epsilon"]),
            "--stage2_steps", str(HYPERPARAMS["stage2_steps"]),
            "--output_dir", output_dir,
            "--gcg_data_path", gcg_data_path,
            "--seed", str(seed),
        ] + extra_precision + extra_cka_scope
        print(f"  [DRY RUN] Train: {' '.join(train_cmd)}")
        print(f"  [DRY RUN] Eval baseline + defended would follow")
        return

    # --- STEP 1: BASELINE EVALUATION ---
    baseline_json = os.path.join(output_dir, f"baseline_{defender}_{anchor}.json")

    # Check baseline cache
    baseline_cached = False
    if baseline_cache_dir:
        cache_file = os.path.join(baseline_cache_dir, f"baseline_{defender}_{anchor}.json")
        if os.path.exists(cache_file):
            print(f"[*] Using cached baseline from {cache_file}")
            baseline_json = cache_file
            baseline_cached = True

    t0 = time.time()
    try:
        if not baseline_cached:
            eval_baseline_cmd = [
                sys.executable, EVAL_SCRIPT,
                "--adapter_path", "none",
                "--defender", defender,
                "--anchor", anchor,
                "--gcg_data_path", gcg_data_path,
                "--n_eval", str(n_eval),
                "--baseline",
                "--output_json", baseline_json,
            ] + (["--low_memory"] if low_memory else []) + (["--verbose"] if verbose else []) + extra_precision

            result = run_command(eval_baseline_cmd, f"Baseline eval: {pair_label}", timeout=3600, gpu=gpu)
            if result.returncode != 0:
                raise RuntimeError(f"Baseline eval failed (rc={result.returncode})")

            # Save to cache if dir provided
            if baseline_cache_dir:
                os.makedirs(baseline_cache_dir, exist_ok=True)
                cache_file = os.path.join(baseline_cache_dir, f"baseline_{defender}_{anchor}.json")
                shutil.copy2(baseline_json, cache_file)
                print(f"[+] Baseline cached to {cache_file}")

        with open(baseline_json, 'r') as f:
            baseline_data = json.load(f)

        bl = baseline_data.get("baseline", baseline_data.get("defended", {}))
        row["baseline_asr_self"] = bl.get("asr_self", "")
        row["baseline_asr_anchor"] = bl.get("asr_anchor", "")
        row["baseline_asr_other"] = bl.get("asr_other", "")
        row["baseline_brr"] = bl.get("brr", "")
        row["baseline_ppl"] = bl.get("ppl", "")
        row["baseline_tdr"] = bl.get("tdr", "")
        row["baseline_cka"] = bl.get("cka_score", "")
    except Exception as e:
        print(f"[!] Baseline eval error: {e}")
        row["status"] = f"BASELINE_FAILED: {str(e)[:100]}"
        append_row(csv_path, row)
        return

    # --- STEP 2: TRAIN DEFENSE ---
    train_cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--anchor", anchor, "--defender", defender,
        "--alignment", HYPERPARAMS["alignment"],
        "--alpha", str(HYPERPARAMS["alpha"]),
        "--beta", str(HYPERPARAMS["beta"]),
        "--gamma", str(HYPERPARAMS["gamma"]),
        "--delta", str(HYPERPARAMS["delta"]),
        "--epsilon", str(HYPERPARAMS["epsilon"]),
        "--stage2_steps", str(HYPERPARAMS["stage2_steps"]),
        "--output_dir", output_dir,
        "--gcg_data_path", gcg_data_path,
        "--seed", str(seed),
    ] + extra_precision + extra_cka_scope

    t_train_start = time.time()
    try:
        result = run_command(train_cmd, f"Train defense: {pair_label}", timeout=7200, gpu=gpu)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed (rc={result.returncode})")

        adapter_path = extract_adapter_path(result.stdout)
        if not adapter_path:
            raise RuntimeError("Could not find adapter path in training output")

        row["adapter_path"] = adapter_path
        # Extract timestamp from adapter dir name (e.g. defender_v2_cka_20260206_143021)
        adapter_basename = os.path.basename(adapter_path)
        parts = adapter_basename.rsplit("_", 2)
        if len(parts) >= 3:
            row["adapter_timestamp"] = f"{parts[-2]}_{parts[-1]}"
        else:
            row["adapter_timestamp"] = adapter_basename
        row["training_time_sec"] = round(time.time() - t_train_start, 1)
    except Exception as e:
        print(f"[!] Training error: {e}")
        row["status"] = f"TRAIN_FAILED: {str(e)[:100]}"
        row["training_time_sec"] = round(time.time() - t_train_start, 1)
        append_row(csv_path, row)
        return

    # --- STEP 3: EVALUATE DEFENDED MODEL ---
    defended_json = os.path.join(output_dir, f"defended_{defender}_{anchor}.json")
    eval_defended_cmd = [
        sys.executable, EVAL_SCRIPT,
        "--adapter_path", adapter_path,
        "--defender", defender,
        "--anchor", anchor,
        "--gcg_data_path", gcg_data_path,
        "--n_eval", str(n_eval),
        "--output_json", defended_json,
    ] + (["--low_memory"] if low_memory else []) + (["--verbose"] if verbose else []) + extra_precision

    t_eval_start = time.time()
    try:
        result = run_command(eval_defended_cmd, f"Eval defended: {pair_label}", timeout=3600, gpu=gpu)
        if result.returncode != 0:
            raise RuntimeError(f"Defended eval failed (rc={result.returncode})")

        with open(defended_json, 'r') as f:
            defended_data = json.load(f)

        dd = defended_data.get("defended", {})
        row["defended_asr_self"] = dd.get("asr_self", "")
        row["defended_asr_anchor"] = dd.get("asr_anchor", "")
        row["defended_asr_other"] = dd.get("asr_other", "")
        row["defended_brr"] = dd.get("brr", "")
        row["defended_ppl"] = dd.get("ppl", "")
        row["defended_tdr"] = dd.get("tdr", "")
        row["defended_cka"] = dd.get("cka_score", "")

        row["eval_time_sec"] = round(time.time() - t_eval_start, 1)

        # Compute deltas
        for metric in ["asr_self", "asr_anchor", "asr_other", "brr"]:
            bl_val = row.get(f"baseline_{metric}", "")
            def_val = row.get(f"defended_{metric}", "")
            if bl_val != "" and def_val != "":
                try:
                    row[f"delta_{metric}"] = round(float(def_val) - float(bl_val), 4)
                except (ValueError, TypeError):
                    row[f"delta_{metric}"] = ""

        # Examples
        def_examples = defended_data.get("defended_examples", {})
        attack_examples = def_examples.get("succeeded_attacks", {})
        all_attack_ex = []
        for attack_type_examples in attack_examples.values():
            all_attack_ex.extend(attack_type_examples)
        row["sample_succeeded_attacks"] = format_examples_for_csv(
            {"items": all_attack_ex}, "items", max_items=3
        )
        row["sample_refused_benign"] = format_examples_for_csv(
            {"items": def_examples.get("refused_benign", [])}, "items", max_items=3
        )

        row["status"] = "OK"

    except Exception as e:
        print(f"[!] Defended eval error: {e}")
        row["status"] = f"EVAL_FAILED: {str(e)[:100]}"
        row["eval_time_sec"] = round(time.time() - t_eval_start, 1)

    append_row(csv_path, row)

    total_time = round(time.time() - t0, 1)
    print(f"\n[+] Pair {pair_label} completed in {total_time}s — status: {row['status']}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run All-Pairs Defense Experiment")

    parser.add_argument("--anchor", type=str, default=None,
                        help="Run only this anchor (default: all)")
    parser.add_argument("--defender", type=str, default=None,
                        help="Run only this defender (default: all)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Override model list (default: all 10)")
    parser.add_argument("--output_csv", type=str, default="all_pairs_results.csv",
                        help="Output CSV path")
    parser.add_argument("--output_dir", type=str, default="./two_stage_outputs_v2",
                        help="Directory for adapters and JSON files")
    parser.add_argument("--gcg_data_path", type=str,
                        default="../outputs/advbench_suffixes_all_models_fixed.csv",
                        help="Path to GCG attack data")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of eval samples per attack type")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Re-run pairs that previously failed (default: skip them)")

    # Multi-GPU / multi-worker
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU id to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="This worker's id (0-indexed)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--low_memory", action="store_true",
                        help="Quantize judge to 4-bit (~9GB peak, fits 2080 Ti)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (each pair gets seed + pair_index)")
    parser.add_argument("--baseline_cache_dir", type=str, default=None,
                        help="Directory to cache baseline results. If set, looks for cached "
                             "baseline before running; saves after running if not found.")
    parser.add_argument("--do_self", action="store_true",
                        help="Include self-defense pairs (anchor==defender). Skipped by default.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print succeeded attacks and refused benign prompts during evaluation")
    parser.add_argument("--precision", type=str, default="4bit",
                        choices=["4bit", "fp16", "fp32"],
                        help="Model precision: 4bit (NF4 quantization), fp16, or fp32 (default: 4bit)")
    parser.add_argument("--cka_scope", type=str, default=None,
                        choices=["all", "harmful_only", "benign_only"],
                        help="CKA repulsion scope for training (default: not set)")

    args = parser.parse_args()

    models = args.models if args.models else MODELS
    csv_path = os.path.join(SCRIPT_DIR, args.output_csv)
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Build pair list
    if args.anchor and args.defender:
        pairs = [(args.anchor, args.defender)]
    elif args.anchor:
        pairs = [(args.anchor, d) for d in models]
    elif args.defender:
        pairs = [(a, args.defender) for a in models]
    else:
        pairs = [(a, d) for a in models for d in models]

    # Skip self-defense pairs unless --do_self is passed
    if not args.do_self:
        pairs = [(a, d) for a, d in pairs if a != d]

    # Auto-resume: always skip pairs already in the CSV
    completed = set()
    if not args.dry_run and os.path.exists(csv_path):
        completed = load_completed_pairs(csv_path, retry_failed=args.retry_failed)
        if completed:
            print(f"[*] Auto-resume: {len(completed)} pairs already in CSV (skipping)")

    pairs_to_run = [(a, d) for a, d in pairs if (a, d) not in completed]

    # Shard across workers: each worker takes every num_workers-th pair
    if args.num_workers > 1:
        pairs_to_run = [p for i, p in enumerate(pairs_to_run) if i % args.num_workers == args.worker_id]

    print(f"\n{'='*70}")
    print(f"ALL-PAIRS DEFENSE EXPERIMENT")
    print(f"{'='*70}")
    print(f"Models: {models}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Already completed: {len(completed)}")
    if args.num_workers > 1:
        print(f"Worker: {args.worker_id}/{args.num_workers} | GPU: {args.gpu}")
    print(f"Pairs for this worker: {len(pairs_to_run)}")
    print(f"Output CSV: {csv_path}")
    print(f"Output dir: {output_dir}")
    print(f"Hyperparams: {HYPERPARAMS}")
    print(f"{'='*70}")

    if not args.dry_run:
        init_csv(csv_path)

    # Stagger worker starts to avoid disk I/O bottleneck during model loading
    if args.num_workers > 1 and args.worker_id > 0 and not args.dry_run:
        delay = args.worker_id * 60  # 60s between workers
        print(f"[*] Staggering start: waiting {delay}s (worker {args.worker_id})")
        time.sleep(delay)

    if pairs_to_run:
        print(f"\nFirst pair: {pairs_to_run[0][0]}->{pairs_to_run[0][1]}")
        print(f"Last pair:  {pairs_to_run[-1][0]}->{pairs_to_run[-1][1]}")

    for i, (anchor, defender) in enumerate(pairs_to_run):
        # Re-check CSV before each pair — another worker may have finished it
        if not args.dry_run:
            fresh_completed = load_completed_pairs(csv_path, retry_failed=args.retry_failed)
            if (anchor, defender) in fresh_completed:
                print(f"\n[*] Skipping {anchor}->{defender} (completed by another worker)")
                continue

        print(f"\n\n{'*'*70}")
        print(f"* PAIR {i+1}/{len(pairs_to_run)}: anchor={anchor}, defender={defender}")
        print(f"{'*'*70}")

        run_pair(
            anchor=anchor,
            defender=defender,
            csv_path=csv_path,
            gcg_data_path=args.gcg_data_path,
            output_dir=output_dir,
            n_eval=args.n_eval,
            dry_run=args.dry_run,
            gpu=args.gpu,
            low_memory=args.low_memory,
            seed=args.seed + i,
            baseline_cache_dir=args.baseline_cache_dir,
            verbose=args.verbose,
            precision=args.precision,
            cka_scope=args.cka_scope,
        )

    print(f"\n\n{'='*70}")
    print(f"WORKER {args.worker_id} COMPLETE")
    print(f"{'='*70}")
    if not args.dry_run:
        print(f"Results saved to: {csv_path}")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            n_ok = len(df[df['status'] == 'OK'])
            n_fail = len(df[df['status'] != 'OK'])
            print(f"  Successful: {n_ok}")
            print(f"  Failed: {n_fail}")
            if n_ok > 0:
                ok_df = df[df['status'] == 'OK']
                for col in ['delta_asr_self', 'delta_asr_anchor', 'delta_asr_other', 'delta_brr']:
                    vals = pd.to_numeric(ok_df[col], errors='coerce').dropna()
                    if len(vals) > 0:
                        print(f"  Avg {col}: {vals.mean():.4f}")


if __name__ == "__main__":
    main()
