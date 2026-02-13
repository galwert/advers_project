#!/usr/bin/env python3
"""
Batch Experiment Runner for Patchscopes Refusal Transfer Attacks

This script runs a series of experiments with different configurations and
documents the results in a structured format.

Usage:
    python run_experiments.py [--dry-run] [--subset N] [--output-dir DIR]

Examples:
    # Run all experiments
    python run_experiments.py

    # Dry run (show commands without executing)
    python run_experiments.py --dry-run

    # Run only first 3 experiments
    python run_experiments.py --subset 3

    # Custom output directory
    python run_experiments.py --output-dir ./my_results

    # Run specific experiment
    python run_experiments.py --experiment exp2a_generic_safety_check
"""

import subprocess
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import time

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# Each experiment is a dict with:
#   - name: Human-readable name for the experiment
#   - description: What this experiment tests
#   - hypothesis: Expected outcome
#   - args: Dict of CLI arguments to pass to patchscopes_refusal_transfer.py

EXPERIMENTS: List[Dict[str, Any]] = [
    # =========================================================================
    # EXPERIMENT 1: THE "KEYS VS. LOCK" BASELINE
    # Goal: Establish natural compatibility ceiling between models
    # =========================================================================
    {
        "name": "exp1a_naive_baseline",
        "description": "Exp 1A: Naive Baseline - Source GCG keys on Target model (no intervention)",
        "hypothesis": "Low success (~26%). Llama's keys don't fit Vicuna's lock well.",
        "args": {
            "attack-mode": "none",  # No patching - just run target model with source suffix
            "suffix-source": "source",
            "max-examples": 100,
        }
    },
    {
        "name": "exp1b_strong_baseline",
        "description": "Exp 1B: Strong Baseline - Target GCG keys (THE BAR TO BEAT)",
        "hypothesis": "High success (~71%). This is the control group. Any intervention must beat this.",
        "args": {
            "attack-mode": "none",  # No patching - just run target model with target suffix
            "suffix-source": "target",
            "max-examples": 100,
        }
    },

    # =========================================================================
    # EXPERIMENT 2: THE "ADVERSARIAL GEOMETRY" TEST
    # Goal: Prove that "Adversarial Refusal" is geometrically distinct from "Generic Refusal"
    # =========================================================================
    {
        "name": "exp2a_generic_safety_check",
        "description": "Exp 2A: Generic Projection (Safety Check) - Remove standard refusal vector",
        "hypothesis": "Small gain (+3%). Generic safety vectors don't perfectly align with adversarial triggers.",
        "args": {
            "attack-mode": "level2a_generic",
            "suffix-source": "target",
            "ablation-strength": 1.0,
            "max-examples": 100,
        }
    },
    {
        "name": "exp2b_specific_failure_check",
        "description": "Exp 2B: Specific Projection (Failure Check) - Remove v=Mean(Failures)-Mean(Successes)",
        "hypothesis": "HIGHEST ASR. Captures exact geometric direction of the 'Adversarial Filter'.",
        "args": {
            "attack-mode": "level2b_specific",
            "suffix-source": "target",
            "ablation-strength": 1.0,
            "max-examples": 100,
        }
    },

    # =========================================================================
    # EXPERIMENT 3: THE "DEEP CLEAN" (LAYER DISTRIBUTION)
    # Goal: Test if refusal is distributed across network layers
    # =========================================================================
    {
        "name": "exp3_deep_clean_layers_13_17",
        "description": "Exp 3: Deep Clean - Multi-layer ablation at Semantic→Safety transition (layers 13-17)",
        "hypothesis": "Higher ASR than single-layer. Proves safety features are redundant across depth.",
        "args": {
            "attack-mode": "multi_layer_v2",
            "suffix-source": "target",
            "ablation-strength": 1.0,
            "ablation-layers": "13,14,15,16,17",
            "max-examples": 100,
        }
    },

    # =========================================================================
    # EXPERIMENT 4: THE "SURGICAL STRIKE" (SEMANTIC TARGETING)
    # Goal: Maximize coherence by only ablating dangerous token positions
    # =========================================================================
    {
        "name": "exp4_surgical_strike",
        "description": "Exp 4: Surgical Strike - Target only harmful keywords with α=1.5",
        "hypothesis": "Best Perplexity. Neutral tokens untouched, harm signal aggressively removed.",
        "args": {
            "attack-mode": "targeted_positions",
            "suffix-source": "target",
            "target-keywords": "bomb,kill,hack,steal,poison,create,write",
            "ablation-strength": 1.5,
            "max-examples": 100,
        }
    },

    # =========================================================================
    # EXPERIMENT 5: THE "OVER-ABLATION" SWEEP
    # Goal: Test robustness of refusal direction and find optimal ablation strength
    # =========================================================================
    {
        "name": "exp5a_over_ablation_1.5",
        "description": "Exp 5A: Over-ablation α=1.5 (50% past neutral)",
        "hypothesis": "ASR may increase but perplexity might start degrading.",
        "args": {
            "attack-mode": "level2b_specific",
            "suffix-source": "target",
            "ablation-strength": 1.5,
            "max-examples": 100,
        }
    },
    {
        "name": "exp5b_over_ablation_2.0",
        "description": "Exp 5B: Over-ablation α=2.0 (double strength)",
        "hypothesis": "Find the 'Safety Margin' - peak ASR vs perplexity tradeoff.",
        "args": {
            "attack-mode": "level2b_specific",
            "suffix-source": "target",
            "ablation-strength": 2.0,
            "max-examples": 100,
        }
    },

    # =========================================================================
    # ADDITIONAL EXPERIMENTS (OPTIONAL - for completeness)
    # =========================================================================
    {
        "name": "exp_baseline_no_intervention",
        "description": "Pure Baseline: No intervention at all (just target model with target suffix)",
        "hypothesis": "Should match exp1b if level1 direct patch has no effect.",
        "args": {
            "attack-mode": "none",
            "suffix-source": "target",
            "max-examples": 100,
        }
    },
    {
        "name": "exp_level3_procrustes",
        "description": "Level 3: Full Procrustes rotation (expected: gibberish)",
        "hypothesis": "High perplexity/gibberish. Proves rotation is NOT global.",
        "args": {
            "attack-mode": "level3_procrustes",
            "suffix-source": "target",
            "max-examples": 100,
        }
    },
    {
        "name": "exp_level4_hybrid",
        "description": "Level 4: Hybrid - Rotate refusal vector only, not hidden state",
        "hypothesis": "Black-box attack capability test.",
        "args": {
            "attack-mode": "level4_hybrid",
            "suffix-source": "target",
            "ablation-strength": 1.0,
            "max-examples": 100,
        }
    },
    {
        "name": "exp_generic_alpha2.0",
        "description": "Level 2A Generic with α=2.0 (for comparison with specific)",
        "hypothesis": "Compare generic vs specific at same ablation strength.",
        "args": {
            "attack-mode": "level2a_generic",
            "suffix-source": "target",
            "ablation-strength": 2.0,
            "max-examples": 100,
        }
    },
]

# Define the core experiments (the main 7 from your spec)
CORE_EXPERIMENTS = [
    "exp1a_naive_baseline",
    "exp1b_strong_baseline",
    "exp2a_generic_safety_check",
    "exp2b_specific_failure_check",
    "exp3_deep_clean_layers_13_17",
    "exp4_surgical_strike",
    "exp5a_over_ablation_1.5",
    "exp5b_over_ablation_2.0",
]

# =============================================================================
# DEFAULT SETTINGS (applied to all experiments unless overridden)
# =============================================================================

DEFAULT_SETTINGS = {
    "source": "llama2",
    "target": "vicuna",
    "device": "cuda",
    # "no-judge": True,  # Uncomment for faster runs (heuristics only)
}

# =============================================================================
# RUNNER FUNCTIONS
# =============================================================================

def build_command(experiment: Dict, base_script: str = "patchscopes_refusal_transfer.py",
                  source: str = None, target: str = None, device: str = None,
                  no_judge: bool = False) -> List[str]:
    """Build the command line for an experiment."""
    cmd = ["python", base_script]

    # Build runtime settings from CLI args (overrides DEFAULT_SETTINGS)
    runtime_settings = dict(DEFAULT_SETTINGS)
    if source:
        runtime_settings["source"] = source
    if target:
        runtime_settings["target"] = target
    if device:
        runtime_settings["device"] = device
    if no_judge:
        runtime_settings["no-judge"] = True

    # Apply runtime settings, then experiment-specific args
    all_args = {**runtime_settings, **experiment["args"]}

    for key, value in all_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    return cmd


def run_experiment(experiment: Dict, output_dir: Path, dry_run: bool = False,
                   source: str = None, target: str = None, device: str = None,
                   no_judge: bool = False) -> Dict:
    """Run a single experiment and return results."""
    name = experiment["name"]
    description = experiment["description"]
    hypothesis = experiment.get("hypothesis", "N/A")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"Description: {description}")
    print(f"Hypothesis: {hypothesis}")
    print(f"{'='*80}")

    cmd = build_command(experiment, source=source, target=target, device=device, no_judge=no_judge)
    cmd_str = " ".join(cmd)
    print(f"Command: {cmd_str}")

    result = {
        "name": name,
        "description": description,
        "hypothesis": hypothesis,
        "command": cmd_str,
        "args": experiment["args"],
        "start_time": datetime.now().isoformat(),
        "status": "pending",
    }

    if dry_run:
        print("[DRY RUN] Would execute the above command")
        result["status"] = "dry_run"
        result["end_time"] = datetime.now().isoformat()
        return result

    # Run the experiment
    try:
        start_time = time.time()

        # Run and capture output
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per experiment
        )

        elapsed_time = time.time() - start_time

        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["return_code"] = process.returncode
        result["elapsed_seconds"] = elapsed_time
        result["status"] = "success" if process.returncode == 0 else "failed"

        # Try to parse metrics from output
        metrics = parse_metrics_from_output(process.stdout)
        result.update(metrics)

        if metrics.get("asr_percent") is not None:
            asr = metrics["asr_percent"]
            success_ppl = metrics.get("success_perplexity", "N/A")
            failed_ppl = metrics.get("failed_perplexity", "N/A")
            gibberish = metrics.get("gibberish_count", "N/A")

            # Format PPL values
            if isinstance(success_ppl, float):
                success_ppl = f"{success_ppl:.2f}"
            if isinstance(failed_ppl, float):
                failed_ppl = f"{failed_ppl:.2f}"

            print(f"✓ Completed in {elapsed_time:.1f}s")
            print(f"  ASR: {asr}% | Gibberish: {gibberish}")
            print(f"  PPL (success): {success_ppl} | PPL (failed): {failed_ppl}")
        else:
            print(f"✓ Completed in {elapsed_time:.1f}s - ASR: Could not parse")

        # Save individual experiment log
        log_path = output_dir / f"{name}_log.txt"
        with open(log_path, 'w') as f:
            f.write(f"Experiment: {name}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Hypothesis: {hypothesis}\n")
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Return Code: {process.returncode}\n")
            f.write(f"Elapsed Time: {elapsed_time:.1f}s\n")
            f.write(f"\n{'='*40} STDOUT {'='*40}\n")
            f.write(process.stdout)
            f.write(f"\n{'='*40} STDERR {'='*40}\n")
            f.write(process.stderr)

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Experiment timed out after 1 hour"
        print(f"✗ TIMEOUT after 1 hour")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"✗ ERROR: {e}")

    result["end_time"] = datetime.now().isoformat()
    return result


def parse_metrics_from_output(stdout: str) -> Dict:
    """Parse ASR and other metrics from experiment output."""
    import re

    metrics = {}

    # Look for ASR patterns (in order of specificity)
    asr_patterns = [
        # "Success: 74 (74.0% ASR)" format
        r'\((\d+\.?\d*)%?\s*ASR\)',
        # "asr_percent  74.0" from pandas dataframe
        r'asr_percent\s+(\d+\.?\d*)',
        # "Running ASR: 74/100 = 74.0%"
        r'Running ASR:.*=\s*(\d+\.?\d*)%',
        # Generic "ASR: 74%" or "ASR 74"
        r'ASR[:\s]+(\d+\.?\d*)%?',
        # Table format: "74.0  74  100" (asr_percent  successful  total)
        r'^\s*(\d+\.?\d*)\s+\d+\s+100\s*$',
        # "llama2 vicuna  74.0  74  100" format
        r'\w+\s+\w+\s+(\d+\.?\d*)\s+\d+\s+100',
    ]
    for pattern in asr_patterns:
        match = re.search(pattern, stdout, re.MULTILINE)
        if match:
            metrics["asr_percent"] = float(match.group(1))
            break

    # Look for perplexity - multiple formats
    ppl_patterns = [
        r'Avg Perplexity[:\s]+(\d+\.?\d*)',
        r'avg_perplexity[:\s]+(\d+\.?\d*)',
        r'Perplexity[:\s]+(\d+\.?\d*)\s*\(lower',  # "Perplexity: 3.43 (lower = more natural)"
    ]
    for pattern in ppl_patterns:
        ppl_match = re.search(pattern, stdout, re.IGNORECASE)
        if ppl_match:
            metrics["avg_perplexity"] = float(ppl_match.group(1))
            break

    # Look for success/failed perplexity breakdown
    success_ppl = re.search(r'Successful responses[:\s]+(\d+\.?\d*)', stdout)
    if success_ppl:
        metrics["success_perplexity"] = float(success_ppl.group(1))

    failed_ppl = re.search(r'Failed responses[:\s]+(\d+\.?\d*)', stdout)
    if failed_ppl:
        metrics["failed_perplexity"] = float(failed_ppl.group(1))

    # Look for gibberish count
    gibberish_match = re.search(r'Gibberish[:\s]+(\d+)', stdout)
    if gibberish_match:
        metrics["gibberish_count"] = int(gibberish_match.group(1))

    # Real word ratio
    rwr_match = re.search(r'Real Word Ratio[:\s]+(\d+\.?\d*)%', stdout)
    if rwr_match:
        metrics["real_word_ratio"] = float(rwr_match.group(1))

    # Success count
    success_match = re.search(r'Success[:\s]+(\d+)', stdout)
    if success_match:
        metrics["success_count"] = int(success_match.group(1))

    # Total count
    total_match = re.search(r'Total[:\s]+(\d+)', stdout)
    if total_match:
        metrics["total_count"] = int(total_match.group(1))

    return metrics


def generate_summary_report(results: List[Dict], output_dir: Path):
    """Generate a summary report of all experiments."""

    # Sort by ASR (descending)
    sorted_results = sorted(
        [r for r in results if r.get("asr_percent") is not None],
        key=lambda x: x.get("asr_percent", 0),
        reverse=True
    )

    # Add experiments without ASR at the end
    no_asr = [r for r in results if r.get("asr_percent") is None]
    sorted_results.extend(no_asr)

    # Generate markdown report
    report_path = output_dir / "experiment_report.md"
    with open(report_path, 'w') as f:
        f.write("# Patchscopes Refusal Transfer - Experiment Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Key findings
        f.write("## Key Findings\n\n")

        baseline = next((r for r in results if r["name"] == "exp1b_strong_baseline"), None)
        best = max([r for r in results if r.get("asr_percent")], key=lambda x: x["asr_percent"], default=None)

        if baseline and baseline.get("asr_percent"):
            f.write(f"- **Strong Baseline (Target GCG):** {baseline['asr_percent']:.1f}% ASR\n")
        if best and best.get("asr_percent"):
            f.write(f"- **Best Result:** {best['asr_percent']:.1f}% ASR ({best['name']})\n")
            if baseline and baseline.get("asr_percent"):
                improvement = best['asr_percent'] - baseline['asr_percent']
                f.write(f"- **Improvement over baseline:** {improvement:+.1f}%\n")

        # Summary table
        f.write("\n## Summary Table\n\n")
        f.write("| Rank | Experiment | ASR (%) | PPL (success) | PPL (failed) | Gibberish | Status |\n")
        f.write("|------|------------|---------|---------------|--------------|-----------|--------|\n")

        for i, r in enumerate(sorted_results, 1):
            asr = r.get("asr_percent", "N/A")
            if isinstance(asr, float):
                asr = f"{asr:.1f}"
            success_ppl = r.get("success_perplexity", "N/A")
            if isinstance(success_ppl, float):
                success_ppl = f"{success_ppl:.2f}"
            failed_ppl = r.get("failed_perplexity", "N/A")
            if isinstance(failed_ppl, float):
                failed_ppl = f"{failed_ppl:.2f}"
            gib = r.get("gibberish_count", "N/A")
            status = r.get("status", "unknown")
            f.write(f"| {i} | {r['name']} | {asr} | {success_ppl} | {failed_ppl} | {gib} | {status} |\n")

        # Detailed results
        f.write("\n## Detailed Results\n\n")
        for r in sorted_results:
            f.write(f"### {r['name']}\n\n")
            f.write(f"**Description:** {r['description']}\n\n")
            f.write(f"**Hypothesis:** {r.get('hypothesis', 'N/A')}\n\n")

            # Metrics
            f.write("**Results:**\n")
            if r.get("asr_percent") is not None:
                f.write(f"- ASR: {r['asr_percent']:.1f}%\n")
            if r.get("success_count") is not None and r.get("total_count") is not None:
                f.write(f"- Success/Total: {r['success_count']}/{r['total_count']}\n")
            if r.get("gibberish_count") is not None:
                f.write(f"- Gibberish count: {r['gibberish_count']}\n")
            if r.get("real_word_ratio") is not None:
                f.write(f"- Real Word Ratio: {r['real_word_ratio']:.1f}%\n")

            # Perplexity breakdown
            f.write("\n**Perplexity:**\n")
            if r.get("avg_perplexity") is not None:
                f.write(f"- Average: {r['avg_perplexity']:.2f}\n")
            if r.get("success_perplexity") is not None:
                f.write(f"- Successful responses: {r['success_perplexity']:.2f}\n")
            if r.get("failed_perplexity") is not None:
                f.write(f"- Failed responses: {r['failed_perplexity']:.2f}\n")

            f.write(f"\n**Command:**\n```bash\n{r['command']}\n```\n\n")
            if r.get("error"):
                f.write(f"**Error:** {r['error']}\n\n")
            f.write("---\n\n")

    print(f"\n📊 Report saved to: {report_path}")

    # Generate CSV summary with ALL metrics
    csv_path = output_dir / "experiment_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "name", "description", "hypothesis",
            # Core metrics
            "asr_percent", "success_count", "total_count",
            # Perplexity breakdown
            "avg_perplexity", "success_perplexity", "failed_perplexity",
            # Quality metrics
            "gibberish_count", "real_word_ratio",
            # Experiment config
            "attack_mode", "ablation_strength", "suffix_source",
            # Run info
            "status", "elapsed_seconds"
        ])
        writer.writeheader()
        for i, r in enumerate(sorted_results, 1):
            writer.writerow({
                "rank": i,
                "name": r["name"],
                "description": r["description"],
                "hypothesis": r.get("hypothesis", ""),
                # Core metrics
                "asr_percent": r.get("asr_percent", ""),
                "success_count": r.get("success_count", ""),
                "total_count": r.get("total_count", ""),
                # Perplexity breakdown
                "avg_perplexity": r.get("avg_perplexity", ""),
                "success_perplexity": r.get("success_perplexity", ""),
                "failed_perplexity": r.get("failed_perplexity", ""),
                # Quality metrics
                "gibberish_count": r.get("gibberish_count", ""),
                "real_word_ratio": r.get("real_word_ratio", ""),
                # Experiment config
                "attack_mode": r["args"].get("attack-mode", ""),
                "ablation_strength": r["args"].get("ablation-strength", 1.0),
                "suffix_source": r["args"].get("suffix-source", ""),
                # Run info
                "status": r.get("status", ""),
                "elapsed_seconds": r.get("elapsed_seconds", ""),
            })

    print(f"📈 CSV saved to: {csv_path}")

    # Save full JSON results
    json_path = output_dir / "experiment_results.json"
    with open(json_path, 'w') as f:
        # Remove stdout/stderr for cleaner JSON (they're in individual logs)
        clean_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items() if k not in ['stdout', 'stderr']}
            clean_results.append(clean_r)
        json.dump(clean_results, f, indent=2)

    print(f"📁 JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments for Patchscopes Refusal Transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXPERIMENT ORDER (The "Long Run"):
  1. exp1a_naive_baseline      - Baseline 1 (Source GCG)
  2. exp1b_strong_baseline     - Baseline 2 (Target GCG) - THE BAR TO BEAT
  3. exp2a_generic_safety_check - Safety Check (generic refusal removal)
  4. exp2b_specific_failure_check - Failure Check (CRITICAL - empirical vector)
  5. exp3_deep_clean_layers_13_17 - Depth Check (multi-layer ablation)
  6. exp4_surgical_strike      - Semantic Check (keyword targeting)
  7. exp5a/b_over_ablation     - Over-ablation sweep

QUICK START:
  python run_experiments.py --core        # Run only core experiments (1-8)
  python run_experiments.py --dry-run     # Preview commands
  python run_experiments.py --list        # List all experiments
        """
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show commands without executing them"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Run only the first N experiments"
    )
    parser.add_argument(
        "--core", action="store_true",
        help="Run only the 8 core experiments (skip optional ones)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./experiment_results",
        help="Directory to save results (default: ./experiment_results)"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Run only a specific experiment by name"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available experiments"
    )
    parser.add_argument(
        "--source", type=str, default="llama2",
        help="Source model name (default: llama2)"
    )
    parser.add_argument(
        "--target", type=str, default="vicuna",
        help="Target model name (default: vicuna)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM judge for faster runs (heuristics only)"
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n" + "="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)
        print("\nCORE EXPERIMENTS (--core flag):")
        for i, name in enumerate(CORE_EXPERIMENTS, 1):
            exp = next(e for e in EXPERIMENTS if e["name"] == name)
            print(f"  {i}. {name}")
            print(f"     {exp['description']}")
            print(f"     Hypothesis: {exp.get('hypothesis', 'N/A')}")
            print()

        print("\nADDITIONAL EXPERIMENTS:")
        for exp in EXPERIMENTS:
            if exp["name"] not in CORE_EXPERIMENTS:
                print(f"  • {exp['name']}")
                print(f"    {exp['description']}")
                print()
        return

    # Setup output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔬 Patchscopes Refusal Transfer - Batch Experiment Runner")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔄 Model pair: {args.source} -> {args.target}")
    if args.no_judge:
        print(f"⚡ Fast mode: LLM judge disabled (heuristics only)")

    # Filter experiments
    if args.core:
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] in CORE_EXPERIMENTS]
        # Sort by CORE_EXPERIMENTS order
        experiments_to_run = sorted(experiments_to_run, key=lambda x: CORE_EXPERIMENTS.index(x["name"]))
    else:
        experiments_to_run = EXPERIMENTS.copy()

    if args.experiment:
        experiments_to_run = [e for e in experiments_to_run if e["name"] == args.experiment]
        if not experiments_to_run:
            print(f"❌ Experiment '{args.experiment}' not found")
            return

    if args.subset:
        experiments_to_run = experiments_to_run[:args.subset]

    print(f"📋 Running {len(experiments_to_run)} experiments")

    if args.dry_run:
        print("🏃 DRY RUN MODE - commands will not be executed")

    # Run experiments
    results = []
    for i, experiment in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}]", end="")
        result = run_experiment(
            experiment, output_dir,
            dry_run=args.dry_run,
            source=args.source,
            target=args.target,
            device=args.device,
            no_judge=args.no_judge
        )
        results.append(result)

        # Save intermediate results
        json_path = output_dir / "experiment_results_partial.json"
        with open(json_path, 'w') as f:
            clean_results = [{k: v for k, v in r.items() if k not in ['stdout', 'stderr']} for r in results]
            json.dump(clean_results, f, indent=2)

    # Generate final report
    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80)
    generate_summary_report(results, output_dir)

    # Print quick summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get("asr_percent") is not None]
    if successful:
        best = max(successful, key=lambda x: x["asr_percent"])
        print(f"🏆 Best ASR: {best['asr_percent']:.1f}% ({best['name']})")

        baseline = next((r for r in results if r["name"] == "exp1b_strong_baseline"), None)
        if baseline and baseline.get("asr_percent") is not None:
            print(f"📊 Strong Baseline ASR: {baseline['asr_percent']:.1f}%")
            improvement = best['asr_percent'] - baseline['asr_percent']
            print(f"📈 Best improvement over baseline: {improvement:+.1f}%")

        # Perplexity comparison
        best_ppl = min([r for r in successful if r.get("avg_perplexity")],
                       key=lambda x: x["avg_perplexity"], default=None)
        if best_ppl:
            print(f"🎯 Best Perplexity: {best_ppl['avg_perplexity']:.2f} ({best_ppl['name']})")


if __name__ == "__main__":
    main()
