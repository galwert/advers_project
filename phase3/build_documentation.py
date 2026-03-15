#!/usr/bin/env python3
"""Build comprehensive documentation of all WildGuard-era ablations and best configs."""

import json
import glob
import os
import shutil
import csv
from collections import defaultdict
from pathlib import Path

OUTDIR = "/home/wertheizer/advers_project/phase3/7b_defense_wildguard_outputs"
SAVE_DIR = "/home/wertheizer/advers_project/phase3/saved_results/documentation"

# Model ID mapping
MODEL_IDS = {
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen14b": "Qwen/Qwen1.5-14B-Chat",
    "starling": "Nexusflow/Starling-LM-7B-beta",
}

ANCHOR_IDS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "qwen14b": "Qwen/Qwen1.5-14B-Chat",
    "q14": "Qwen/Qwen1.5-14B-Chat",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
    "phi2": "microsoft/phi-2",
    "phi": "microsoft/phi-2",
    "p": "microsoft/phi-2",
    "y": "01-ai/Yi-1.5-9B-Chat",
    "l2": "meta-llama/Llama-2-7b-chat-hf",
    "l3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "q": "Qwen/Qwen1.5-7B-Chat",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "starling": "Nexusflow/Starling-LM-7B-beta",
    "self": "SELF",
}


def infer_model(config_name):
    """Infer defender model from config name."""
    for prefix in ["qwen14b", "qwen7b", "qwen", "vicuna", "llama3", "llama2_13b",
                    "yi9b", "mistral7b", "mistral_nemo", "mistral", "nemo", "starling"]:
        if config_name.startswith(prefix):
            # Normalize
            if prefix == "qwen7b":
                return "qwen"
            if prefix == "mistral7b":
                return "mistral"
            if prefix in ("mistral_nemo", "nemo"):
                return "nemo"
            return prefix
    return "unknown"


def infer_anchor(config_name, model):
    """Try to infer anchor from config name patterns."""
    # Pattern: model_anc_ANCHOR_scope or model_PRESET_ANCHOR_scope
    parts = config_name.split("_")

    # Check for self-repulsion
    if "self" in parts:
        return "self"

    # For newer configs: mistral_anc_l2_hx -> anchor=l2
    # model_prefix_anchor_scope
    for i, p in enumerate(parts):
        if p in ("anc", "safe", "gen", "str"):
            if i + 1 < len(parts):
                anchor_code = parts[i + 1]
                if anchor_code in ANCHOR_IDS:
                    return anchor_code
        if p == "q14":
            return "q14"

    # For ab2 configs: qwen_ab2_hx -> anchor from training_metrics
    return None


def get_training_metrics(adapter_path):
    """Load training_metrics.json from adapter directory."""
    tm_path = os.path.join(adapter_path, "training_metrics.json")
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            return json.load(f)
    return None


def load_all_evals():
    """Load all eval JSON files."""
    evals = {}
    for f in sorted(glob.glob(os.path.join(OUTDIR, "eval_*.json"))):
        name = os.path.basename(f).replace("eval_", "").replace(".json", "")
        try:
            data = json.load(open(f))
            evals[name] = data
        except:
            pass
    return evals


def load_all_benchmarks():
    """Load all benchmark JSON files."""
    benchmarks = {}
    for f in sorted(glob.glob(os.path.join(OUTDIR, "bench_*.json"))):
        name = os.path.basename(f).replace("bench_", "").replace(".json", "")
        try:
            data = json.load(open(f))
            benchmarks[name] = data
        except:
            pass
    return benchmarks


def find_adapter_path(config_name):
    """Find the adapter path from eval JSON or by searching SLURM logs."""
    eval_f = os.path.join(OUTDIR, f"eval_{config_name}.json")
    if os.path.exists(eval_f):
        d = json.load(open(eval_f))
        # Check defended_examples for adapter hint, or check baseline
        dd = d.get("defended", d)
        # Some files store adapter_path
        ap = dd.get("adapter_path") or d.get("adapter_path")
        if ap and os.path.exists(ap):
            return ap

    # Search eval log files
    log_f = os.path.join(OUTDIR, f"eval_{config_name}.log")
    if os.path.exists(log_f):
        with open(log_f) as f:
            for line in f:
                if "adapter" in line.lower() and "defender_v2_cka" in line:
                    # Extract path
                    for token in line.split():
                        if "defender_v2_cka" in token:
                            path = token.strip().rstrip(",").rstrip(")")
                            if os.path.isdir(path):
                                return path
                            # Try relative
                            rel = os.path.join(OUTDIR, os.path.basename(path))
                            if os.path.isdir(rel):
                                return rel

    # Search SLURM logs for config->adapter mapping
    for slurm in sorted(glob.glob(os.path.join(OUTDIR, "slurm_*.out"))):
        try:
            with open(slurm) as f:
                content = f.read()
                # Look for pattern: config_name ... Adapter saved to: PATH
                # or "eval_config_name" near an adapter path
                idx = content.find(config_name)
                while idx != -1:
                    # Search nearby for adapter path
                    region = content[max(0, idx - 200):idx + 2000]
                    for line in region.split("\n"):
                        if "Adapter saved to:" in line or "adapter_path" in line:
                            for token in line.split():
                                if "defender_v2_cka" in token:
                                    path = token.strip().rstrip(",")
                                    if os.path.isdir(path):
                                        return path
                                    rel = os.path.join(OUTDIR, os.path.basename(path))
                                    if os.path.isdir(rel):
                                        return rel
                        if "Loading adapter from" in line:
                            for token in line.split():
                                if "defender_v2_cka" in token:
                                    path = token.strip().rstrip("...")
                                    if os.path.isdir(path):
                                        return path
                                    rel = os.path.join(OUTDIR, os.path.basename(path))
                                    if os.path.isdir(rel):
                                        return rel
                    idx = content.find(config_name, idx + 1)
        except:
            pass

    return None


def extract_asr(data, section="defended"):
    """Extract ASR values from eval data."""
    dd = data.get(section, data)
    asr_s = dd.get("asr_self", None)
    asr_a = dd.get("asr_anchor", None)
    asr_o = dd.get("asr_other", None)
    bgr = dd.get("bgr", None)
    brr = dd.get("brr", None)
    ppl = dd.get("ppl", None)

    # Convert float 0-1 to percentage
    for v_name in ["asr_s", "asr_a", "asr_o", "bgr", "brr"]:
        v = locals()[v_name]
        if v is not None and isinstance(v, float) and v <= 1.0:
            locals()[v_name] = v * 100

    return {
        "asr_self": round(asr_s * 100, 1) if asr_s is not None and asr_s <= 1 else asr_s,
        "asr_anchor": round(asr_a * 100, 1) if asr_a is not None and asr_a <= 1 else asr_a,
        "asr_other": round(asr_o * 100, 1) if asr_o is not None and asr_o <= 1 else asr_o,
        "bgr": round(bgr * 100, 1) if bgr is not None and bgr <= 1 else bgr,
        "brr": round(brr * 100, 1) if brr is not None and brr <= 1 else brr,
        "ppl": round(ppl, 2) if ppl is not None else None,
    }


def extract_bench(data, section="defended"):
    """Extract benchmark values."""
    dd = data.get(section, data)
    xs = dd.get("xstest_refusal_rate")
    orb = dd.get("orbench_refusal_rate")
    mt = dd.get("mtbench_score")
    mmlu = dd.get("mmlu_accuracy") or dd.get("mmlu_score")

    return {
        "xstest": round(xs * 100, 1) if xs is not None else None,
        "orbench": round(orb * 100, 1) if orb is not None else None,
        "mtbench": round(mt, 2) if mt is not None else None,
        "mmlu": round(mmlu * 100, 1) if mmlu is not None else None,
    }


def get_hyperparams(adapter_path, config_name=None):
    """Get hyperparameters from adapter's training_metrics.json."""
    if not adapter_path:
        return {}

    tm = get_training_metrics(adapter_path)
    if not tm:
        return {}

    cfg = tm.get("config", {})

    # Also get base model from adapter_config.json
    ac_path = os.path.join(adapter_path, "adapter_config.json")
    base_model = None
    if os.path.exists(ac_path):
        ac = json.load(open(ac_path))
        base_model = ac.get("base_model_name_or_path")

    return {
        "defender_id": base_model or cfg.get("defender_id"),
        "anchor_id": cfg.get("anchor_id"),
        "alpha_refusal": cfg.get("alpha"),
        "beta_coherency": cfg.get("beta"),
        "gamma_cka": cfg.get("gamma"),
        "delta_lm": cfg.get("delta"),
        "epsilon_kl": cfg.get("epsilon"),
        "zeta_sep": cfg.get("zeta"),
        "cka_scope": cfg.get("cka_scope"),
        "steps": cfg.get("stage2_steps"),
        "lora_r": cfg.get("lora_r"),
        "lr": cfg.get("stage2_lr"),
        "use_borderline": cfg.get("use_borderline"),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading all eval files...")
    evals = load_all_evals()
    print(f"  Found {len(evals)} eval files")

    print("Loading all benchmark files...")
    benchmarks = load_all_benchmarks()
    print(f"  Found {len(benchmarks)} benchmark files")

    # =========================================================================
    # TABLE 1: ALL ABLATIONS (WildGuard era)
    # =========================================================================
    print("\nBuilding Table 1: All Ablations...")

    # Collect baselines first
    baselines = {}
    for name, data in evals.items():
        if "baseline" in name:
            model = infer_model(name.replace("_baseline", "").replace("_v2", "").replace("_v3", ""))
            if model != "unknown":
                baselines[model] = extract_asr(data, "baseline")
                # Also get baseline bench
                for bname, bdata in benchmarks.items():
                    if model in bname and "baseline" in bname:
                        baselines[model].update(extract_bench(bdata, "baseline"))

    rows = []
    for name, data in sorted(evals.items()):
        # Skip baseline-only files
        if "baseline" in name and "defended" not in data:
            continue
        if "defended" not in data:
            continue

        model = infer_model(name)
        adapter_path = find_adapter_path(name)
        hp = get_hyperparams(adapter_path, name)
        asr = extract_asr(data, "defended")
        bench = extract_bench(benchmarks.get(name, {}), "defended") if name in benchmarks else {}

        # Get baseline for delta
        bl = baselines.get(model, {})

        row = {
            "config": name,
            "model": model,
            "model_id": hp.get("defender_id") or MODEL_IDS.get(model, ""),
            "anchor_id": hp.get("anchor_id", ""),
            "gamma_cka": hp.get("gamma_cka", ""),
            "epsilon_kl": hp.get("epsilon_kl", ""),
            "delta_lm": hp.get("delta_lm", ""),
            "alpha_refusal": hp.get("alpha_refusal", ""),
            "beta_coherency": hp.get("beta_coherency", ""),
            "cka_scope": hp.get("cka_scope", ""),
            "steps": hp.get("steps", ""),
            "lora_r": hp.get("lora_r", ""),
            "use_borderline": hp.get("use_borderline", ""),
            "asr_self": asr["asr_self"],
            "asr_anchor": asr["asr_anchor"],
            "asr_other": asr["asr_other"],
            "bgr": asr["bgr"],
            "brr": asr["brr"],
            "ppl": asr["ppl"],
            "xstest": bench.get("xstest", ""),
            "orbench": bench.get("orbench", ""),
            "mtbench": bench.get("mtbench", ""),
            "mmlu": bench.get("mmlu", ""),
            "adapter_path": os.path.basename(adapter_path) if adapter_path else "",
            "full_adapter_path": adapter_path or "",
        }

        # Add baseline delta columns
        if bl:
            row["delta_asr_self"] = round(asr["asr_self"] - bl.get("asr_self", 0), 1) if asr["asr_self"] is not None else ""
            row["delta_asr_anchor"] = round(asr["asr_anchor"] - bl.get("asr_anchor", 0), 1) if asr["asr_anchor"] is not None else ""
            row["delta_asr_other"] = round(asr["asr_other"] - bl.get("asr_other", 0), 1) if asr["asr_other"] is not None else ""
        else:
            row["delta_asr_self"] = ""
            row["delta_asr_anchor"] = ""
            row["delta_asr_other"] = ""

        rows.append(row)

    # Write CSV
    csv_path = os.path.join(SAVE_DIR, "all_ablations.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {len(rows)} rows to {csv_path}")

    # =========================================================================
    # TABLE 2: BASELINES
    # =========================================================================
    baseline_rows = []
    for name, data in sorted(evals.items()):
        if "baseline" not in name:
            continue
        model = infer_model(name.replace("_baseline", "").replace("_v2", "").replace("_v3", ""))
        bl_asr = extract_asr(data, "baseline")
        bl_bench = {}
        for bname, bdata in benchmarks.items():
            if model in bname and "baseline" in bname:
                bl_bench = extract_bench(bdata, "baseline")

        baseline_rows.append({
            "model": model,
            "model_id": MODEL_IDS.get(model, ""),
            "asr_self": bl_asr["asr_self"],
            "asr_anchor": bl_asr["asr_anchor"],
            "asr_other": bl_asr["asr_other"],
            "bgr": bl_asr["bgr"],
            "xstest": bl_bench.get("xstest", ""),
            "orbench": bl_bench.get("orbench", ""),
            "mtbench": bl_bench.get("mtbench", ""),
            "mmlu": bl_bench.get("mmlu", ""),
        })

    csv_path2 = os.path.join(SAVE_DIR, "baselines.csv")
    if baseline_rows:
        with open(csv_path2, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=baseline_rows[0].keys())
            writer.writeheader()
            writer.writerows(baseline_rows)
        print(f"  Wrote {len(baseline_rows)} baselines to {csv_path2}")

    # =========================================================================
    # TABLE 3: BEST CONFIGS (ASR ≤ 20%, BGR ≤ 5%)
    # =========================================================================
    print("\nBuilding Table 3: Best Configs...")

    best_rows = []
    for row in rows:
        asr_s = row["asr_self"]
        asr_a = row["asr_anchor"]
        asr_o = row["asr_other"]
        bgr = row["bgr"]

        if asr_s is None or asr_a is None or asr_o is None:
            continue

        max_asr = max(asr_s, asr_a, asr_o)
        if max_asr <= 20 and (bgr is None or bgr <= 5):
            best_rows.append(row)

    # Sort by max ASR, then by model
    best_rows.sort(key=lambda r: (r["model"], max(r["asr_self"], r["asr_anchor"], r["asr_other"])))

    csv_path3 = os.path.join(SAVE_DIR, "best_configs.csv")
    if best_rows:
        with open(csv_path3, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=best_rows[0].keys())
            writer.writeheader()
            writer.writerows(best_rows)
        print(f"  Wrote {len(best_rows)} best configs to {csv_path3}")

    # =========================================================================
    # SAVE WORKING CONFIGS: Copy adapters + outputs
    # =========================================================================
    print("\nSaving working config outputs...")

    working_dir = os.path.join(SAVE_DIR, "working_configs")
    os.makedirs(working_dir, exist_ok=True)

    saved_count = 0
    for row in best_rows:
        config = row["config"]
        model = row["model"]
        adapter_path = row["full_adapter_path"]

        if not adapter_path or not os.path.isdir(adapter_path):
            print(f"  SKIP {config}: no adapter path")
            continue

        # Create model_name/adapter_dirname/
        adapter_name = os.path.basename(adapter_path)
        config_dir = os.path.join(working_dir, model, config)
        os.makedirs(config_dir, exist_ok=True)

        # 1. Copy adapter (symlink to save space)
        adapter_dest = os.path.join(config_dir, "adapter")
        if not os.path.exists(adapter_dest):
            os.symlink(adapter_path, adapter_dest)

        # 2. Save hyperparameters JSON
        hp = get_hyperparams(adapter_path, config)
        hp["config_name"] = config
        hp["original_adapter_path"] = adapter_path
        hp["adapter_dirname"] = adapter_name
        with open(os.path.join(config_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f, indent=2)

        # 3. Copy eval JSON (has full ASR responses)
        eval_src = os.path.join(OUTDIR, f"eval_{config}.json")
        if os.path.exists(eval_src):
            shutil.copy2(eval_src, os.path.join(config_dir, f"eval_{config}.json"))

        # 4. Copy eval log
        eval_log = os.path.join(OUTDIR, f"eval_{config}.log")
        if os.path.exists(eval_log):
            shutil.copy2(eval_log, os.path.join(config_dir, f"eval_{config}.log"))

        # 5. Copy benchmark JSON
        bench_src = os.path.join(OUTDIR, f"bench_{config}.json")
        if os.path.exists(bench_src):
            shutil.copy2(bench_src, os.path.join(config_dir, f"bench_{config}.json"))

        # 6. Copy any attack results
        for pattern in [f"adaptive_{model}*defended*", f"advanced_{model}*defended*",
                        f"fresh_eval_{config}*"]:
            for atk_f in glob.glob(os.path.join(OUTDIR, pattern)):
                shutil.copy2(atk_f, os.path.join(config_dir, os.path.basename(atk_f)))

        # 7. Save summary JSON
        summary = {
            "config": config,
            "model": model,
            "model_id": row["model_id"],
            "anchor_id": hp.get("anchor_id", ""),
            "adapter_path": adapter_path,
            "asr": {"self": row["asr_self"], "anchor": row["asr_anchor"], "other": row["asr_other"]},
            "bgr": row["bgr"],
            "benchmarks": {
                "xstest": row.get("xstest", ""),
                "orbench": row.get("orbench", ""),
                "mtbench": row.get("mtbench", ""),
                "mmlu": row.get("mmlu", ""),
            },
            "hyperparameters": hp,
        }
        with open(os.path.join(config_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        saved_count += 1

    print(f"  Saved {saved_count} working configs to {working_dir}")

    # =========================================================================
    # PRINT SUMMARY TABLES
    # =========================================================================
    print("\n" + "=" * 120)
    print("TABLE: BASELINES")
    print("=" * 120)
    print(f"{'Model':<15} {'Model ID':<45} {'ASR(s/a/o)':<18} {'BGR':<6} {'XS':<8} {'OR':<8} {'MT':<8} {'MMLU':<8}")
    print("-" * 120)
    for r in baseline_rows:
        asr_str = f"{r['asr_self']:.0f}/{r['asr_anchor']:.0f}/{r['asr_other']:.0f}%"
        bgr_str = f"{r['bgr']:.0f}%" if r['bgr'] is not None else "-"
        xs_str = f"{r['xstest']:.1f}%" if r.get('xstest') else "-"
        or_str = f"{r['orbench']:.1f}%" if r.get('orbench') else "-"
        mt_str = f"{r['mtbench']:.2f}" if r.get('mtbench') else "-"
        mmlu_str = f"{r['mmlu']:.1f}%" if r.get('mmlu') else "-"
        print(f"{r['model']:<15} {r['model_id']:<45} {asr_str:<18} {bgr_str:<6} {xs_str:<8} {or_str:<8} {mt_str:<8} {mmlu_str:<8}")

    print("\n" + "=" * 160)
    print("TABLE: ALL ABLATIONS (WildGuard judge, sorted by model then ASR)")
    print("=" * 160)
    header = (f"{'Config':<30} {'Model':<10} {'Anchor':<20} {'γ':<5} {'ε':<5} {'δ':<6} "
              f"{'Scope':<8} {'Steps':<6} {'BL':<4} "
              f"{'ASR(s/a/o)':<16} {'ΔASR(s/a/o)':<18} {'BGR':<5} "
              f"{'XS':<7} {'OR':<7} {'MT':<6} {'MMLU':<6} {'Adapter':<30}")
    print(header)
    print("-" * 160)

    # Sort by model, then max ASR
    rows_sorted = sorted(rows, key=lambda r: (
        r["model"],
        max(r["asr_self"] or 999, r["asr_anchor"] or 999, r["asr_other"] or 999)
    ))

    for r in rows_sorted:
        asr_str = f"{r['asr_self']:.0f}/{r['asr_anchor']:.0f}/{r['asr_other']:.0f}%"
        d_s = r.get("delta_asr_self", "")
        d_a = r.get("delta_asr_anchor", "")
        d_o = r.get("delta_asr_other", "")
        delta_str = f"{d_s:+.0f}/{d_a:+.0f}/{d_o:+.0f}" if d_s != "" else "-"
        bgr_str = f"{r['bgr']:.0f}%" if r['bgr'] is not None else "-"
        xs_str = f"{r['xstest']:.1f}%" if r.get('xstest') and r['xstest'] != "" else "-"
        or_str = f"{r['orbench']:.1f}%" if r.get('orbench') and r['orbench'] != "" else "-"
        mt_str = f"{r['mtbench']:.2f}" if r.get('mtbench') and r['mtbench'] != "" else "-"
        mmlu_str = f"{r['mmlu']:.1f}%" if r.get('mmlu') and r['mmlu'] != "" else "-"
        gamma_str = str(r['gamma_cka']) if r['gamma_cka'] != "" else "-"
        eps_str = str(r['epsilon_kl']) if r['epsilon_kl'] != "" else "-"
        delta_lm_str = str(r['delta_lm']) if r['delta_lm'] != "" else "-"
        scope_str = str(r['cka_scope']) if r['cka_scope'] != "" else "-"
        steps_str = str(r['steps']) if r['steps'] != "" else "-"
        bl_str = "Y" if r.get('use_borderline') else "N" if r.get('use_borderline') is False else "-"
        anchor_str = str(r['anchor_id'])[:20] if r['anchor_id'] else "-"
        adapter_str = r['adapter_path'][:30] if r['adapter_path'] else "-"

        print(f"{r['config']:<30} {r['model']:<10} {anchor_str:<20} {gamma_str:<5} {eps_str:<5} {delta_lm_str:<6} "
              f"{scope_str:<8} {steps_str:<6} {bl_str:<4} "
              f"{asr_str:<16} {delta_str:<18} {bgr_str:<5} "
              f"{xs_str:<7} {or_str:<7} {mt_str:<6} {mmlu_str:<6} {adapter_str:<30}")

    print(f"\nTotal: {len(rows)} configs")

    print("\n" + "=" * 160)
    print("TABLE: BEST CONFIGS (max ASR ≤ 20%, BGR ≤ 5%)")
    print("=" * 160)
    print(header)
    print("-" * 160)

    for r in best_rows:
        asr_str = f"{r['asr_self']:.0f}/{r['asr_anchor']:.0f}/{r['asr_other']:.0f}%"
        d_s = r.get("delta_asr_self", "")
        d_a = r.get("delta_asr_anchor", "")
        d_o = r.get("delta_asr_other", "")
        delta_str = f"{d_s:+.0f}/{d_a:+.0f}/{d_o:+.0f}" if d_s != "" else "-"
        bgr_str = f"{r['bgr']:.0f}%" if r['bgr'] is not None else "-"
        xs_str = f"{r['xstest']:.1f}%" if r.get('xstest') and r['xstest'] != "" else "-"
        or_str = f"{r['orbench']:.1f}%" if r.get('orbench') and r['orbench'] != "" else "-"
        mt_str = f"{r['mtbench']:.2f}" if r.get('mtbench') and r['mtbench'] != "" else "-"
        mmlu_str = f"{r['mmlu']:.1f}%" if r.get('mmlu') and r['mmlu'] != "" else "-"
        gamma_str = str(r['gamma_cka']) if r['gamma_cka'] != "" else "-"
        eps_str = str(r['epsilon_kl']) if r['epsilon_kl'] != "" else "-"
        delta_lm_str = str(r['delta_lm']) if r['delta_lm'] != "" else "-"
        scope_str = str(r['cka_scope']) if r['cka_scope'] != "" else "-"
        steps_str = str(r['steps']) if r['steps'] != "" else "-"
        bl_str = "Y" if r.get('use_borderline') else "N" if r.get('use_borderline') is False else "-"
        anchor_str = str(r['anchor_id'])[:20] if r['anchor_id'] else "-"
        adapter_str = r['adapter_path'][:30] if r['adapter_path'] else "-"

        print(f"{r['config']:<30} {r['model']:<10} {anchor_str:<20} {gamma_str:<5} {eps_str:<5} {delta_lm_str:<6} "
              f"{scope_str:<8} {steps_str:<6} {bl_str:<4} "
              f"{asr_str:<16} {delta_str:<18} {bgr_str:<5} "
              f"{xs_str:<7} {or_str:<7} {mt_str:<6} {mmlu_str:<6} {adapter_str:<30}")

    print(f"\nTotal best: {len(best_rows)} configs")

    # =========================================================================
    # SAVE model-grouped summary
    # =========================================================================
    model_summary = defaultdict(list)
    for r in best_rows:
        model_summary[r["model"]].append(r)

    summary_path = os.path.join(SAVE_DIR, "best_configs_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "generated": "2026-03-10",
            "judge": "allenai/wildguard",
            "selection_criteria": "max(asr_self, asr_anchor, asr_other) <= 20% AND bgr <= 5%",
            "total_configs_tested": len(rows),
            "total_passing": len(best_rows),
            "models": {
                model: {
                    "count": len(configs),
                    "configs": [c["config"] for c in configs]
                }
                for model, configs in model_summary.items()
            },
            "baselines": {r["model"]: {k: v for k, v in r.items() if k != "model_id"} for r in baseline_rows},
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
