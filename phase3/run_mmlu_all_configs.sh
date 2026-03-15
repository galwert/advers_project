#!/bin/bash
#SBATCH --job-name=mmlu_all
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_mmlu_all_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_mmlu_all_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs/mmlu_results"
ARCHIVE="./saved_results/final_archive"
mkdir -p "$OUTDIR"

python3 << 'PYEOF'
import os, sys, json, subprocess, tempfile

OUTDIR = "./7b_defense_wildguard_outputs/mmlu_results"
ARCHIVE = "./saved_results/final_archive"

# All configs that need MMLU (model_id, adapter_path, config_name)
CONFIGS = [
    # Best balanced (missing MMLU)
    ("mistralai/Mistral-7B-Instruct-v0.2", f"{ARCHIVE}/adapters/mistral_anc_y_ax", "mistral_anc_y_ax"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", f"{ARCHIVE}/adapters/llama3_lx_d", "llama3_lx_d"),
    ("01-ai/Yi-1.5-9B-Chat", f"{ARCHIVE}/adapters/yi9b_ya_a", "yi9b_ya_a"),
    ("mistralai/Mistral-Nemo-Instruct-2407", f"{ARCHIVE}/adapters/nemo_ab2_ax", "nemo_ab2_ax"),
    ("Qwen/Qwen1.5-14B-Chat", f"{ARCHIVE}/adapters/qwen14b_q14_b", "qwen14b_q14_b"),
    # 3 Mistral operating points
    ("mistralai/Mistral-7B-Instruct-v0.2", f"{ARCHIVE}/adapters/mistral7b_abl_weak", "mistral7b_abl_weak"),
    ("mistralai/Mistral-7B-Instruct-v0.2", f"{ARCHIVE}/adapters/mistral_anc_q_ax", "mistral_anc_q_ax"),
    ("mistralai/Mistral-7B-Instruct-v0.2", f"{ARCHIVE}/adapters/mistral_anc_l3_ax", "mistral_anc_l3_ax"),
]

# Also run missing baselines (nemo, qwen14b, vicuna don't have baseline MMLU)
BASELINES = [
    ("mistralai/Mistral-Nemo-Instruct-2407", "nemo_baseline"),
    ("Qwen/Qwen1.5-14B-Chat", "qwen14b_baseline"),
    ("lmsys/vicuna-7b-v1.5", "vicuna_baseline"),
]


def run_mmlu(model_id, adapter_path=None, tag="eval"):
    """Run MMLU via lm-eval-harness. Returns accuracy or None."""
    model_args = [
        f"pretrained={model_id}",
        "dtype=float16",
        "trust_remote_code=True",
    ]
    if adapter_path and os.path.exists(adapter_path):
        model_args.append(f"peft={adapter_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", ",".join(model_args),
            "--tasks", "mmlu",
            "--num_fewshot", "5",
            "--batch_size", "auto:4",
            "--output_path", tmpdir,
        ]

        print(f"\n--- MMLU 5-shot ({tag}) ---")
        print(f"[*] Model: {model_id}")
        if adapter_path:
            print(f"[*] Adapter: {adapter_path}")
        print(f"[*] Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            print(f"[!] MMLU failed (rc={result.returncode})")
            print(f"    stderr (last 500): {result.stderr[-500:]}")
            return None

        results_json = None
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.startswith("results") and f.endswith(".json"):
                    results_json = os.path.join(root, f)
                    break
            if results_json:
                break

        if not results_json:
            print("[!] No results JSON found")
            return None

        with open(results_json) as f:
            lm_results = json.load(f)

        results_dict = lm_results.get("results", {})
        if "mmlu" in results_dict:
            acc = results_dict["mmlu"].get("acc,none", results_dict["mmlu"].get("acc"))
        else:
            accs = []
            for task_name, task_results in results_dict.items():
                if "mmlu" in task_name:
                    a = task_results.get("acc,none", task_results.get("acc"))
                    if a is not None:
                        accs.append(a)
            acc = sum(accs) / len(accs) if accs else None

        if acc is not None:
            print(f"MMLU Accuracy: {acc*100:.1f}%")
        return acc


all_results = {}

# Run missing baselines first
print("=" * 60)
print("  MISSING BASELINE MMLUs")
print("=" * 60)
for model_id, name in BASELINES:
    out_file = f"{OUTDIR}/mmlu_{name}.json"
    if os.path.exists(out_file):
        print(f"\n[SKIP] {out_file} exists")
        d = json.load(open(out_file))
        all_results[name] = d.get("mmlu_accuracy")
        continue

    acc = run_mmlu(model_id, adapter_path=None, tag=name)
    result = {"name": name, "model_id": model_id, "mmlu_accuracy": acc}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_file}")
    all_results[name] = acc

# Run defended configs
print("\n" + "=" * 60)
print("  DEFENDED CONFIG MMLUs")
print("=" * 60)
for model_id, adapter_path, name in CONFIGS:
    out_file = f"{OUTDIR}/mmlu_{name}.json"
    if os.path.exists(out_file):
        print(f"\n[SKIP] {out_file} exists")
        d = json.load(open(out_file))
        all_results[name] = d.get("mmlu_accuracy")
        continue

    acc = run_mmlu(model_id, adapter_path=adapter_path, tag=name)
    result = {"name": name, "model_id": model_id, "adapter_path": adapter_path, "mmlu_accuracy": acc}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_file}")
    all_results[name] = acc

# Summary
print("\n" + "=" * 60)
print("  MMLU RESULTS SUMMARY")
print("=" * 60)
print(f"\n  {'Config':<25} {'MMLU':>8}")
print(f"  {'-'*25} {'-'*8}")
for name, acc in all_results.items():
    if acc is not None:
        print(f"  {name:<25} {acc*100:>7.1f}%")
    else:
        print(f"  {name:<25}    FAIL")

PYEOF

echo ""
echo "[+] Done"
