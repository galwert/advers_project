#!/bin/bash
#SBATCH --job-name=final_5
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_final_5models_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_final_5models_%j.err
# ==========================================================
# Final evaluation of 5 best models
# Saves: full ASR responses, baseline + defended metrics
# ==========================================================
# FILL IN: Replace placeholders with actual best configs after
# benchmark results come in.

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

EVAL_SCRIPT="evaluate_v2.py"
OUTDIR="./7b_defense_wildguard_outputs"
SAVEDIR="./saved_results/final_5models"
GCG_DATA="../outputs/advbench_suffixes_all_models_fixed.csv"

mkdir -p "$SAVEDIR"

# ================================================================
# FILL THESE IN with the 5 best configs after benchmarks complete
# Format: defender:anchor:adapter_dir:label:precision
# ================================================================
CONFIGS=(
    # 7B models (at least 2)
    "PLACEHOLDER_7B_1"
    "PLACEHOLDER_7B_2"
    # 14B models (at least 2)
    "PLACEHOLDER_14B_1"
    "PLACEHOLDER_14B_2"
    # 5th model
    "PLACEHOLDER_5TH"
)

# Example configs (uncomment and fill when ready):
# "vicuna:llama2:defender_v2_cka_20260305_001243:vicuna7b_f5b:fp16"
# "llama3:qwen:defender_v2_cka_20260304_102254:llama3_8b_r1a:fp16"
# "qwen:llama3:defender_v2_cka_20260305_104300:qwen7b_o6c:fp16"
# "yi9b:mistral:defender_v2_cka_20260305_115819:yi9b_xs_b:fp16"
# "mistral_nemo:qwen:defender_v2_cka_20260305_112301:nemo_12b_e:fp16"

for CONFIG in "${CONFIGS[@]}"; do
    if [[ "$CONFIG" == PLACEHOLDER* ]]; then
        echo "[!] SKIP placeholder: $CONFIG"
        continue
    fi

    IFS=':' read -r defender anchor adapter label precision <<< "$CONFIG"
    ADAPTER_PATH="$OUTDIR/$adapter"

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "[!] SKIP $label — adapter not found: $ADAPTER_PATH"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  FINAL EVAL: $label"
    echo "  defender=$defender anchor=$anchor precision=$precision"
    echo "  adapter=$ADAPTER_PATH"
    echo "============================================================"

    # Run full eval with --baseline and --save_all_responses
    python $EVAL_SCRIPT \
        --adapter_path "$ADAPTER_PATH" \
        --defender $defender --anchor $anchor \
        --precision $precision --cka_per_group --verbose --baseline \
        --output_json "$SAVEDIR/eval_${label}.json" \
        --save_all_responses "$SAVEDIR/responses_${label}.json" \
        2>&1 | tee "$SAVEDIR/eval_${label}.log"

    # Copy adapter
    echo "[+] Copying adapter to $SAVEDIR/${label}_adapter/"
    mkdir -p "$SAVEDIR/${label}_adapter"
    cp "$ADAPTER_PATH"/adapter_config.json "$SAVEDIR/${label}_adapter/" 2>/dev/null
    cp "$ADAPTER_PATH"/adapter_model.safetensors "$SAVEDIR/${label}_adapter/" 2>/dev/null
    cp "$ADAPTER_PATH"/borderline_train_prompts.json "$SAVEDIR/${label}_adapter/" 2>/dev/null

    echo "[+] $label done"
done

# ================================================================
# Generate final summary JSON
# ================================================================
echo ""
echo "============================================================"
echo "  GENERATING FINAL SUMMARY"
echo "============================================================"

python3 << 'PYEOF'
import json, glob, os

savedir = "./saved_results/final_5models"
summary = {"models": {}}

# Collect all baselines
baselines = {}
for f in glob.glob("./7b_defense_wildguard_outputs/bench_*baseline*.json"):
    label = os.path.basename(f).replace("bench_", "").replace(".json", "")
    try:
        d = json.load(open(f))
        dd = d.get('baseline', d.get('defended', {}))
        baselines[label] = dd
    except: pass

# Collect final eval results
for f in sorted(glob.glob(f"{savedir}/eval_*.json")):
    label = os.path.basename(f).replace("eval_", "").replace(".json", "")
    try:
        d = json.load(open(f))
        model_entry = {
            "defended_metrics": d.get("defended", {}),
            "baseline_metrics": d.get("baseline", {}),
        }

        # Also pull bench results if they exist
        bench_f = f"./7b_defense_wildguard_outputs/bench_{label}.json"
        if os.path.exists(bench_f):
            b = json.load(open(bench_f))
            bd = b.get('defended', b)
            model_entry["benchmark"] = {
                "xstest_refusal_rate": bd.get("xstest_refusal_rate", None),
                "orbench_refusal_rate": bd.get("orbench_refusal_rate", None),
                "mmlu_accuracy": bd.get("mmlu_accuracy", None),
                "mtbench_score": bd.get("mtbench_score", None),
            }

        model_entry["has_full_responses"] = os.path.exists(f"{savedir}/responses_{label}.json")
        summary["models"][label] = model_entry
    except Exception as e:
        print(f"  Error processing {label}: {e}")

summary["baselines"] = baselines

with open(f"{savedir}/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[+] Final summary: {savedir}/final_summary.json")

# Print table
print(f"\n{'Model':<25} {'ASR_s':>6} {'ASR_a':>6} {'ASR_o':>6} {'BGR':>6} {'PPL':>6} {'XS':>6} {'OR':>6} {'MMLU':>6} {'MT':>6}")
print("-" * 100)
for label, m in summary["models"].items():
    dm = m.get("defended_metrics", {})
    bm = m.get("benchmark", {})
    print(f"{label:<25} "
          f"{dm.get('asr_self',0)*100:>5.1f}% "
          f"{dm.get('asr_anchor',0)*100:>5.1f}% "
          f"{dm.get('asr_other',0)*100:>5.1f}% "
          f"{dm.get('bgr',0)*100:>5.1f}% "
          f"{dm.get('ppl',0):>6.2f} "
          f"{(bm.get('xstest_refusal_rate') or 0)*100:>5.1f}% "
          f"{(bm.get('orbench_refusal_rate') or 0)*100:>5.1f}% "
          f"{(bm.get('mmlu_accuracy') or 0)*100:>5.1f}% "
          f"{bm.get('mtbench_score', 0):>5.2f}")
PYEOF

echo ""
echo "============================================================"
echo "  DONE — all results in $SAVEDIR/"
echo "============================================================"
ls -la "$SAVEDIR/"
