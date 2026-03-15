#!/bin/bash
#SBATCH --job-name=bl_fp32
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bl_fp32_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bl_fp32_%j.err

# Re-run baselines that were originally at fp16. Must match defended (fp32).
# Llama3-8B, Yi-9B were fp16. Mistral needs verification.
# Nemo (12B) and Qwen14B stay fp16 — need A100 for fp32.

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Models to re-run at fp32 (all ≤9B, fit on L40)
MODELS=(
    "llama3:bench_llama3_8b_baseline_fp32"
    "yi9b:bench_yi9b_baseline_fp32"
    "mistral:bench_mistral_baseline_fp32"
)

for M in "${MODELS[@]}"; do
    IFS=':' read -r defender label <<< "$M"

    if [ -f "$OUTDIR/${label}.json" ]; then
        echo "[SKIP] $label already exists"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  BASELINE (fp32): $defender"
    echo "============================================================"
    python $BENCH_SCRIPT \
        --defender $defender \
        --precision fp32 \
        --baseline_only \
        --output_json $OUTDIR/${label}.json \
        --verbose \
        2>&1 | tee $OUTDIR/${label}.log
done

echo ""
echo "============================================================"
echo "  BASELINES (fp32) COMPLETE"
echo "============================================================"
for M in "${MODELS[@]}"; do
    IFS=':' read -r defender label <<< "$M"
    if [ -f "$OUTDIR/${label}.json" ]; then
        python3 -c "
import json
d = json.load(open('$OUTDIR/${label}.json'))
bl = d.get('baseline', {})
xs = bl.get('xstest_refusal_rate', 0)*100
orb = bl.get('orbench_refusal_rate', 0)*100
mt = bl.get('mtbench_score', 0)
mmlu = bl.get('mmlu_accuracy')
mmlu_s = f'{mmlu*100:.1f}%' if mmlu else 'N/A'
print(f'  $label: XS={xs:.1f}% OR={orb:.1f}% MT={mt:.2f} MMLU={mmlu_s}')
" 2>/dev/null
    else
        echo "  $label: MISSING"
    fi
done
