#!/bin/bash
#SBATCH --job-name=bench_bl2
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_baselines2_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_baselines2_%j.err
# ==========================================================
# Missing baselines: Llama-3-8B, Mistral-Nemo 12B, Starling-7B
# Vicuna MMLU rerun (fp16 — OOM'd with fp32)
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

MODELS=(
    "llama3:fp16:bench_llama3_8b_baseline"
    "mistral_nemo:fp16:bench_nemo_12b_baseline"
    "starling:fp32:bench_starling7b_baseline"
)

for M in "${MODELS[@]}"; do
    IFS=':' read -r defender precision label <<< "$M"

    echo ""
    echo "============================================================"
    echo "  BASELINE: $label ($precision)"
    echo "============================================================"
    python $BENCH_SCRIPT \
        --defender $defender \
        --precision $precision \
        --baseline_only \
        --output_json $OUTDIR/${label}.json \
        --verbose \
        2>&1 | tee $OUTDIR/${label}.log

    python3 -c "
import json
d = json.load(open('$OUTDIR/${label}.json'))
dd = d.get('baseline', d.get('defended', {}))
xs = dd.get('xstest_refusal_rate', 0)*100
orb = dd.get('orbench_refusal_rate', 0)*100
mmlu = dd.get('mmlu_accuracy', 0)*100
mt = dd.get('mtbench_score', 0)
print(f'  $label: XS={xs:.1f}% OR={orb:.1f}% MMLU={mmlu:.1f}% MT={mt:.2f}')
" 2>/dev/null || echo "  Failed to parse $label"
done

echo ""
echo "============================================================"
echo "  ALL BASELINES COMPLETE"
echo "============================================================"
