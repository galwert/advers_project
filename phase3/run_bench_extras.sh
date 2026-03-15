#!/bin/bash
#SBATCH --job-name=bench_ext
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_ext_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_ext_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# VT-A (vicuna, α=0.05, harmful_only) ASR=4/6/6%
# VT-C (vicuna, α=0.05, multi-layer, harmful_only) ASR=4/6/7%
# cka_all (qwen, scope=all) ASR=7/1/4%
CONFIGS=(
    "vicuna:defender_v2_cka_20260305_220356:vicuna_vt_a"
    "vicuna:defender_v2_cka_20260305_223913:vicuna_vt_c"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r defender adapter label <<< "$CONFIG"
    ADAPTER_PATH="$OUTDIR/$adapter"

    echo ""
    echo "============================================================"
    echo "  BENCHMARK: $label (defender=$defender)"
    echo "  Adapter: $ADAPTER_PATH"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --defender $defender \
        --adapter_path "$ADAPTER_PATH" \
        --precision fp16 --no_baseline \
        --exclude_train_prompts \
        --output_json $OUTDIR/bench_${label}.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_${label}.log

    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
dd = d.get('defended', d)
xs=dd.get('xstest_refusal_rate',0)*100; orb=dd.get('orbench_refusal_rate',0)*100
mmlu=dd.get('mmlu_accuracy',0)*100; mt=dd.get('mtbench_score',0)
print(f'  $label: XS={xs:.1f}% OR={orb:.1f}% MMLU={mmlu:.1f}% MT={mt:.2f}')
" 2>/dev/null || echo "  Failed to parse $label"
done

echo ""
echo "============================================================"
echo "  EXTRA BENCHMARKS COMPLETE"
echo "============================================================"
