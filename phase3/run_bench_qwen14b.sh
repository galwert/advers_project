#!/bin/bash
#SBATCH --job-name=bench_q14b
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_q14b_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_q14b_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Qwen-14B baseline benchmark
if [ ! -f "$OUTDIR/bench_qwen14b_baseline_v2.json" ]; then
    echo ""
    echo "============================================================"
    echo "  BENCHMARK: qwen14b baseline"
    echo "============================================================"
    python $BENCH_SCRIPT \
        --defender qwen-14b \
        --precision fp16 \
        --output_json $OUTDIR/bench_qwen14b_baseline_v2.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_qwen14b_baseline_v2.log
fi

# Qwen-14B defended configs
CONFIGS=(
    "qwen14b_q14_a:defender_v2_cka_20260307_132358"
    "qwen14b_q14_b:defender_v2_cka_20260307_134259"
    "qwen14b_q14_c:defender_v2_cka_20260307_140824"
    "qwen14b_q14_d:defender_v2_cka_20260307_143243"
    "qwen14b_q14_e:defender_v2_cka_20260307_145044"
    "qwen14b_q14_f:defender_v2_cka_20260307_151944"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r label adapter <<< "$CONFIG"
    ADAPTER_PATH="$OUTDIR/$adapter"

    if [ -f "$OUTDIR/bench_${label}.json" ]; then
        echo "[SKIP] bench_${label}.json exists"
        continue
    fi

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "[ERROR] Adapter not found: $ADAPTER_PATH"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  BENCHMARK: $label"
    echo "  Adapter: $ADAPTER_PATH"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --defender qwen-14b \
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
mt=dd.get('mtbench_score',0)
print(f'  $label: XS={xs:.1f}% OR={orb:.1f}% MT={mt:.2f}')
" 2>/dev/null || echo "  Failed to parse $label"
done

echo ""
echo "============================================================"
echo "  QWEN-14B BENCHMARKS COMPLETE"
echo "============================================================"
