#!/bin/bash
#SBATCH --job-name=bench_yc_nc
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_yc_nc_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_yc_nc_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

CONFIGS=(
    "yi9b:defender_v2_cka_20260306_103341:yi9b_yc_a"
    "yi9b:defender_v2_cka_20260306_104827:yi9b_yc_b"
    "yi9b:defender_v2_cka_20260306_110214:yi9b_yc_c"
    "yi9b:defender_v2_cka_20260306_111558:yi9b_yc_d"
    "yi9b:defender_v2_cka_20260306_113022:yi9b_yc_e"
    "yi9b:defender_v2_cka_20260306_114429:yi9b_yc_f"
    "mistral_nemo:defender_v2_cka_20260306_115805:nemo_nc_a"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r defender adapter label <<< "$CONFIG"
    ADAPTER_PATH="$OUTDIR/$adapter"

    if [ -f "$OUTDIR/bench_${label}.json" ]; then
        echo "[SKIP] bench_${label}.json already exists"
        continue
    fi

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "[ERROR] Adapter not found: $ADAPTER_PATH"
        continue
    fi

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
echo "  YC/NC BENCHMARKS COMPLETE"
echo "============================================================"
