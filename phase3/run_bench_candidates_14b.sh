#!/bin/bash
#SBATCH --job-name=bench_14b
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_14b_cands_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_14b_cands_%j.err
# ==========================================================
# Benchmark 14B candidates (relaxed 20% ASR gate)
# Yi-9B (~9B) and Mistral-Nemo (~12B)
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Format: defender:adapter_dir:label
CONFIGS=(
    "yi9b:defender_v2_cka_20260305_115819:yi9b_xs_b"
    "yi9b:defender_v2_cka_20260305_011618:yi9b_c"
    "yi9b:defender_v2_cka_20260305_015705:yi9b_e"
    "mistral_nemo:defender_v2_cka_20260305_103052:nemo_12b_b"
    "mistral_nemo:defender_v2_cka_20260305_112301:nemo_12b_e"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r defender adapter label <<< "$CONFIG"
    ADAPTER_PATH="$OUTDIR/$adapter"

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "[!] SKIP $label — adapter not found: $ADAPTER_PATH"
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

    # Quick summary
    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
dd = d.get('defended', d)
xs = dd.get('xstest_refusal_rate', 0)*100
orb = dd.get('orbench_refusal_rate', 0)*100
mmlu = dd.get('mmlu_accuracy', 0)*100
mt = dd.get('mtbench_score', 0)
print(f'  RESULT $label: XS={xs:.1f}% OR={orb:.1f}% MMLU={mmlu:.1f}% MT={mt:.2f}')
" 2>/dev/null || echo "  Failed to parse $label"
done

echo ""
echo "============================================================"
echo "  14B CANDIDATE BENCHMARKS COMPLETE"
echo "============================================================"
for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r _ _ label <<< "$CONFIG"
    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
dd = d.get('defended', d)
xs = dd.get('xstest_refusal_rate', 0)*100
orb = dd.get('orbench_refusal_rate', 0)*100
mmlu = dd.get('mmlu_accuracy', 0)*100
mt = dd.get('mtbench_score', 0)
print(f'  $label: XS={xs:.1f}% OR={orb:.1f}% MMLU={mmlu:.1f}% MT={mt:.2f}')
" 2>/dev/null || echo "  $label: no results"
done
