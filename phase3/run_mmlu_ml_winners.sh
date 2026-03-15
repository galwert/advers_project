#!/bin/bash
#SBATCH --job-name=mmlu_ml
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_mmlu_ml_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_mmlu_ml_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

run_mmlu() {
    local label="$1"
    local adapter="$2"

    echo ""
    echo "============================================================"
    echo "  MMLU: $label"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --defender mistral \
        --adapter_path "$adapter" \
        --output_json "$OUTDIR/mmlu_ml_${label}.json" \
        --no_baseline --skip_mtbench --low_memory

    if [ -f "$OUTDIR/mmlu_ml_${label}.json" ]; then
        python3 -c "
import json
d = json.load(open('$OUTDIR/mmlu_ml_${label}.json'))
dd = d.get('defended', d)
mmlu = dd.get('mmlu_accuracy', 'N/A')
print(f'  ${label}: MMLU={mmlu*100:.1f}%' if isinstance(mmlu, float) else f'  ${label}: MMLU={mmlu}')
"
    fi
}

echo "====================================================="
echo "  MMLU for multi-layer promising configs"
echo "  (5 from v1 ablation with 0/0/0 ASR)"
echo "====================================================="

# V1 promising configs
run_mmlu "single_0.625"   "$OUTDIR/defender_v2_cka_20260311_000818"
run_mmlu "concat_3L"      "$OUTDIR/defender_v2_cka_20260311_011722"
run_mmlu "concat_3L_wide" "$OUTDIR/defender_v2_cka_20260311_012612"
run_mmlu "concat_5L"      "$OUTDIR/defender_v2_cka_20260311_013507"
run_mmlu "weighted_5L"    "$OUTDIR/defender_v2_cka_20260311_010845"

echo ""
echo "[+] MMLU complete"
