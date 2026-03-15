#!/bin/bash
#SBATCH --job-name=bch_ml
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bch_ml_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bch_ml_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

run_bench() {
    local label="$1"
    local adapter="$2"

    echo ""
    echo "============================================================"
    echo "  BENCH: $label"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --defender mistral \
        --adapter_path "$adapter" \
        --output_json "$OUTDIR/bench_ml_${label}.json" \
        --no_baseline --skip_mmlu --low_memory

    if [ -f "$OUTDIR/bench_ml_${label}.json" ]; then
        python3 -c "
import json
d = json.load(open('$OUTDIR/bench_ml_${label}.json'))
dd = d.get('defended', d)
xs = dd.get('xstest_refusal_rate', 'N/A')
orb = dd.get('orbench_refusal_rate', 'N/A')
mt = dd.get('mt_bench_score', 'N/A')
xs_s = f'{xs*100:.1f}%' if isinstance(xs, float) else xs
orb_s = f'{orb*100:.1f}%' if isinstance(orb, float) else orb
mt_s = f'{mt:.2f}' if isinstance(mt, float) else mt
print(f'  ${label}: XS={xs_s} OR={orb_s} MT={mt_s}')
"
    fi
}

# 5 promising configs (0/0/0 ASR, BGR < 15%)
run_bench "single_0.625"   "$OUTDIR/defender_v2_cka_20260311_000818"
run_bench "concat_3L"      "$OUTDIR/defender_v2_cka_20260311_011722"
run_bench "concat_3L_wide" "$OUTDIR/defender_v2_cka_20260311_012612"
run_bench "concat_5L"      "$OUTDIR/defender_v2_cka_20260311_013507"
run_bench "weighted_5L"    "$OUTDIR/defender_v2_cka_20260311_010845"

echo ""
echo "============================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "============================================================"
