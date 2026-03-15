#!/bin/bash
#SBATCH --job-name=bm_2
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bm_2_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bm_2_%j.err

# Bench missing configs: Yi9b + Qwen7b_f5a

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

run_bench() {
    local TAG=$1 DEFENDER=$2 ADAPTER=$3
    if [ -f "$OUTDIR/bench_${TAG}.json" ]; then
        echo "[SKIP] $TAG bench exists"; return
    fi
    if [ ! -d "$ADAPTER" ]; then
        echo "[ERROR] Adapter not found for $TAG: $ADAPTER"; return
    fi
    echo ""
    echo "============================================================"
    echo "  BENCH: $TAG (defender=$DEFENDER)"
    echo "============================================================"
    python $BENCH_SCRIPT \
        --adapter_path "$ADAPTER" --defender $DEFENDER --precision fp16 --no_baseline --skip_mmlu \
        --output_json $OUTDIR/bench_${TAG}.json --verbose \
        2>&1 | tee $OUTDIR/bench_${TAG}.log
}

run_bench "yi9b_q14_an" "yi9b" "$OUTDIR/defender_v2_cka_20260309_062605"
run_bench "yi9b_self_an" "yi9b" "$OUTDIR/defender_v2_cka_20260309_051811"
run_bench "yi9b_anc_q" "yi9b" "$OUTDIR/defender_v2_cka_20260308_121111"
run_bench "qwen7b_f5a" "qwen" "$OUTDIR/defender_v2_cka_20260304_203006"

echo "[+] bm_2 done"
