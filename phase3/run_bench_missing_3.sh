#!/bin/bash
#SBATCH --job-name=bm_3
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bm_3_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bm_3_%j.err

# Bench missing configs: Vicuna + Llama3

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

run_bench "vicuna_self_ax" "vicuna" "$OUTDIR/defender_v2_cka_20260309_020908"
run_bench "vicuna_anc_m" "vicuna" "$OUTDIR/defender_v2_cka_20260308_111433"
run_bench "llama3_anc_m" "llama3" "$OUTDIR/defender_v2_cka_20260308_115130"
run_bench "llama3_ld_d" "llama3" "$OUTDIR/defender_v2_cka_20260306_150228"

echo "[+] bm_3 done"
