#!/bin/bash
#SBATCH --job-name=msw_bm
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_msw_bm_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_msw_bm_%j.err

# Bench configs that pass the relaxed BGR<=4.4% gate but don't have bench yet:
# mst_hr_g03_e12_d08: ASR=0% BGR=3.3% BRR=1.1%
# msw_g02_e01:        ASR=11% BGR=2.2% BRR=0.0%
# msw_g03_e02_a005:   ASR=7% BGR=2.2% BRR=0.0%
# msw_g05_e04:        ASR=10% BGR=2.2% BRR=0.0%
# msw_g05_e04_d02:    ASR=0% BGR=3.3% BRR=0.0%

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

run_bench() {
    local TAG=$1

    if [ -f "$OUTDIR/bench_${TAG}.json" ]; then
        echo "[SKIP] $TAG bench exists"; return
    fi

    ADAPTER=$(grep -oP 'Adapter saved to: \K.*' "$OUTDIR/train_${TAG}.log" 2>/dev/null | tail -1)
    if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
        echo "[ERROR] Cannot find adapter for $TAG"; return
    fi

    echo ""
    echo "============================================================"
    echo "  BENCH: $TAG (adapter: $ADAPTER)"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --adapter_path "$ADAPTER" --defender mistral --precision fp32 --no_baseline \
        --output_json $OUTDIR/bench_${TAG}.json --verbose \
        2>&1 | tee $OUTDIR/bench_${TAG}.log
}

run_bench "mst_hr_g03_e12_d08"
run_bench "msw_g02_e01"
run_bench "msw_g03_e02_a005"
run_bench "msw_g05_e04"
run_bench "msw_g05_e04_d02"

echo ""
echo "[+] All missing benches done"
