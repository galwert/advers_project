#!/bin/bash
#SBATCH --job-name=bench_bl
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_baselines_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_baselines_%j.err
# ==========================================================
# Baseline benchmarks: Yi-9B and Qwen-7B (no adapter)
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Yi-9B baseline
echo ""
echo "============================================================"
echo "  BASELINE BENCHMARK: Yi-1.5-9B-Chat"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender yi9b \
    --precision fp16 \
    --baseline_only \
    --output_json $OUTDIR/bench_yi9b_baseline.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_yi9b_baseline.log

# Qwen baseline
echo ""
echo "============================================================"
echo "  BASELINE BENCHMARK: Qwen1.5-7B-Chat"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender qwen \
    --precision fp32 \
    --baseline_only \
    --output_json $OUTDIR/bench_qwen7b_baseline.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_qwen7b_baseline.log

echo ""
echo "============================================================"
echo "  BASELINES COMPLETE"
echo "============================================================"
for label in yi9b_baseline qwen7b_baseline; do
    echo "--- $label ---"
    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
# baseline runs store under 'baseline' key, or 'defended' if no adapter
dd = d.get('baseline', d.get('defended', {}))
print(f'  MMLU: {dd[\"mmlu_accuracy\"]*100:.1f}%  MT-Bench: {dd[\"mtbench_score\"]:.2f}  OR-Bench: {dd[\"orbench_refusal_rate\"]*100:.1f}%  XSTest: {dd[\"xstest_refusal_rate\"]*100:.1f}%')
" 2>/dev/null || echo "  Failed to parse"
done
