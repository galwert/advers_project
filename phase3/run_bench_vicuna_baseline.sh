#!/bin/bash
#SBATCH --job-name=bench_vbl
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_vicuna_bl_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_vicuna_bl_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

echo ""
echo "============================================================"
echo "  BASELINE BENCHMARK: Vicuna-7B-v1.5"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender vicuna \
    --precision fp32 \
    --baseline_only \
    --output_json $OUTDIR/bench_vicuna7b_baseline.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_vicuna7b_baseline.log

echo ""
echo "--- RESULTS ---"
python3 -c "
import json
d = json.load(open('$OUTDIR/bench_vicuna7b_baseline.json'))
dd = d.get('baseline', d.get('defended', {}))
print(f'  MMLU: {dd[\"mmlu_accuracy\"]*100:.1f}%  MT-Bench: {dd[\"mtbench_score\"]:.2f}  OR-Bench: {dd[\"orbench_refusal_rate\"]*100:.1f}%  XSTest: {dd[\"xstest_refusal_rate\"]*100:.1f}%')
" 2>/dev/null || echo "  Failed to parse"
