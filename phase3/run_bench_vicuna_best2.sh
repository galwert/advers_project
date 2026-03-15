#!/bin/bash
#SBATCH --job-name=bench_vic
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_vicuna_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_vicuna_%j.err
# ==========================================================
# Benchmark best 2 Vicuna configs: f5b and f5c
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Vicuna f5b (gamma=2.0, delta=0.03)
echo ""
echo "============================================================"
echo "  BENCHMARK: vicuna7b_f5b"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender vicuna \
    --adapter_path "$OUTDIR/defender_v2_cka_20260305_001243" \
    --precision fp32 --no_baseline \
    --exclude_train_prompts \
    --output_json $OUTDIR/bench_vicuna7b_f5b.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_vicuna7b_f5b.log

# Vicuna f5c (gamma=3.0, delta=0.03)
echo ""
echo "============================================================"
echo "  BENCHMARK: vicuna7b_f5c"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender vicuna \
    --adapter_path "$OUTDIR/defender_v2_cka_20260305_003059" \
    --precision fp32 --no_baseline \
    --exclude_train_prompts \
    --output_json $OUTDIR/bench_vicuna7b_f5c.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_vicuna7b_f5c.log

echo ""
echo "============================================================"
echo "  VICUNA BENCHMARKS COMPLETE"
echo "============================================================"
for label in vicuna7b_f5b vicuna7b_f5c; do
    echo "--- $label ---"
    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
dd = d['defended']
print(f'  MMLU: {dd[\"mmlu_accuracy\"]*100:.1f}%  MT-Bench: {dd[\"mtbench_score\"]:.2f}  OR-Bench: {dd[\"orbench_refusal_rate\"]*100:.1f}%  XSTest: {dd[\"xstest_refusal_rate\"]*100:.1f}%')
" 2>/dev/null || echo "  Failed to parse"
done
