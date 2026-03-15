#!/bin/bash
#SBATCH --job-name=bench_sel
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_selected_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_selected_%j.err
# ==========================================================
# Benchmark selected configs: llama2 f5b, mistral f4a, mistral f4c
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Llama-2 f5b
echo ""
echo "============================================================"
echo "  BENCHMARK: llama2_7b_f5b"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender llama2 \
    --adapter_path "$OUTDIR/defender_v2_cka_20260304_202359" \
    --precision fp32 --no_baseline \
    --exclude_train_prompts \
    --output_json $OUTDIR/bench_llama2_7b_f5b.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_llama2_7b_f5b.log

# Mistral f4a
echo ""
echo "============================================================"
echo "  BENCHMARK: mistral7b_f4a"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender mistral \
    --adapter_path "$OUTDIR/defender_v2_cka_20260304_195813" \
    --precision fp32 --no_baseline \
    --exclude_train_prompts \
    --output_json $OUTDIR/bench_mistral7b_f4a.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_mistral7b_f4a.log

# Mistral f4c
echo ""
echo "============================================================"
echo "  BENCHMARK: mistral7b_f4c"
echo "============================================================"
python $BENCH_SCRIPT \
    --defender mistral \
    --adapter_path "$OUTDIR/defender_v2_cka_20260304_205111" \
    --precision fp32 --no_baseline \
    --exclude_train_prompts \
    --output_json $OUTDIR/bench_mistral7b_f4c.json \
    --verbose \
    2>&1 | tee $OUTDIR/bench_mistral7b_f4c.log

echo ""
echo "============================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "============================================================"
echo ""
for label in llama2_7b_f5b mistral7b_f4a mistral7b_f4c; do
    echo "--- $label ---"
    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
def pct(v): return v*100 if v<=1 else v
mmlu = pct(d.get('mmlu', d.get('mmlu_accuracy', 0)))
mt = d.get('mt_bench', d.get('mt_bench_score', 0))
orb = pct(d.get('or_bench', d.get('or_bench_refusal', 0)))
xs = pct(d.get('xstest', d.get('xstest_refusal', 0)))
print(f'  MMLU: {mmlu:.1f}%  MT-Bench: {mt:.2f}  OR-Bench: {orb:.1f}%  XSTest: {xs:.1f}%')
" 2>/dev/null || echo "  Failed to parse"
done
