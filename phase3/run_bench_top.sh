#!/bin/bash
#SBATCH --job-name=bench_top
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_top_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_top_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Top unbenchmarked configs sorted by total ASR (most secure first)
# llama3_lb_d: 0/0/0% total=0 (benign_only, γ=3.0)
# llama3_lb_b: 1/0/0% total=1 (benign_only, γ=2.5)
# vicuna_vb_e: 1/2/2% total=5 (benign_only, γ=2.0 α=0.12)
# vicuna_vb_a: 1/2/3% total=6 (benign_only, γ=2.0 α=0.15)
# yi9b_ya_a:   3/0/0% total=3 (all scope, γ=1.5)
# yi9b_xs_e:   2/3/2% total=7 (harmful_only)
# nemo_na_b:   4/5/0% total=9 (all scope, γ=2.0)
# nemo_na_a:   8/4/3% total=15 (all scope, γ=1.5)
# vicuna7b_f5d: 0/0/1% total=1 (harmful_only)
# qwen7b_cka_all: 7/1/4% total=12 (all scope)
# qwen7b_f2a:  0/0/1% total=1

CONFIGS=(
    "llama3:defender_v2_cka_20260305_231628:llama3_lb_d"
    "llama3:defender_v2_cka_20260305_225154:llama3_lb_b"
    "vicuna:defender_v2_cka_20260306_013320:vicuna_vb_e"
    "vicuna:defender_v2_cka_20260306_003125:vicuna_vb_a"
    "yi9b:defender_v2_cka_20260306_021923:yi9b_ya_a"
    "yi9b:defender_v2_cka_20260305_132608:yi9b_xs_e"
    "mistral_nemo:defender_v2_cka_20260306_032947:nemo_na_b"
    "mistral_nemo:defender_v2_cka_20260306_030448:nemo_na_a"
    "vicuna:defender_v2_cka_20260305_004939:vicuna7b_f5d"
    "qwen:defender_v2_cka_20260304_112501:qwen7b_cka_all"
    "qwen:defender_v2_cka_20260304_130434:qwen7b_f2a"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r defender adapter label <<< "$CONFIG"
    ADAPTER_PATH="$OUTDIR/$adapter"

    # Skip if benchmark already exists
    if [ -f "$OUTDIR/bench_${label}.json" ]; then
        echo "[SKIP] bench_${label}.json already exists"
        continue
    fi

    # Verify adapter exists
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "[ERROR] Adapter not found: $ADAPTER_PATH"
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

    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
dd = d.get('defended', d)
xs=dd.get('xstest_refusal_rate',0)*100; orb=dd.get('orbench_refusal_rate',0)*100
mmlu=dd.get('mmlu_accuracy',0)*100; mt=dd.get('mtbench_score',0)
print(f'  $label: XS={xs:.1f}% OR={orb:.1f}% MMLU={mmlu:.1f}% MT={mt:.2f}')
" 2>/dev/null || echo "  Failed to parse $label"
done

echo ""
echo "============================================================"
echo "  TOP BENCHMARKS COMPLETE"
echo "============================================================"
