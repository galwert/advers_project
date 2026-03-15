#!/bin/bash
#SBATCH --job-name=mist_yi2
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_mist_yi2_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_mist_yi2_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

DEFENDER="mistralai/Mistral-7B-Instruct-v0.2"
ANCHOR="yi9b"
CACHE_FILE="$OUTDIR/anchor_cache_yi9b_layer05.pt"

COMMON_BASE="--alignment cka --use_borderline --precision fp32 \
  --lora_r 32 --stage2_lr 2e-4 \
  --target_layer_pct 0.5 \
  --gcg_data_path ../outputs/advbench_suffixes_all_models_fixed.csv \
  --output_dir $OUTDIR"

# Step 0: precompute Yi-9B anchor cache
if [ ! -f "$CACHE_FILE" ]; then
    echo "=== Precomputing Yi-9B anchor cache ==="
    python $TRAIN_SCRIPT \
        --defender $DEFENDER --anchor $ANCHOR \
        --anchor_precision fp16 \
        --cka_scope all \
        --gamma 1.0 --alpha 0.8 --beta 1.0 --epsilon 0.15 --delta 0.04 \
        --stage2_steps 600 \
        $COMMON_BASE \
        --save_anchor_cache "$CACHE_FILE"
fi

COMMON="$COMMON_BASE --load_anchor_cache $CACHE_FILE"

declare -A ADAPTER_MAP

do_train() {
    local label="$1"
    shift
    echo ""
    echo "========================================"
    echo "  TRAIN: $label"
    echo "========================================"
    BEFORE=$(ls -d $OUTDIR/defender_v2_cka_* 2>/dev/null | sort)
    python $TRAIN_SCRIPT --defender $DEFENDER --anchor $ANCHOR $COMMON "$@"
    AFTER=$(ls -d $OUTDIR/defender_v2_cka_* 2>/dev/null | sort)
    ADAPTER=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | tail -1)
    if [ -z "$ADAPTER" ] || [ ! -f "$ADAPTER/adapter_config.json" ]; then
        echo "[!] Training failed for $label"
        return 1
    fi
    ADAPTER_MAP[$label]="$ADAPTER"
    echo "[+] $label adapter: $ADAPTER"
}

do_eval() {
    local label="$1"
    local adapter="${ADAPTER_MAP[$label]}"
    [ -z "$adapter" ] && echo "[SKIP] $label" && return 1
    echo ""
    echo "========================================"
    echo "  EVAL+BENCH: $label"
    echo "========================================"

    python $EVAL_SCRIPT \
        --adapter_path "$adapter" \
        --defender mistral --anchor yi9b \
        --output_json "$OUTDIR/eval_yi2_${label}.json" \
        --save_all_responses "$OUTDIR/responses_yi2_${label}.json" \
        --precision fp32 --low_memory

    python $BENCH_SCRIPT \
        --defender mistral \
        --adapter_path "$adapter" \
        --output_json "$OUTDIR/bench_yi2_${label}.json" \
        --no_baseline --skip_mmlu --low_memory

    echo "[+] Done: $label"
}

echo "====================================================="
echo "  Mistral + Yi-9B anchor — fine-tuning around best config"
echo "  Base: mistral_anc_y_ax (γ=1.0 α=0.8 δ=0.04 ε=0.15 scope=all steps=600)"
echo "  Result: ASR=0/0/0, XS=3.4%, OR=11.2%, MT=6.29 (Δ=-0.09)"
echo "====================================================="

# Run 1: More preservation — higher ε, slightly lower γ
# Idea: stronger KL keeps output distribution closer to baseline → better MT
do_train "yi_preserve" \
    --cka_scope all --stage2_steps 600 \
    --gamma 0.8 --alpha 0.8 --beta 1.0 --epsilon 0.25 --delta 0.06

# Run 2: Shorter training — 400 steps instead of 600
# Idea: less training = less drift from baseline, already 0% ASR so maybe we can afford it
do_train "yi_short" \
    --cka_scope all --stage2_steps 400 \
    --gamma 1.0 --alpha 0.8 --beta 1.0 --epsilon 0.15 --delta 0.04

echo ""
echo "=== EVALUATION ==="
do_eval "yi_preserve"
do_eval "yi_short"

echo ""
echo "====================================================="
echo "  SUMMARY"
echo "====================================================="
python3 -c "
import json, os
OUTDIR = './7b_defense_wildguard_outputs'
BXS, BOR, BMT = 0.08, 0.221, 6.38
for label in ['yi_preserve', 'yi_short']:
    ef = f'{OUTDIR}/eval_yi2_{label}.json'
    bf = f'{OUTDIR}/bench_yi2_{label}.json'
    if not os.path.exists(ef): continue
    d = json.load(open(ef)).get('defended', json.load(open(ef)))
    s,a,o = d.get('asr_self',0), d.get('asr_anchor',0), d.get('asr_other',0)
    bgr = d.get('bgr',0)
    xs = orb = mt = '—'
    if os.path.exists(bf):
        bd = json.load(open(bf)).get('defended', json.load(open(bf)))
        if 'xstest_refusal_rate' in bd:
            v = bd['xstest_refusal_rate']; xs = f'{v*100:.1f}%({(v-BXS)*100:+.1f})'
        if 'orbench_refusal_rate' in bd:
            v = bd['orbench_refusal_rate']; orb = f'{v*100:.1f}%({(v-BOR)*100:+.1f})'
        if 'mt_bench_score' in bd:
            v = bd['mt_bench_score']; mt = f'{v:.2f}({v-BMT:+.2f})'
    print(f'  {label:<20} ASR={s*100:.0f}/{a*100:.0f}/{o*100:.0f}  BGR={bgr*100:.1f}%  XS={xs}  OR={orb}  MT={mt}')
print()
print('  Reference (mistral_anc_y_ax): ASR=0/0/0  BGR=0%  XS=3.4%(-4.6)  OR=11.2%(-10.9)  MT=6.29(-0.09)')
"
