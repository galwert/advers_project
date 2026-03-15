#!/bin/bash
# 13B Defense Training — 4 runs on L40S GPUs
# Llama-2-13B (anchor: Qwen-7B) and Qwen-1.5-14B (anchor: Llama-3-8B)

set -e
cd /home/wertheizer/advers_project/phase3

CONDA_ENV="defense_env"
SCRIPT="two_stage_defense_v2.py"
OUTDIR="./13b_defense_outputs"
GCG_DATA="../outputs/advbench_suffixes_all_models_fixed.csv"

# Common args
COMMON="--alignment cka --cka_scope harmful_only --use_borderline --precision fp16 \
        --target_layer_pct 0.5 --lora_r 32 --stage2_steps 200 --stage2_lr 5e-5 \
        --gcg_data_path $GCG_DATA --output_dir $OUTDIR"

echo "=== Run 1: Llama-2-13B (γ=2.0) ==="
conda run --no-banner -n $CONDA_ENV python $SCRIPT \
    --defender meta-llama/Llama-2-13b-chat-hf \
    --anchor qwen \
    --gamma 2.0 --alpha 0.15 --beta 1.0 --epsilon 0.4 --delta 0 \
    $COMMON \
    2>&1 | tee $OUTDIR/llama2_13b_g2.0.log

echo "=== Run 2: Llama-2-13B (γ=1.0) ==="
conda run --no-banner -n $CONDA_ENV python $SCRIPT \
    --defender meta-llama/Llama-2-13b-chat-hf \
    --anchor qwen \
    --gamma 1.0 --alpha 0.15 --beta 1.0 --epsilon 1.0 --delta 0 \
    $COMMON \
    2>&1 | tee $OUTDIR/llama2_13b_g1.0.log

echo "=== Run 3: Qwen-1.5-14B (γ=3.0) ==="
conda run --no-banner -n $CONDA_ENV python $SCRIPT \
    --defender Qwen/Qwen1.5-14B-Chat \
    --anchor llama3 \
    --gamma 3.0 --alpha 0.15 --beta 1.0 --epsilon 0.4 --delta 0 \
    $COMMON \
    2>&1 | tee $OUTDIR/qwen_14b_g3.0.log

echo "=== Run 4: Qwen-1.5-14B (γ=1.5) ==="
conda run --no-banner -n $CONDA_ENV python $SCRIPT \
    --defender Qwen/Qwen1.5-14B-Chat \
    --anchor llama3 \
    --gamma 1.5 --alpha 0.15 --beta 1.0 --epsilon 0.4 --delta 0 \
    $COMMON \
    2>&1 | tee $OUTDIR/qwen_14b_g1.5.log

echo "=== All 13B defense runs complete ==="
