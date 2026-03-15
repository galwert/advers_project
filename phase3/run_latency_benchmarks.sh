#!/bin/bash
# Lightweight latency/memory benchmarks for all defended models
set -e
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
CONDA_ENV="defense_env"
OUTDIR="./latency_results"
mkdir -p $OUTDIR

echo "=== Mistral-7B latency benchmark (layer 0.5 adapter) ==="
conda run -n $CONDA_ENV python latency_benchmark.py \
    --model_id mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter_path ./layer_sweep_outputs/mistral_layer0.5/defender_v2_cka_20260223_193031 \
    --precision fp16 --n_prompts 20 --n_runs 3 \
    --output $OUTDIR/latency_mistral7b.json

echo "=== Llama-2-13B latency benchmark ==="
conda run -n $CONDA_ENV python latency_benchmark.py \
    --model_id NousResearch/Llama-2-13b-chat-hf \
    --adapter_path ./13b_defense_outputs/defender_v2_cka_20260301_194502 \
    --precision fp16 --n_prompts 20 --n_runs 3 \
    --output $OUTDIR/latency_llama13b.json

echo "=== Qwen-14B latency benchmark ==="
conda run -n $CONDA_ENV python latency_benchmark.py \
    --model_id Qwen/Qwen1.5-14B-Chat \
    --adapter_path ./13b_defense_outputs/defender_v2_cka_20260301_003436 \
    --precision fp16 --n_prompts 20 --n_runs 3 \
    --output $OUTDIR/latency_qwen14b.json

echo "=== All latency benchmarks complete ==="
