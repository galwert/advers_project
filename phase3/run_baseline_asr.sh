#!/bin/bash
#SBATCH --job-name=baseline_asr
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_baseline_asr_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_baseline_asr_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs"

# Qwen-7B baseline
echo "===== QWEN-7B BASELINE ====="
python evaluate_v2.py \
    --defender qwen --anchor llama2 \
    --adapter_path none --baseline \
    --output_json ${OUTDIR}/eval_qwen7b_baseline_v2.json \
    --low_memory \
    2>&1 | tee train_qwen7b_baseline_v2.log

# Vicuna-7B baseline
echo "===== VICUNA-7B BASELINE ====="
python evaluate_v2.py \
    --defender vicuna --anchor llama2 \
    --adapter_path none --baseline \
    --output_json ${OUTDIR}/eval_vicuna7b_baseline_v2.json \
    --low_memory \
    2>&1 | tee train_vicuna7b_baseline_v2.log

# Llama-3-8B baseline
echo "===== LLAMA-3-8B BASELINE ====="
python evaluate_v2.py \
    --defender llama3 --anchor llama2 \
    --adapter_path none --baseline \
    --output_json ${OUTDIR}/eval_llama3_8b_baseline_v2.json \
    --low_memory \
    2>&1 | tee train_llama3_8b_baseline_v2.log

# Yi-9B baseline
echo "===== YI-9B BASELINE ====="
python evaluate_v2.py \
    --defender yi9b --anchor llama2 \
    --adapter_path none --baseline \
    --output_json ${OUTDIR}/eval_yi9b_baseline_v2.json \
    --low_memory \
    2>&1 | tee train_yi9b_baseline_v2.log

# Nemo-12B baseline
echo "===== NEMO-12B BASELINE ====="
python evaluate_v2.py \
    --defender mistral_nemo --anchor llama2 \
    --adapter_path none --baseline \
    --output_json ${OUTDIR}/eval_nemo_12b_baseline_v2.json \
    --low_memory \
    2>&1 | tee train_nemo_12b_baseline_v2.log

echo "===== ALL BASELINES COMPLETE ====="
