#!/bin/bash
#SBATCH --job-name=atk_7b
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_atk_7b_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_atk_7b_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs"
N_PROMPTS=50

# ============================================================
# 7B MODELS: Adaptive GCG + Advanced (Embedding PGD + PAIR)
# Models: qwen, vicuna, llama3
# For each: baseline + defended
# ============================================================

# --- QWEN-7B ---
QWEN_ADAPTER="$OUTDIR/defender_v2_cka_20260307_094710"

# Adaptive GCG baseline
if [ ! -f "$OUTDIR/adaptive_qwen_baseline.json" ]; then
    echo "=== Adaptive GCG: Qwen baseline ==="
    python adaptive_attack.py --model qwen --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_qwen_baseline.json" \
        2>&1 | tee "$OUTDIR/adaptive_qwen_baseline.log"
fi

# Adaptive GCG defended
if [ ! -f "$OUTDIR/adaptive_qwen_defended.json" ]; then
    echo "=== Adaptive GCG: Qwen defended (ab2_hx) ==="
    python adaptive_attack.py --model qwen --adapter_path "$QWEN_ADAPTER" \
        --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_qwen_defended.json" \
        2>&1 | tee "$OUTDIR/adaptive_qwen_defended.log"
fi

# Advanced attacks baseline
if [ ! -f "$OUTDIR/advanced_qwen_baseline.json" ]; then
    echo "=== Advanced: Qwen baseline ==="
    python advanced_attacks.py --model qwen --n_prompts $N_PROMPTS \
        --attacker mistral --attack both \
        --output "$OUTDIR/advanced_qwen_baseline.json" \
        2>&1 | tee "$OUTDIR/advanced_qwen_baseline.log"
fi

# Advanced attacks defended
if [ ! -f "$OUTDIR/advanced_qwen_defended.json" ]; then
    echo "=== Advanced: Qwen defended (ab2_hx) ==="
    python advanced_attacks.py --model qwen --adapter_path "$QWEN_ADAPTER" \
        --n_prompts $N_PROMPTS --attacker mistral --attack both \
        --output "$OUTDIR/advanced_qwen_defended.json" \
        2>&1 | tee "$OUTDIR/advanced_qwen_defended.log"
fi

# --- VICUNA-7B ---
VICUNA_ADAPTER="$OUTDIR/defender_v2_cka_20260307_115715"

if [ ! -f "$OUTDIR/adaptive_vicuna_baseline.json" ]; then
    echo "=== Adaptive GCG: Vicuna baseline ==="
    python adaptive_attack.py --model vicuna --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_vicuna_baseline.json" \
        2>&1 | tee "$OUTDIR/adaptive_vicuna_baseline.log"
fi

if [ ! -f "$OUTDIR/adaptive_vicuna_defended.json" ]; then
    echo "=== Adaptive GCG: Vicuna defended (ab2_ax) ==="
    python adaptive_attack.py --model vicuna --adapter_path "$VICUNA_ADAPTER" \
        --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_vicuna_defended.json" \
        2>&1 | tee "$OUTDIR/adaptive_vicuna_defended.log"
fi

if [ ! -f "$OUTDIR/advanced_vicuna_baseline.json" ]; then
    echo "=== Advanced: Vicuna baseline ==="
    python advanced_attacks.py --model vicuna --n_prompts $N_PROMPTS \
        --attacker mistral --attack both \
        --output "$OUTDIR/advanced_vicuna_baseline.json" \
        2>&1 | tee "$OUTDIR/advanced_vicuna_baseline.log"
fi

if [ ! -f "$OUTDIR/advanced_vicuna_defended.json" ]; then
    echo "=== Advanced: Vicuna defended (ab2_ax) ==="
    python advanced_attacks.py --model vicuna --adapter_path "$VICUNA_ADAPTER" \
        --n_prompts $N_PROMPTS --attacker mistral --attack both \
        --output "$OUTDIR/advanced_vicuna_defended.json" \
        2>&1 | tee "$OUTDIR/advanced_vicuna_defended.log"
fi

# --- LLAMA3-8B ---
LLAMA3_ADAPTER="$OUTDIR/defender_v2_cka_20260307_123357"

if [ ! -f "$OUTDIR/adaptive_llama3_baseline.json" ]; then
    echo "=== Adaptive GCG: Llama3 baseline ==="
    python adaptive_attack.py --model llama3 --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_llama3_baseline.json" \
        2>&1 | tee "$OUTDIR/adaptive_llama3_baseline.log"
fi

if [ ! -f "$OUTDIR/adaptive_llama3_defended.json" ]; then
    echo "=== Adaptive GCG: Llama3 defended (ab2_bx) ==="
    python adaptive_attack.py --model llama3 --adapter_path "$LLAMA3_ADAPTER" \
        --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_llama3_defended.json" \
        2>&1 | tee "$OUTDIR/adaptive_llama3_defended.log"
fi

if [ ! -f "$OUTDIR/advanced_llama3_baseline.json" ]; then
    echo "=== Advanced: Llama3 baseline ==="
    python advanced_attacks.py --model llama3 --n_prompts $N_PROMPTS \
        --attacker mistral --attack both \
        --output "$OUTDIR/advanced_llama3_baseline.json" \
        2>&1 | tee "$OUTDIR/advanced_llama3_baseline.log"
fi

if [ ! -f "$OUTDIR/advanced_llama3_defended.json" ]; then
    echo "=== Advanced: Llama3 defended (ab2_bx) ==="
    python advanced_attacks.py --model llama3 --adapter_path "$LLAMA3_ADAPTER" \
        --n_prompts $N_PROMPTS --attacker mistral --attack both \
        --output "$OUTDIR/advanced_llama3_defended.json" \
        2>&1 | tee "$OUTDIR/advanced_llama3_defended.log"
fi

echo ""
echo "============================================================"
echo "  7B ATTACKS COMPLETE"
echo "============================================================"

# Print summary
for model in qwen vicuna llama3; do
    echo ""
    echo "=== $model ==="
    for attack in adaptive advanced; do
        for variant in baseline defended; do
            f="$OUTDIR/${attack}_${model}_${variant}.json"
            if [ -f "$f" ]; then
                python3 -c "
import json; d=json.load(open('$f'))
asr = sum(1 for r in d.get('results',[]) if r.get('judge_success',False)) / max(len(d.get('results',[])),1) * 100
print(f'  ${attack}_${variant}: {asr:.1f}% ASR ({len(d.get(\"results\",[]))} prompts)')
" 2>/dev/null || echo "  ${attack}_${variant}: parse error"
            fi
        done
    done
done
