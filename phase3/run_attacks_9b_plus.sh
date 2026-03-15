#!/bin/bash
#SBATCH --job-name=atk_9b+
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_atk_9bplus_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_atk_9bplus_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs"
N_PROMPTS=50

# ============================================================
# 9B+ MODELS: Adaptive GCG + Advanced (Embedding PGD + PAIR)
# Models: yi9b, nemo (12B), qwen-14b
# For each: baseline + defended
#
# NOTE: qwen-14b uses diverse_attacks_13b.py (takes full HF IDs)
# yi9b and nemo use adaptive_attack.py / advanced_attacks.py
# ============================================================

# --- YI-9B ---
YI9B_ADAPTER="$OUTDIR/defender_v2_cka_20260307_141547"

if [ ! -f "$OUTDIR/adaptive_yi9b_baseline.json" ]; then
    echo "=== Adaptive GCG: Yi-9B baseline ==="
    python adaptive_attack.py --model yi9b --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_yi9b_baseline.json" \
        2>&1 | tee "$OUTDIR/adaptive_yi9b_baseline.log"
fi

if [ ! -f "$OUTDIR/adaptive_yi9b_defended.json" ]; then
    echo "=== Adaptive GCG: Yi-9B defended (ab2_an) ==="
    python adaptive_attack.py --model yi9b --adapter_path "$YI9B_ADAPTER" \
        --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_yi9b_defended.json" \
        2>&1 | tee "$OUTDIR/adaptive_yi9b_defended.log"
fi

if [ ! -f "$OUTDIR/advanced_yi9b_baseline.json" ]; then
    echo "=== Advanced: Yi-9B baseline ==="
    python advanced_attacks.py --model yi9b --n_prompts $N_PROMPTS \
        --attacker mistral --attack both \
        --output "$OUTDIR/advanced_yi9b_baseline.json" \
        2>&1 | tee "$OUTDIR/advanced_yi9b_baseline.log"
fi

if [ ! -f "$OUTDIR/advanced_yi9b_defended.json" ]; then
    echo "=== Advanced: Yi-9B defended (ab2_an) ==="
    python advanced_attacks.py --model yi9b --adapter_path "$YI9B_ADAPTER" \
        --n_prompts $N_PROMPTS --attacker mistral --attack both \
        --output "$OUTDIR/advanced_yi9b_defended.json" \
        2>&1 | tee "$OUTDIR/advanced_yi9b_defended.log"
fi

# --- NEMO-12B ---
NEMO_ADAPTER="$OUTDIR/defender_v2_cka_20260307_161510"

if [ ! -f "$OUTDIR/adaptive_nemo_baseline.json" ]; then
    echo "=== Adaptive GCG: Nemo baseline ==="
    python adaptive_attack.py --model mistral_nemo --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_nemo_baseline.json" \
        2>&1 | tee "$OUTDIR/adaptive_nemo_baseline.log"
fi

if [ ! -f "$OUTDIR/adaptive_nemo_defended.json" ]; then
    echo "=== Adaptive GCG: Nemo defended (ab2_ax) ==="
    python adaptive_attack.py --model mistral_nemo --adapter_path "$NEMO_ADAPTER" \
        --n_prompts $N_PROMPTS --gcg_steps 500 \
        --output "$OUTDIR/adaptive_nemo_defended.json" \
        2>&1 | tee "$OUTDIR/adaptive_nemo_defended.log"
fi

if [ ! -f "$OUTDIR/advanced_nemo_baseline.json" ]; then
    echo "=== Advanced: Nemo baseline ==="
    python advanced_attacks.py --model mistral_nemo --n_prompts $N_PROMPTS \
        --attacker qwen --attack both \
        --output "$OUTDIR/advanced_nemo_baseline.json" \
        2>&1 | tee "$OUTDIR/advanced_nemo_baseline.log"
fi

if [ ! -f "$OUTDIR/advanced_nemo_defended.json" ]; then
    echo "=== Advanced: Nemo defended (ab2_ax) ==="
    python advanced_attacks.py --model mistral_nemo --adapter_path "$NEMO_ADAPTER" \
        --n_prompts $N_PROMPTS --attacker qwen --attack both \
        --output "$OUTDIR/advanced_nemo_defended.json" \
        2>&1 | tee "$OUTDIR/advanced_nemo_defended.log"
fi

# --- QWEN-14B ---
# Uses diverse_attacks_13b.py which takes full HF model IDs
# and supports --precision fp16 for memory management
# PAIR attacker = Mistral-7B (fits alongside 14B in fp16)
Q14_ADAPTER="$OUTDIR/defender_v2_cka_20260307_140824"

if [ ! -f "$OUTDIR/diverse_qwen14b_baseline.json" ]; then
    echo "=== Diverse attacks: Qwen-14B baseline ==="
    python diverse_attacks_13b.py \
        --model_id "Qwen/Qwen1.5-14B-Chat" \
        --precision fp16 --n_prompts $N_PROMPTS \
        --attacks autodan semantic_rewrite embedding_pgd pair \
        --attacker_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
        --output "$OUTDIR/diverse_qwen14b_baseline.json" \
        2>&1 | tee "$OUTDIR/diverse_qwen14b_baseline.log"
fi

if [ ! -f "$OUTDIR/diverse_qwen14b_defended.json" ]; then
    echo "=== Diverse attacks: Qwen-14B defended (q14_c) ==="
    python diverse_attacks_13b.py \
        --model_id "Qwen/Qwen1.5-14B-Chat" \
        --adapter_path "$Q14_ADAPTER" \
        --precision fp16 --n_prompts $N_PROMPTS \
        --attacks autodan semantic_rewrite embedding_pgd pair \
        --attacker_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
        --output "$OUTDIR/diverse_qwen14b_defended.json" \
        2>&1 | tee "$OUTDIR/diverse_qwen14b_defended.log"
fi

echo ""
echo "============================================================"
echo "  9B+ ATTACKS COMPLETE"
echo "============================================================"

# Print summary
for model in yi9b nemo; do
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

echo ""
echo "=== qwen-14b ==="
for variant in baseline defended; do
    f="$OUTDIR/diverse_qwen14b_${variant}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json; d=json.load(open('$f'))
results = d.get('results', d.get('attack_results', {}))
if isinstance(results, dict):
    for atk, atk_results in results.items():
        if isinstance(atk_results, list):
            asr = sum(1 for r in atk_results if r.get('judge_success', r.get('success',False))) / max(len(atk_results),1) * 100
            print(f'  {atk}_${variant}: {asr:.1f}% ({len(atk_results)} prompts)')
elif isinstance(results, list):
    asr = sum(1 for r in results if r.get('judge_success',False)) / max(len(results),1) * 100
    print(f'  diverse_${variant}: {asr:.1f}% ({len(results)} prompts)')
" 2>/dev/null || echo "  diverse_${variant}: parse error"
    fi
done
