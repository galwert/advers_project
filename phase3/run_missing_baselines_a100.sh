#!/bin/bash
#SBATCH --job-name=miss_a100
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_miss_a100_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_miss_a100_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs"
N_PROMPTS=50

# 1. Nemo advanced baseline (PGD + PAIR) — OOMed on L40
echo "=== Advanced: Nemo baseline ==="
python advanced_attacks.py --model mistral_nemo --n_prompts $N_PROMPTS \
    --attacker qwen --attack both \
    --output "$OUTDIR/advanced_nemo_baseline.json" \
    2>&1 | tee "$OUTDIR/advanced_nemo_baseline.log"

# 2. Qwen-14B diverse baseline — OOMed on L40 (14B model + 7B attacker)
echo "=== Diverse attacks: Qwen-14B baseline ==="
python diverse_attacks_13b.py \
    --model_id "Qwen/Qwen1.5-14B-Chat" \
    --precision fp16 --n_prompts $N_PROMPTS \
    --attacks autodan semantic_rewrite embedding_pgd pair \
    --attacker_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
    --output "$OUTDIR/diverse_qwen14b_baseline.json" \
    2>&1 | tee "$OUTDIR/diverse_qwen14b_baseline.log"

echo ""
echo "=== DONE ==="
for f in advanced_nemo_baseline diverse_qwen14b_baseline; do
    if [ -f "$OUTDIR/${f}.json" ]; then
        python3 -c "
import json
d = json.load(open('$OUTDIR/${f}.json'))
n = len(d.get('results', []))
asr_keys = [k for k in d if 'asr' in k.lower()]
print(f'  $f: {n} prompts')
for k in asr_keys:
    print(f'    {k}: {d[k]}')
" 2>/dev/null
    else
        echo "  $f: MISSING"
    fi
done
