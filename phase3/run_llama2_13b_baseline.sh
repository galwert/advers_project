#!/bin/bash
#SBATCH --job-name=l2_13b_bl
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_l2_13b_bl_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_l2_13b_bl_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

EVAL_SCRIPT="evaluate_v2.py"
OUTDIR="./7b_defense_wildguard_outputs"

echo "=== Llama-2-13B Baseline ASR (WildGuard judge) ==="
python $EVAL_SCRIPT \
    --adapter_path none \
    --defender llama2-13b \
    --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_llama2_13b_baseline_v2.json \
    2>&1 | tee $OUTDIR/eval_llama2_13b_baseline_v2.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_llama2_13b_baseline_v2.json'))
for k in ['baseline','defended']:
    if k in d:
        dd = d[k]
        s=int(round(dd['asr_self']*100)); a=int(round(dd['asr_anchor']*100)); o=int(round(dd['asr_other']*100))
        b=int(round(dd['bgr']*100))
        print(f'  Llama-2-13B [{k}]: ASR={s}/{a}/{o}% BGR={b}%')
" 2>/dev/null

echo "=== DONE ==="
