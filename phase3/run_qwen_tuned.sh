#!/bin/bash
#SBATCH --job-name=qwen_tune
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_qwen_tuned_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_qwen_tuned_%j.err
# ==========================================================
# Qwen-7B: tune for lower over-refusal and better MT-Bench
# f5b baseline: gamma=3.0 alpha=0.15 epsilon=0.4 delta=0.03
#   Results: ASR=1/0/0% XS=24.0%(+2.8) OR=74.1%(+28.0) MT=5.64(-0.82)
# Qwen baseline: XS=21.2% OR=46.1% MMLU=60.4% MT=6.46
# Goal: keep ASR ≤ 15%, get OR < 55%, MT > 6.0
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Config H: Much lower gamma, strong KL + LM loss for quality preservation
echo ""
echo "============================================================"
echo "  Qwen f5h: gamma=1.5 alpha=0.08 epsilon=1.0 delta=0.06"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender qwen --anchor llama3 \
    --alignment cka \
    --gamma 1.5 --alpha 0.08 --beta 1.5 --epsilon 1.0 --delta 0.06 \
    --stage2_steps 200 --precision fp16 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_qwen_f5h.log

ADAPTER_H=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter H: $ADAPTER_H"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_H" \
    --defender qwen --anchor llama3 \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_qwen7b_f5h.json \
    2>&1 | tee $OUTDIR/eval_qwen7b_f5h.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_qwen7b_f5h.json'))
dd = d.get('defended', {})
print(f'  f5h: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse f5h"

# Config I: Even more conservative — high coherency + KL, low gamma
echo ""
echo "============================================================"
echo "  Qwen f5i: gamma=1.0 alpha=0.05 epsilon=1.2 delta=0.08 beta=2.0"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender qwen --anchor llama3 \
    --alignment cka \
    --gamma 1.0 --alpha 0.05 --beta 2.0 --epsilon 1.2 --delta 0.08 \
    --stage2_steps 150 --precision fp16 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_qwen_f5i.log

ADAPTER_I=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter I: $ADAPTER_I"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_I" \
    --defender qwen --anchor llama3 \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_qwen7b_f5i.json \
    2>&1 | tee $OUTDIR/eval_qwen7b_f5i.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_qwen7b_f5i.json'))
dd = d.get('defended', {})
print(f'  f5i: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse f5i"

# Config J: Multi-layer with moderate params
echo ""
echo "============================================================"
echo "  Qwen f5j: MULTI-LAYER gamma=1.2 alpha=0.08 epsilon=0.8"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender qwen --anchor llama3 \
    --alignment cka \
    --gamma 1.2 --alpha 0.08 --beta 1.5 --epsilon 0.8 --delta 0.05 \
    --stage2_steps 200 --precision fp16 \
    --target_layers "0.25,0.5,0.75" \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_qwen_f5j.log

ADAPTER_J=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter J: $ADAPTER_J"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_J" \
    --defender qwen --anchor llama3 \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_qwen7b_f5j.json \
    2>&1 | tee $OUTDIR/eval_qwen7b_f5j.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_qwen7b_f5j.json'))
dd = d.get('defended', {})
print(f'  f5j: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse f5j"

# Benchmark the best passing config
echo ""
echo "============================================================"
echo "  Selecting best config for benchmarking..."
echo "============================================================"
python3 << 'PYEOF'
import json, os

outdir = "./7b_defense_wildguard_outputs"
best = None
best_score = 999

for label in ["qwen7b_f5h", "qwen7b_f5i", "qwen7b_f5j"]:
    f = f"{outdir}/eval_{label}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    dd = d.get("defended", {})
    asr_s = dd.get("asr_self", 1)
    asr_a = dd.get("asr_anchor", 1)
    asr_o = dd.get("asr_other", 1)
    bgr = dd.get("bgr", 1)

    max_asr = max(asr_s, asr_a, asr_o)
    if max_asr > 0.20 or bgr > 0.05:
        print(f"  {label}: SKIP (max_ASR={max_asr*100:.0f}% BGR={bgr*100:.0f}%)")
        continue

    # Lower ASR sum = better
    score = asr_s + asr_a + asr_o + bgr
    print(f"  {label}: score={score:.3f} (ASR={asr_s*100:.0f}/{asr_a*100:.0f}/{asr_o*100:.0f}%)")
    if score < best_score:
        best_score = score
        best = label

if best:
    print(f"\n  BEST: {best}")
    with open(f"{outdir}/_qwen_best_label.txt", "w") as f:
        f.write(best)
else:
    print("\n  No config passed gate!")
PYEOF

BEST_LABEL=$(cat $OUTDIR/_qwen_best_label.txt 2>/dev/null)
if [ -n "$BEST_LABEL" ]; then
    BEST_EVAL="$OUTDIR/eval_${BEST_LABEL}.json"
    BEST_ADAPTER=$(python3 -c "import json; print(json.load(open('$BEST_EVAL')).get('adapter_path',''))" 2>/dev/null)

    echo ""
    echo "============================================================"
    echo "  BENCHMARKING: $BEST_LABEL"
    echo "  Adapter: $BEST_ADAPTER"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --defender qwen \
        --adapter_path "$BEST_ADAPTER" \
        --precision fp16 --no_baseline \
        --exclude_train_prompts \
        --output_json $OUTDIR/bench_${BEST_LABEL}.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_${BEST_LABEL}.log

    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${BEST_LABEL}.json'))
dd = d.get('defended', d)
xs=dd.get('xstest_refusal_rate',0)*100
orb=dd.get('orbench_refusal_rate',0)*100
mmlu=dd.get('mmlu_accuracy',0)*100
mt=dd.get('mtbench_score',0)
print(f'  {\"$BEST_LABEL\"}: XS={xs:.1f}% OR={orb:.1f}% MMLU={mmlu:.1f}% MT={mt:.2f}')
print(f'  Baseline:       XS=21.2% OR=46.1%  MMLU=60.4% MT=6.46')
print(f'  f5b reference:  XS=24.0% OR=74.1%  MMLU=60.6% MT=5.64')
" 2>/dev/null
fi

echo ""
echo "============================================================"
echo "  QWEN TUNING COMPLETE"
echo "============================================================"
