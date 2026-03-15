#!/bin/bash
#SBATCH --job-name=vic_tune
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_vicuna_tuned_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_vicuna_tuned_%j.err
# ==========================================================
# Vicuna-7B: tune for lower over-refusal (XSTest/OR-Bench)
# f5b baseline: gamma=2.0 alpha=0.15 epsilon=0.4 delta=0.03
#   Results: ASR=0/0/1% XS=15.6%(+6.0) OR=52.2%(+24.2)
# Goal: keep ASR ≤ 10%, reduce XS and OR closer to baseline
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# Config E: Lower gamma + alpha, higher KL regularization (single-layer)
echo ""
echo "============================================================"
echo "  Vicuna f5e: gamma=1.2 alpha=0.08 epsilon=0.7 (gentle)"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender vicuna --anchor qwen \
    --alignment cka \
    --gamma 1.2 --alpha 0.08 --beta 1.0 --epsilon 0.7 --delta 0.03 \
    --stage2_steps 200 --precision fp16 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_vicuna_f5e.log

ADAPTER_E=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter E: $ADAPTER_E"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_E" \
    --defender vicuna --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_vicuna7b_f5e.json \
    2>&1 | tee $OUTDIR/eval_vicuna7b_f5e.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_vicuna7b_f5e.json'))
dd = d.get('defended', {})
print(f'  f5e: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse f5e"

# Config F: Even gentler — minimal refusal direction
echo ""
echo "============================================================"
echo "  Vicuna f5f: gamma=1.5 alpha=0.05 epsilon=0.8 delta=0.05"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender vicuna --anchor qwen \
    --alignment cka \
    --gamma 1.5 --alpha 0.05 --beta 1.5 --epsilon 0.8 --delta 0.05 \
    --stage2_steps 200 --precision fp16 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_vicuna_f5f.log

ADAPTER_F=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter F: $ADAPTER_F"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_F" \
    --defender vicuna --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_vicuna7b_f5f.json \
    2>&1 | tee $OUTDIR/eval_vicuna7b_f5f.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_vicuna7b_f5f.json'))
dd = d.get('defended', {})
print(f'  f5f: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse f5f"

# Config G: Multi-layer with gentle params
echo ""
echo "============================================================"
echo "  Vicuna f5g: MULTI-LAYER gamma=1.0 alpha=0.08 epsilon=0.6"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender vicuna --anchor qwen \
    --alignment cka \
    --gamma 1.0 --alpha 0.08 --beta 1.0 --epsilon 0.6 --delta 0.03 \
    --stage2_steps 200 --precision fp16 \
    --target_layers "0.25,0.5,0.75" \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_vicuna_f5g.log

ADAPTER_G=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter G: $ADAPTER_G"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_G" \
    --defender vicuna --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_vicuna7b_f5g.json \
    2>&1 | tee $OUTDIR/eval_vicuna7b_f5g.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_vicuna7b_f5g.json'))
dd = d.get('defended', {})
print(f'  f5g: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse f5g"

# Benchmark the best one (lowest over-refusal with ASR ≤ 20%)
echo ""
echo "============================================================"
echo "  Selecting best config for benchmarking..."
echo "============================================================"
python3 << 'PYEOF'
import json, os

outdir = "./7b_defense_wildguard_outputs"
best = None
best_score = 999

for label in ["vicuna7b_f5e", "vicuna7b_f5f", "vicuna7b_f5g"]:
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
        print(f"  {label}: SKIP (ASR={max_asr*100:.0f}% or BGR={bgr*100:.0f}%)")
        continue

    # Score: sum of ASR + BGR (lower = better, less over-refusal expected)
    score = asr_s + asr_a + asr_o + bgr
    print(f"  {label}: score={score:.3f} (ASR={asr_s*100:.0f}/{asr_a*100:.0f}/{asr_o*100:.0f}%)")
    if score < best_score:
        best_score = score
        best = label

if best:
    print(f"\n  BEST: {best}")
    with open(f"{outdir}/_vicuna_best_label.txt", "w") as f:
        f.write(best)
else:
    print("\n  No config passed gate!")
PYEOF

BEST_LABEL=$(cat $OUTDIR/_vicuna_best_label.txt 2>/dev/null)
if [ -n "$BEST_LABEL" ]; then
    BEST_EVAL="$OUTDIR/eval_${BEST_LABEL}.json"
    BEST_ADAPTER=$(python3 -c "import json; print(json.load(open('$BEST_EVAL')).get('adapter_path',''))" 2>/dev/null)

    echo ""
    echo "============================================================"
    echo "  BENCHMARKING: $BEST_LABEL"
    echo "  Adapter: $BEST_ADAPTER"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --defender vicuna \
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
print(f'  Baseline:       XS=9.6%  OR=28.0%  MMLU=N/A   MT=5.67')
print(f'  f5b reference:  XS=15.6% OR=52.2%  MMLU=50.0% MT=5.71')
" 2>/dev/null
fi

echo ""
echo "============================================================"
echo "  VICUNA TUNING COMPLETE"
echo "============================================================"
