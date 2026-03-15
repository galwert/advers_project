#!/bin/bash
#SBATCH --job-name=ml_cka
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_multilayer_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_multilayer_%j.err
# ==========================================================
# Test multi-layer CKA on Llama-3-8B (previously failed configs)
# Uses 3 layers (25%, 50%, 75%) instead of single middle layer
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
OUTDIR="./7b_defense_wildguard_outputs"

echo ""
echo "============================================================"
echo "  MULTI-LAYER CKA TEST: Llama-3-8B (3 layers: 25/50/75%)"
echo "============================================================"

# Config A: moderate gamma, multi-layer — comparable to f3a which failed at single-layer
echo ""
echo "[1/2] Training Llama-3-8B multi-layer config A (gamma=2.0)"
python $TRAIN_SCRIPT \
    --defender llama3 --anchor qwen \
    --gamma 2.0 --alpha 0.15 --beta 1.0 --epsilon 0.4 --delta 0 \
    --stage2_steps 200 --precision fp16 \
    --alignment cka \
    --target_layers "0.25,0.5,0.75" \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_llama3_ml_a.log

# Find the latest adapter
ADAPTER_A=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter A: $ADAPTER_A"

# Evaluate
echo ""
echo "[1/2] Evaluating config A..."
python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_A" \
    --defender llama3 --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_llama3_ml_a.json \
    2>&1 | tee $OUTDIR/eval_llama3_ml_a.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_llama3_ml_a.json'))
dd = d.get('defended', {})
print(f'  Config A: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse config A"

# Config B: stronger gamma + more KL, multi-layer
echo ""
echo "[2/2] Training Llama-3-8B multi-layer config B (gamma=3.0, epsilon=0.6)"
python $TRAIN_SCRIPT \
    --defender llama3 --anchor qwen \
    --gamma 3.0 --alpha 0.15 --beta 1.0 --epsilon 0.6 --delta 0.06 \
    --stage2_steps 250 --precision fp16 \
    --alignment cka \
    --target_layers "0.25,0.5,0.75" \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_llama3_ml_b.log

ADAPTER_B=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter B: $ADAPTER_B"

echo ""
echo "[2/2] Evaluating config B..."
python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_B" \
    --defender llama3 --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_llama3_ml_b.json \
    2>&1 | tee $OUTDIR/eval_llama3_ml_b.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_llama3_ml_b.json'))
dd = d.get('defended', {})
print(f'  Config B: asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse config B"

# ============================================================
# Config C: Mistral-7B - had good ASR (6/2/3%) but bad benchmarks
# (MT=4.16, OR=57.1%). Try multi-layer with LOWER gamma to
# reduce over-refusal while maintaining defense via multi-layer.
# Original: gamma=1.0, alpha=0.15, epsilon=1.0, delta=0.06, 300 steps
# ============================================================
echo ""
echo "============================================================"
echo "  MULTI-LAYER: Mistral-7B config C (gamma=0.7, lower alpha)"
echo "  Original f4c: ASR=6/2/3% but MT=4.16, OR=57.1%"
echo "============================================================"

python $TRAIN_SCRIPT \
    --defender mistral --anchor llama2 \
    --gamma 0.7 --alpha 0.10 --beta 1.0 --epsilon 0.8 --delta 0.06 \
    --stage2_steps 250 --precision fp16 \
    --alignment cka \
    --target_layers "0.25,0.5,0.75" \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_mistral_ml_c.log

ADAPTER_C=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter C: $ADAPTER_C"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_C" \
    --defender mistral --anchor llama2 \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_mistral_ml_c.json \
    2>&1 | tee $OUTDIR/eval_mistral_ml_c.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_mistral_ml_c.json'))
dd = d.get('defended', {})
print(f'  Config C (Mistral ML): asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse config C"

# Config D: Mistral-7B with even gentler touch
echo ""
echo "============================================================"
echo "  MULTI-LAYER: Mistral-7B config D (gamma=0.5, minimal refusal)"
echo "============================================================"

python $TRAIN_SCRIPT \
    --defender mistral --anchor llama2 \
    --gamma 0.5 --alpha 0.05 --beta 1.5 --epsilon 1.0 --delta 0.08 \
    --stage2_steps 200 --precision fp16 \
    --alignment cka \
    --target_layers "0.25,0.5,0.75" \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_mistral_ml_d.log

ADAPTER_D=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter D: $ADAPTER_D"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER_D" \
    --defender mistral --anchor llama2 \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_mistral_ml_d.json \
    2>&1 | tee $OUTDIR/eval_mistral_ml_d.log

python3 -c "
import json
d = json.load(open('$OUTDIR/eval_mistral_ml_d.json'))
dd = d.get('defended', {})
print(f'  Config D (Mistral ML): asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse config D"

echo ""
echo "============================================================"
echo "  ALL MULTI-LAYER CKA TESTS COMPLETE"
echo "============================================================"
echo ""
echo "  Llama-3-8B (previously failed single-layer):"
echo "    Config A (gamma=2.0, 3-layer):"
python3 -c "
import json
d = json.load(open('$OUTDIR/eval_llama3_ml_a.json'))
dd = d.get('defended', {})
print(f'      asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null
echo "    Config B (gamma=3.0, 3-layer):"
python3 -c "
import json
d = json.load(open('$OUTDIR/eval_llama3_ml_b.json'))
dd = d.get('defended', {})
print(f'      asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null
echo "    Ref single-layer best (r1a): 0/0/0% ASR"
echo "    Ref single-layer worst (f3a): 58/66/50% ASR"
echo ""
echo "  Mistral-7B (good ASR, bad benchmarks single-layer):"
echo "    Config C (gamma=0.7, 3-layer):"
python3 -c "
import json
d = json.load(open('$OUTDIR/eval_mistral_ml_c.json'))
dd = d.get('defended', {})
print(f'      asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null
echo "    Config D (gamma=0.5, 3-layer):"
python3 -c "
import json
d = json.load(open('$OUTDIR/eval_mistral_ml_d.json'))
dd = d.get('defended', {})
print(f'      asr_s={dd.get(\"asr_self\",0)*100:.0f}% asr_a={dd.get(\"asr_anchor\",0)*100:.0f}% asr_o={dd.get(\"asr_other\",0)*100:.0f}% bgr={dd.get(\"bgr\",0)*100:.0f}% ppl={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null
echo "    Ref single-layer f4c: ASR=6/2/3% XS=30% OR=57% MT=4.16"
