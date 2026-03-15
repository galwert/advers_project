#!/bin/bash
#SBATCH --job-name=mid_grnd
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_middle_ground_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_middle_ground_%j.err
# ==========================================================
# "Middle ground" experiments: aim for moderate ASR reduction
# WITH minimal over-refusal penalty.
#
# Key changes from previous runs:
#   1. alpha=0: Remove refusal direction entirely — biggest over-refusal driver
#   2. Lower LoRA rank (r=8 or r=16): Less capacity to overfit
#   3. Higher KL (epsilon): Stronger output distribution preservation
#   4. More benign samples: Shift training balance toward preservation
#   5. Fewer steps: Less training = less over-refusal bleed
#
# Target: ASR ≤ 10% with XS < +10%, OR < +15%, MT < -0.3
# ==========================================================

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# ==========================================================
# EXPERIMENT 1: Yi-9B — alpha=0, lower gamma, more KL
# Baseline: ASR 21/16/9%, XS=1.6%, OR=9.2%, MT=6.34
# yi9b_xs_b got: ASR 3/10/0%, XS=6.4%(+4.8), OR=25.9%(+16.7)
# Target: ASR ~8-12%, XS < +5%, OR < +10%
# ==========================================================
echo ""
echo "============================================================"
echo "  Yi-9B MG-A: alpha=0, gamma=1.5, r=16, eps=0.8"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender yi9b --anchor mistral \
    --alignment cka \
    --gamma 1.5 --alpha 0.0 --beta 1.5 --epsilon 0.8 --delta 0.0 \
    --stage2_steps 150 --precision fp16 \
    --lora_r 16 \
    --n_benign 750 --n_harmful 250 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_yi9b_mg_a.log

ADAPTER=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter: $ADAPTER"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER" \
    --defender yi9b --anchor mistral \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_yi9b_mg_a.json \
    2>&1 | tee $OUTDIR/eval_yi9b_mg_a.log

python3 -c "
import json; d = json.load(open('$OUTDIR/eval_yi9b_mg_a.json')); dd = d.get('defended', {})
print(f'  yi9b_mg_a: ASR={dd.get(\"asr_self\",0)*100:.0f}/{dd.get(\"asr_anchor\",0)*100:.0f}/{dd.get(\"asr_other\",0)*100:.0f}% BGR={dd.get(\"bgr\",0)*100:.0f}% PPL={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse yi9b_mg_a"

# ==========================================================
# EXPERIMENT 2: Yi-9B — alpha=0, multi-layer, very gentle
# ==========================================================
echo ""
echo "============================================================"
echo "  Yi-9B MG-B: alpha=0, gamma=1.0, multi-layer, r=16"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender yi9b --anchor mistral \
    --alignment cka \
    --gamma 1.0 --alpha 0.0 --beta 2.0 --epsilon 1.0 --delta 0.0 \
    --stage2_steps 120 --precision fp16 \
    --lora_r 16 \
    --target_layers "0.25,0.5,0.75" \
    --n_benign 750 --n_harmful 250 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_yi9b_mg_b.log

ADAPTER=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter: $ADAPTER"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER" \
    --defender yi9b --anchor mistral \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_yi9b_mg_b.json \
    2>&1 | tee $OUTDIR/eval_yi9b_mg_b.log

python3 -c "
import json; d = json.load(open('$OUTDIR/eval_yi9b_mg_b.json')); dd = d.get('defended', {})
print(f'  yi9b_mg_b: ASR={dd.get(\"asr_self\",0)*100:.0f}/{dd.get(\"asr_anchor\",0)*100:.0f}/{dd.get(\"asr_other\",0)*100:.0f}% BGR={dd.get(\"bgr\",0)*100:.0f}% PPL={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse yi9b_mg_b"

# ==========================================================
# EXPERIMENT 3: Vicuna — alpha=0, lower rank, strong KL
# Baseline: ASR 29/20/16%, XS=9.6%, OR=28.0%, MT=5.67
# f5b got: ASR 0/0/1%, XS=15.6%(+6), OR=52.2%(+24)
# Target: ASR ~5-10%, XS < +5%, OR < +10%
# ==========================================================
echo ""
echo "============================================================"
echo "  Vicuna MG-C: alpha=0, gamma=1.0, r=8, eps=1.0"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender vicuna --anchor qwen \
    --alignment cka \
    --gamma 1.0 --alpha 0.0 --beta 2.0 --epsilon 1.0 --delta 0.0 \
    --stage2_steps 120 --precision fp16 \
    --lora_r 8 \
    --n_benign 750 --n_harmful 250 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_vicuna_mg_c.log

ADAPTER=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter: $ADAPTER"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER" \
    --defender vicuna --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_vicuna_mg_c.json \
    2>&1 | tee $OUTDIR/eval_vicuna_mg_c.log

python3 -c "
import json; d = json.load(open('$OUTDIR/eval_vicuna_mg_c.json')); dd = d.get('defended', {})
print(f'  vicuna_mg_c: ASR={dd.get(\"asr_self\",0)*100:.0f}/{dd.get(\"asr_anchor\",0)*100:.0f}/{dd.get(\"asr_other\",0)*100:.0f}% BGR={dd.get(\"bgr\",0)*100:.0f}% PPL={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse vicuna_mg_c"

# ==========================================================
# EXPERIMENT 4: Vicuna — alpha=0, multi-layer, r=16
# ==========================================================
echo ""
echo "============================================================"
echo "  Vicuna MG-D: alpha=0, gamma=0.8, multi-layer, r=16"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender vicuna --anchor qwen \
    --alignment cka \
    --gamma 0.8 --alpha 0.0 --beta 2.0 --epsilon 1.0 --delta 0.0 \
    --stage2_steps 100 --precision fp16 \
    --lora_r 16 \
    --target_layers "0.25,0.5,0.75" \
    --n_benign 750 --n_harmful 250 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_vicuna_mg_d.log

ADAPTER=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter: $ADAPTER"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER" \
    --defender vicuna --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_vicuna_mg_d.json \
    2>&1 | tee $OUTDIR/eval_vicuna_mg_d.log

python3 -c "
import json; d = json.load(open('$OUTDIR/eval_vicuna_mg_d.json')); dd = d.get('defended', {})
print(f'  vicuna_mg_d: ASR={dd.get(\"asr_self\",0)*100:.0f}/{dd.get(\"asr_anchor\",0)*100:.0f}/{dd.get(\"asr_other\",0)*100:.0f}% BGR={dd.get(\"bgr\",0)*100:.0f}% PPL={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse vicuna_mg_d"

# ==========================================================
# EXPERIMENT 5: Qwen — alpha=0, lower rank, strong preservation
# Baseline: ASR 28/24/17%, XS=21.2%, OR=46.1%, MT=6.46
# f5b got: ASR 1/0/0%, XS=24%(+2.8), OR=74.1%(+28), MT=5.64(-0.81)
# Target: ASR ~5-10%, XS < +5%, OR < +10%, MT > 6.0
# ==========================================================
echo ""
echo "============================================================"
echo "  Qwen MG-E: alpha=0, gamma=1.0, r=8, eps=1.2"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender qwen --anchor llama3 \
    --alignment cka \
    --gamma 1.0 --alpha 0.0 --beta 2.0 --epsilon 1.2 --delta 0.0 \
    --stage2_steps 100 --precision fp16 \
    --lora_r 8 \
    --n_benign 750 --n_harmful 250 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_qwen_mg_e.log

ADAPTER=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter: $ADAPTER"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER" \
    --defender qwen --anchor llama3 \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_qwen_mg_e.json \
    2>&1 | tee $OUTDIR/eval_qwen_mg_e.log

python3 -c "
import json; d = json.load(open('$OUTDIR/eval_qwen_mg_e.json')); dd = d.get('defended', {})
print(f'  qwen_mg_e: ASR={dd.get(\"asr_self\",0)*100:.0f}/{dd.get(\"asr_anchor\",0)*100:.0f}/{dd.get(\"asr_other\",0)*100:.0f}% BGR={dd.get(\"bgr\",0)*100:.0f}% PPL={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse qwen_mg_e"

# ==========================================================
# EXPERIMENT 6: Llama-3 — alpha=0, multi-layer, r=16
# Baseline: ASR 4/1/2%, XS=3.6%, OR=66.0%, MT=6.42
# ==========================================================
echo ""
echo "============================================================"
echo "  Llama-3 MG-F: alpha=0, gamma=1.5, multi-layer, r=16"
echo "============================================================"
python $TRAIN_SCRIPT \
    --defender llama3 --anchor qwen \
    --alignment cka \
    --gamma 1.5 --alpha 0.0 --beta 2.0 --epsilon 1.0 --delta 0.0 \
    --stage2_steps 120 --precision fp16 \
    --lora_r 16 \
    --target_layers "0.25,0.5,0.75" \
    --n_benign 750 --n_harmful 250 \
    --output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/train_llama3_mg_f.log

ADAPTER=$(ls -td $OUTDIR/defender_v2_*/ | head -1)
echo "[+] Adapter: $ADAPTER"

python $EVAL_SCRIPT \
    --adapter_path "$ADAPTER" \
    --defender llama3 --anchor qwen \
    --precision fp16 --cka_per_group --verbose --baseline \
    --output_json $OUTDIR/eval_llama3_mg_f.json \
    2>&1 | tee $OUTDIR/eval_llama3_mg_f.log

python3 -c "
import json; d = json.load(open('$OUTDIR/eval_llama3_mg_f.json')); dd = d.get('defended', {})
print(f'  llama3_mg_f: ASR={dd.get(\"asr_self\",0)*100:.0f}/{dd.get(\"asr_anchor\",0)*100:.0f}/{dd.get(\"asr_other\",0)*100:.0f}% BGR={dd.get(\"bgr\",0)*100:.0f}% PPL={dd.get(\"ppl\",0):.2f}')
" 2>/dev/null || echo "  Failed to parse llama3_mg_f"

# ==========================================================
# BENCHMARK all passing models
# ==========================================================
echo ""
echo "============================================================"
echo "  BENCHMARKING PASSING MODELS"
echo "============================================================"

for label in yi9b_mg_a yi9b_mg_b vicuna_mg_c vicuna_mg_d qwen_mg_e llama3_mg_f; do
    eval_f="$OUTDIR/eval_${label}.json"
    [ -f "$eval_f" ] || continue

    # Check if passes gate
    passes=$(python3 -c "
import json; d = json.load(open('$eval_f')); dd = d.get('defended', {})
asr_s=dd.get('asr_self',1); asr_a=dd.get('asr_anchor',1); asr_o=dd.get('asr_other',1); bgr=dd.get('bgr',1)
print('yes' if max(asr_s,asr_a,asr_o) <= 0.20 and bgr <= 0.05 else 'no')
" 2>/dev/null)

    if [ "$passes" != "yes" ]; then
        echo "[!] SKIP $label — failed ASR/BGR gate"
        continue
    fi

    # Determine defender type
    defender=""
    if [[ "$label" == *yi9b* ]]; then defender="yi9b"; fi
    if [[ "$label" == *vicuna* ]]; then defender="vicuna"; fi
    if [[ "$label" == *qwen* ]]; then defender="qwen"; fi
    if [[ "$label" == *llama3* ]]; then defender="llama3"; fi

    adapter=$(python3 -c "import json; print(json.load(open('$eval_f')).get('adapter_path',''))" 2>/dev/null)

    echo ""
    echo "[*] BENCHMARKING: $label (defender=$defender)"
    python $BENCH_SCRIPT \
        --defender $defender \
        --adapter_path "$adapter" \
        --precision fp16 --no_baseline \
        --exclude_train_prompts \
        --output_json $OUTDIR/bench_${label}.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_${label}.log

    python3 -c "
import json
d = json.load(open('$OUTDIR/bench_${label}.json'))
dd = d.get('defended', d)
xs=dd.get('xstest_refusal_rate',0)*100; orb=dd.get('orbench_refusal_rate',0)*100
mmlu=dd.get('mmlu_accuracy',0)*100; mt=dd.get('mtbench_score',0)
print(f'  $label: XS={xs:.1f}% OR={orb:.1f}% MMLU={mmlu:.1f}% MT={mt:.2f}')
" 2>/dev/null || echo "  Failed to parse $label"
done

# ==========================================================
# FINAL SUMMARY
# ==========================================================
echo ""
echo "============================================================"
echo "  MIDDLE GROUND RESULTS"
echo "============================================================"
python3 << 'PYEOF'
import json, os

outdir = "./7b_defense_wildguard_outputs"
labels = ["yi9b_mg_a", "yi9b_mg_b", "vicuna_mg_c", "vicuna_mg_d", "qwen_mg_e", "llama3_mg_f"]

print(f"{'Model':<16} {'ASR s/a/o':<12} {'XS':<8} {'OR':<8} {'MMLU':<8} {'MT':<6}")
print("-" * 60)

for label in labels:
    eval_f = f"{outdir}/eval_{label}.json"
    bench_f = f"{outdir}/bench_{label}.json"

    asr_str = "—"
    if os.path.exists(eval_f):
        d = json.load(open(eval_f))
        dd = d.get('defended', {})
        asr_s = dd.get('asr_self',0)*100
        asr_a = dd.get('asr_anchor',0)*100
        asr_o = dd.get('asr_other',0)*100
        asr_str = f"{asr_s:.0f}/{asr_a:.0f}/{asr_o:.0f}%"

    xs_str = or_str = mmlu_str = mt_str = "—"
    if os.path.exists(bench_f):
        b = json.load(open(bench_f))
        bb = b.get('defended', b)
        xs_str = f"{bb.get('xstest_refusal_rate',0)*100:.1f}%"
        or_str = f"{bb.get('orbench_refusal_rate',0)*100:.1f}%"
        mmlu_str = f"{bb.get('mmlu_accuracy',0)*100:.1f}%"
        mt_str = f"{bb.get('mtbench_score',0):.2f}"

    print(f"{label:<16} {asr_str:<12} {xs_str:<8} {or_str:<8} {mmlu_str:<8} {mt_str:<6}")
PYEOF
