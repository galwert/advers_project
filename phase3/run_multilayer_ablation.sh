#!/bin/bash
#SBATCH --job-name=ml_ablat
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_ml_ablation_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_ml_ablation_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

DEFENDER="mistralai/Mistral-7B-Instruct-v0.2"
ANCHOR="llama2"

# Base hyperparams from best Mistral config
GAMMA=1.0
ALPHA=0.8    # refusal
DELTA=0.04   # LM
EPSILON=0.15 # KL
BETA=1.0     # coherency
STEPS=300
SCOPE="harmful_only"

COMMON="--alignment cka --cka_scope $SCOPE --use_borderline --precision fp16 \
  --lora_r 32 --stage2_lr 2e-4 \
  --gamma $GAMMA --alpha $ALPHA --beta $BETA --epsilon $EPSILON --delta $DELTA \
  --stage2_steps $STEPS \
  --gcg_data_path ../outputs/advbench_suffixes_all_models_fixed.csv \
  --output_dir $OUTDIR"

# Map: label -> adapter_path (filled during training)
declare -A ADAPTER_MAP

do_train() {
    local label="$1"
    shift
    local extra_args="$@"

    echo ""
    echo "========================================"
    echo "  TRAIN: $label"
    echo "========================================"

    # Record adapter dirs before training
    BEFORE=$(ls -d $OUTDIR/defender_v2_cka_* 2>/dev/null | sort)

    python $TRAIN_SCRIPT \
        --defender $DEFENDER --anchor $ANCHOR \
        $COMMON \
        $extra_args

    # Find the NEW adapter (diff between before and after)
    AFTER=$(ls -d $OUTDIR/defender_v2_cka_* 2>/dev/null | sort)
    ADAPTER=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | tail -1)

    if [ -z "$ADAPTER" ] || [ ! -f "$ADAPTER/adapter_config.json" ]; then
        echo "[!] Training failed for $label — no new adapter found"
        return 1
    fi
    ADAPTER_MAP[$label]="$ADAPTER"
    echo "[+] $label adapter: $ADAPTER"
}

do_eval() {
    local label="$1"
    local adapter="${ADAPTER_MAP[$label]}"

    if [ -z "$adapter" ]; then
        echo "[SKIP] No adapter for $label"
        return 1
    fi

    echo ""
    echo "========================================"
    echo "  EVAL: $label"
    echo "========================================"

    # ASR eval
    python $EVAL_SCRIPT \
        --adapter_path "$adapter" \
        --defender mistral --anchor llama2 \
        --output_json "$OUTDIR/eval_ml_${label}.json" \
        --save_all_responses "$OUTDIR/responses_ml_${label}.json" \
        --precision 4bit --low_memory

    # Benchmark (XS + OR + MT, skip MMLU)
    python $BENCH_SCRIPT \
        --adapter "$adapter" \
        --model_name mistral \
        --output_json "$OUTDIR/bench_ml_${label}.json" \
        --no_baseline --skip_mmlu

    echo "[+] Eval done: $label"
}

echo "====================================================="
echo "  MULTI-LAYER CKA ABLATION — Mistral-7B"
echo "  Base: γ=$GAMMA α=$ALPHA δ=$DELTA ε=$EPSILON"
echo "  scope=$SCOPE steps=$STEPS anchor=$ANCHOR"
echo "====================================================="

# ═══════════════════════════════════════════════
# PHASE 1: ALL TRAINING (sequential, each run frees GPU)
# ═══════════════════════════════════════════════
echo ""
echo "=== PHASE 1: TRAINING ==="

# Group 1: Single-layer positions
do_train "single_0.25"    --target_layer_pct 0.25
do_train "single_0.375"   --target_layer_pct 0.375
do_train "single_0.4375"  --target_layer_pct 0.4375
do_train "single_0.46875" --target_layer_pct 0.46875
do_train "single_0.53125" --target_layer_pct 0.53125
do_train "single_0.5625"  --target_layer_pct 0.5625
do_train "single_0.625"   --target_layer_pct 0.625
do_train "single_0.75"    --target_layer_pct 0.75

# Group 2: Weighted multi-layer
do_train "weighted_0.375_0.5" \
    --target_layers "0.375,0.5" --layer_weights "0.3,1.0"

do_train "weighted_0.5_0.625" \
    --target_layers "0.5,0.625" --layer_weights "1.0,0.3"

do_train "weighted_3L_light" \
    --target_layers "0.375,0.5,0.625" --layer_weights "0.3,1.0,0.3"

do_train "weighted_3L_medium" \
    --target_layers "0.375,0.5,0.625" --layer_weights "0.5,1.0,0.5"

do_train "weighted_3L_wide" \
    --target_layers "0.25,0.5,0.75" --layer_weights "0.3,1.0,0.3"

do_train "weighted_5L" \
    --target_layers "0.25,0.375,0.5,0.625,0.75" --layer_weights "0.2,0.5,1.0,0.5,0.2"

# Group 3: Concat mode
do_train "concat_3L" \
    --target_layers "0.375,0.5,0.625" --cka_multi_mode concat

do_train "concat_3L_wide" \
    --target_layers "0.25,0.5,0.75" --cka_multi_mode concat

do_train "concat_5L" \
    --target_layers "0.25,0.375,0.5,0.625,0.75" --cka_multi_mode concat

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Trained adapters:"
for label in "${!ADAPTER_MAP[@]}"; do
    echo "  $label -> ${ADAPTER_MAP[$label]}"
done

# ═══════════════════════════════════════════════
# PHASE 2: ALL EVALUATIONS
# ═══════════════════════════════════════════════
echo ""
echo "=== PHASE 2: EVALUATION ==="

for label in \
    single_0.25 single_0.375 single_0.4375 single_0.46875 \
    single_0.53125 single_0.5625 single_0.625 single_0.75 \
    weighted_0.375_0.5 weighted_0.5_0.625 \
    weighted_3L_light weighted_3L_medium weighted_3L_wide weighted_5L \
    concat_3L concat_3L_wide concat_5L; do
    do_eval "$label"
done

# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════
echo ""
echo "====================================================="
echo "  ABLATION COMPLETE — SUMMARY"
echo "====================================================="

python3 << 'PYEOF'
import json, os, glob

OUTDIR = "./7b_defense_wildguard_outputs"

print(f"\n{'Config':<25} {'ASR(s/a/o)':<15} {'BGR':<8} {'XS':<8} {'OR':<8} {'MT':<8}")
print("-" * 80)

for f in sorted(glob.glob(f"{OUTDIR}/eval_ml_*.json")):
    label = os.path.basename(f).replace("eval_ml_", "").replace(".json", "")
    try:
        d = json.load(open(f))
        dd = d.get("defended", d)
        fmt = lambda x: f"{x*100:.1f}" if isinstance(x, float) else "N/A"
        asr_s = fmt(dd.get("asr_self", None))
        asr_a = fmt(dd.get("asr_anchor", None))
        asr_o = fmt(dd.get("asr_other", None))
        bgr = fmt(dd.get("bgr", None))
    except:
        asr_s = asr_a = asr_o = bgr = "ERR"

    # Benchmark
    bf = f"{OUTDIR}/bench_ml_{label}.json"
    xs = orb = mt = "N/A"
    if os.path.exists(bf):
        try:
            bd = json.load(open(bf))
            bdd = bd.get("defended", bd)
            xs = f"{bdd.get('xstest_refusal_rate', 0)*100:.1f}" if 'xstest_refusal_rate' in bdd else "N/A"
            orb = f"{bdd.get('orbench_refusal_rate', 0)*100:.1f}" if 'orbench_refusal_rate' in bdd else "N/A"
            mt = f"{bdd.get('mt_bench_score', 0):.2f}" if 'mt_bench_score' in bdd else "N/A"
        except:
            pass

    print(f"{label:<25} {asr_s:>4}/{asr_a:>4}/{asr_o:>4} {bgr:>6}  {xs:>6}  {orb:>6}  {mt:>6}")
PYEOF
