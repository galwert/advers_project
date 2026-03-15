#!/bin/bash
#SBATCH --job-name=ml_abl2
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_ml_ablation2_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_ml_ablation2_%j.err

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
CACHE_FILE="$OUTDIR/anchor_cache_llama2_all_layers.pt"

# Base hyperparams
GAMMA=1.0
ALPHA=0.8
DELTA=0.04
EPSILON=0.15
BETA=1.0
STEPS=300
SCOPE="harmful_only"

# All layers needed across all configs: 0.375, 0.4375, 0.5, 0.5625, 0.59375, 0.625
ALL_LAYERS="0.375,0.4375,0.5,0.5625,0.59375,0.625"

COMMON_BASE="--alignment cka --cka_scope $SCOPE --use_borderline --precision fp32 \
  --lora_r 32 --stage2_lr 2e-4 \
  --gamma $GAMMA --alpha $ALPHA --beta $BETA --epsilon $EPSILON --delta $DELTA \
  --stage2_steps $STEPS \
  --gcg_data_path ../outputs/advbench_suffixes_all_models_fixed.csv \
  --output_dir $OUTDIR"

# ═══════════════════════════════════════════════
# STEP 0: Pre-compute anchor cache with ALL layers
# ═══════════════════════════════════════════════
if [ ! -f "$CACHE_FILE" ]; then
    echo "====================================================="
    echo "  PRECOMPUTING ANCHOR CACHE (all layers)"
    echo "  Layers: $ALL_LAYERS"
    echo "====================================================="

    python $TRAIN_SCRIPT \
        --defender $DEFENDER --anchor $ANCHOR \
        --target_layers "$ALL_LAYERS" \
        --anchor_precision fp16 \
        $COMMON_BASE \
        --save_anchor_cache "$CACHE_FILE"

    if [ ! -f "$CACHE_FILE" ]; then
        echo "[!] FATAL: Anchor cache creation failed"
        exit 1
    fi
    echo "[+] Anchor cache saved to $CACHE_FILE"
else
    echo "[+] Using existing anchor cache: $CACHE_FILE"
fi

# Common args for all training runs (using cache)
COMMON="$COMMON_BASE --load_anchor_cache $CACHE_FILE"

declare -A ADAPTER_MAP

do_train() {
    local label="$1"
    shift
    local extra_args="$@"

    echo ""
    echo "========================================"
    echo "  TRAIN: $label"
    echo "========================================"

    BEFORE=$(ls -d $OUTDIR/defender_v2_cka_* 2>/dev/null | sort)

    python $TRAIN_SCRIPT \
        --defender $DEFENDER --anchor $ANCHOR \
        $COMMON \
        $extra_args

    AFTER=$(ls -d $OUTDIR/defender_v2_cka_* 2>/dev/null | sort)
    ADAPTER=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | tail -1)

    if [ -z "$ADAPTER" ] || [ ! -f "$ADAPTER/adapter_config.json" ]; then
        echo "[!] Training failed for $label"
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
    echo "  EVAL+BENCH: $label"
    echo "========================================"

    # ASR eval
    python $EVAL_SCRIPT \
        --adapter_path "$adapter" \
        --defender mistral --anchor llama2 \
        --output_json "$OUTDIR/eval_ml2_${label}.json" \
        --save_all_responses "$OUTDIR/responses_ml2_${label}.json" \
        --precision fp32 --low_memory

    # Benchmark (XS + OR + MT, skip MMLU — run MMLU separately on winners)
    python $BENCH_SCRIPT \
        --defender mistral \
        --adapter_path "$adapter" \
        --output_json "$OUTDIR/bench_ml2_${label}.json" \
        --no_baseline --skip_mmlu --low_memory

    echo "[+] Done: $label"
}

echo "====================================================="
echo "  MULTI-LAYER ABLATION V2 — Mistral-7B (fp32 + cache)"
echo "  Focus: two sweet spots (0.5, 0.625) + asymmetric profiles"
echo "====================================================="

# ═══════════════════════════════════════════════
# PHASE 1: TRAINING (all use cached anchor)
# ═══════════════════════════════════════════════
echo ""
echo "=== PHASE 1: TRAINING ==="

# A: Fill the gap
do_train "single_0.59375" --target_layer_pct 0.59375

# B: Two sweet spots combined
do_train "duo_equal" \
    --target_layers "0.5,0.625" --layer_weights "1.0,1.0"

do_train "duo_repulse_early" \
    --target_layers "0.5,0.625" --layer_weights "1.0,0.3"

do_train "duo_repulse_late" \
    --target_layers "0.5,0.625" --layer_weights "0.3,1.0"

# C: Asymmetric loss profiles (CKA heavier early, coherency heavier late)
do_train "asym_repulse05_preserve0625" \
    --target_layers "0.5,0.625" \
    --layer_weights "1.0,0.3" \
    --coherency_layer_weights "0.5,2.0"

do_train "asym_3L_repulse_mid" \
    --target_layers "0.4375,0.5,0.625" \
    --layer_weights "0.5,1.0,0.2" \
    --coherency_layer_weights "0.3,1.0,3.0"

do_train "asym_3L_wide" \
    --target_layers "0.375,0.5,0.625" \
    --layer_weights "0.3,1.0,0.2" \
    --coherency_layer_weights "0.3,1.0,3.0"

do_train "asym_strong_preserve" \
    --target_layers "0.5,0.625" \
    --layer_weights "1.0,0.5" \
    --coherency_layer_weights "0.5,5.0"

# D: Concat of sweet spots
do_train "concat_duo" \
    --target_layers "0.5,0.625" --cka_multi_mode concat

do_train "concat_trio_tight" \
    --target_layers "0.5,0.5625,0.625" --cka_multi_mode concat

# E: Hyperparameter variations on duo
do_train "duo_gamma1.5" \
    --target_layers "0.5,0.625" --layer_weights "1.0,0.5" \
    --gamma 1.5

do_train "duo_eps0.3" \
    --target_layers "0.5,0.625" --layer_weights "1.0,0.5" \
    --epsilon 0.3

do_train "duo_gamma0.5" \
    --target_layers "0.5,0.625" --layer_weights "1.0,0.5" \
    --gamma 0.5

do_train "duo_steps500" \
    --target_layers "0.5,0.625" --layer_weights "1.0,0.5" \
    --stage2_steps 500

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Trained adapters:"
for label in "${!ADAPTER_MAP[@]}"; do
    echo "  $label -> ${ADAPTER_MAP[$label]}"
done

# ═══════════════════════════════════════════════
# PHASE 2: EVAL + BENCHMARKS
# ═══════════════════════════════════════════════
echo ""
echo "=== PHASE 2: EVALUATION + BENCHMARKS ==="

for label in \
    single_0.59375 \
    duo_equal duo_repulse_early duo_repulse_late \
    asym_repulse05_preserve0625 asym_3L_repulse_mid asym_3L_wide asym_strong_preserve \
    concat_duo concat_trio_tight \
    duo_gamma1.5 duo_eps0.3 duo_gamma0.5 duo_steps500; do
    do_eval "$label"
done

# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════
echo ""
echo "====================================================="
echo "  ABLATION V2 — SUMMARY"
echo "====================================================="

python3 << 'PYEOF'
import json, os, glob

OUTDIR = "./7b_defense_wildguard_outputs"
BXS, BOR, BMT = 0.08, 0.221, 6.38

print(f"\n{'Config':<30} {'ASR(s/a/o)':<15} {'BGR':<7} {'XS':<12} {'OR':<12} {'MT':<12}")
print("-" * 95)

for f in sorted(glob.glob(f"{OUTDIR}/eval_ml2_*.json")):
    label = os.path.basename(f).replace("eval_ml2_", "").replace(".json", "")
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

    bf = f"{OUTDIR}/bench_ml2_{label}.json"
    xs = orb = mt = "—"
    if os.path.exists(bf):
        try:
            bd = json.load(open(bf))
            bdd = bd.get("defended", bd)
            if 'xstest_refusal_rate' in bdd:
                v = bdd['xstest_refusal_rate']
                xs = f"{v*100:.1f}({(v-BXS)*100:+.1f})"
            if 'orbench_refusal_rate' in bdd:
                v = bdd['orbench_refusal_rate']
                orb = f"{v*100:.1f}({(v-BOR)*100:+.1f})"
            if 'mt_bench_score' in bdd:
                v = bdd['mt_bench_score']
                mt = f"{v:.2f}({v-BMT:+.2f})"
        except:
            pass

    print(f"{label:<30} {asr_s:>4}/{asr_a:>4}/{asr_o:>4} {bgr:>5}  {xs:>10}  {orb:>10}  {mt:>10}")
PYEOF
