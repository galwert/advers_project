#!/bin/bash
#SBATCH --job-name=mst_sw2
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_mst_sw2_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_mst_sw2_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"
GCG_DATA="$OUTDIR/gcg_strings_mistral.json"

COMMON="--alignment cka --use_borderline --precision fp32 \
        --target_layer_pct 0.5 --lora_r 32 --stage2_lr 2e-4 \
        --gcg_data_path $GCG_DATA --output_dir $OUTDIR"

run_config() {
    local TAG=$1
    local GAMMA=$2
    local ALPHA=$3
    local EPS=$4
    local DELTA=$5
    local STEPS=$6

    if [ -f "$OUTDIR/eval_${TAG}.json" ]; then
        echo "[SKIP] $TAG eval exists"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  $TAG: gamma=$GAMMA alpha=$ALPHA eps=$EPS delta=$DELTA steps=$STEPS"
    echo "============================================================"

    # Train and capture adapter path from stdout (grep for the printed path)
    TRAIN_LOG="$OUTDIR/train_${TAG}.log"
    python $TRAIN_SCRIPT \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" \
        --anchor llama2 \
        --gamma $GAMMA --alpha $ALPHA --beta 1.0 --epsilon $EPS --delta $DELTA \
        --stage2_steps $STEPS \
        --cka_scope harmful_only \
        $COMMON \
        2>&1 | tee "$TRAIN_LOG"

    # SAFE: extract adapter path from training log instead of ls -dt
    ADAPTER=$(grep -oP 'Adapter saved to: \K.*' "$TRAIN_LOG" | tail -1)
    if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
        echo "[ERROR] Could not find adapter for $TAG"
        return
    fi
    echo "[*] $TAG adapter: $ADAPTER"

    python $EVAL_SCRIPT \
        --adapter_path "$ADAPTER" \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" --anchor llama2 \
        --precision fp32 --cka_per_group --verbose --baseline \
        --save_all_responses --responses_json $OUTDIR/responses_${TAG}.json \
        --output_json $OUTDIR/eval_${TAG}.json \
        2>&1 | tee $OUTDIR/eval_${TAG}.log

    python $BENCH_SCRIPT \
        --adapter_path "$ADAPTER" \
        --defender mistral \
        --precision fp32 \
        --output_json $OUTDIR/bench_${TAG}.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_${TAG}.log
}

# ── First 3 from previous mst_abl (were corrupted) ──
run_config "mst_abl_v1"  0.15  0.0  0.0   0.0  300
run_config "mst_abl_v2"  0.3   0.0  0.3   0.0  300
run_config "mst_abl_v3"  0.2   0.0  0.2   0.0  300

# ── Grid: gamma-dominant configs (alpha=0) ──

# Very light gamma, no regularization
run_config "msw_g01_e0"   0.1  0.0  0.0  0.0  200
run_config "msw_g02_e0"   0.2  0.0  0.0  0.0  200

# Light gamma + light epsilon
run_config "msw_g01_e01"  0.1  0.0  0.1  0.0  200
run_config "msw_g01_e02"  0.1  0.0  0.2  0.0  200
run_config "msw_g02_e01"  0.2  0.0  0.1  0.0  200
run_config "msw_g02_e04"  0.2  0.0  0.4  0.0  200

# Original gamma level + various epsilon
run_config "msw_g03_e01"  0.3  0.0  0.1  0.0  200
run_config "msw_g03_e02"  0.3  0.0  0.2  0.0  200
run_config "msw_g03_e04"  0.3  0.0  0.4  0.0  200
run_config "msw_g03_e06"  0.3  0.0  0.6  0.0  200

# Stronger gamma + epsilon
run_config "msw_g05_e02"  0.5  0.0  0.2  0.0  200
run_config "msw_g05_e04"  0.5  0.0  0.4  0.0  200
run_config "msw_g05_e06"  0.5  0.0  0.6  0.0  200

# With tiny delta (LM loss)
run_config "msw_g03_e02_d02" 0.3  0.0  0.2  0.02  200
run_config "msw_g03_e04_d02" 0.3  0.0  0.4  0.02  200
run_config "msw_g05_e04_d02" 0.5  0.0  0.4  0.02  200

# With tiny alpha (very light refusal push) — still gamma-dominant
run_config "msw_g03_e02_a005" 0.3  0.05  0.2  0.0  200
run_config "msw_g03_e04_a005" 0.3  0.05  0.4  0.0  200
run_config "msw_g05_e04_a005" 0.5  0.05  0.4  0.0  200

# Fewer steps (150)
run_config "msw_g03_e02_s150" 0.3  0.0  0.2  0.0  150
run_config "msw_g05_e04_s150" 0.5  0.0  0.4  0.0  150

# ── Also benchmark gamma-strong (missing bench) ──
echo ""
echo "============================================================"
echo "  Benchmark gamma-strong (gamma=1.5)"
echo "============================================================"
# gamma-strong adapter is defender_v2_cka_20260304_122704
STRONG_ADAPTER="$OUTDIR/defender_v2_cka_20260304_122704"
if [ -d "$STRONG_ADAPTER" ]; then
    if [ -f "$OUTDIR/bench_mistral7b_abl_strong.json" ]; then
        echo "[SKIP] bench exists"
    else
        python $BENCH_SCRIPT \
            --adapter_path "$STRONG_ADAPTER" \
            --defender mistral \
            --precision fp32 \
            --output_json $OUTDIR/bench_mistral7b_abl_strong.json \
            --verbose \
            2>&1 | tee $OUTDIR/bench_mistral7b_abl_strong.log
    fi
else
    echo "[!] gamma-strong adapter not found at $STRONG_ADAPTER"
fi

# ── Summary ──
echo ""
echo "============================================================"
echo "  MISTRAL SWEEP SUMMARY"
echo "============================================================"
echo ""
printf "%-25s %6s %6s %6s %5s %6s %6s %6s\n" "Config" "ASR_s" "ASR_a" "ASR_o" "BGR" "XS" "OR" "MT"
echo "--------------------------------------------------------------------------------------------------------------"

for ef in $OUTDIR/eval_msw_*.json $OUTDIR/eval_mst_abl_v*.json; do
    if [ -f "$ef" ]; then
        TAG=$(basename "$ef" .json | sed 's/eval_//')
        python3 -c "
import json, os
e = json.load(open('$ef'))
d = e.get('defended', e)
asr_s = d.get('asr_self', -1)
asr_a = d.get('asr_anchor', -1)
asr_o = d.get('asr_other', -1)
bgr = d.get('bgr', -1)

bf = '$ef'.replace('eval_', 'bench_')
xs = or_b = mt = -1
if os.path.exists(bf):
    b = json.load(open(bf))
    bd = b.get('defended', b)
    xs = bd.get('xstest_refusal_rate', -1)
    or_b = bd.get('orbench_refusal_rate', -1)
    mt = bd.get('mtbench_score', -1)
    if isinstance(xs, float) and xs >= 0: xs = xs * 100
    if isinstance(or_b, float) and or_b >= 0: or_b = or_b * 100

def fmt_asr(v):
    if v == -1: return '?'
    if isinstance(v, float): return f'{v*100:.0f}%' if v < 1 else f'{v:.0f}%'
    return str(v)
def fmt(v):
    if v == -1: return '?'
    return f'{v:.1f}' if isinstance(v, float) else str(v)

print(f'  $TAG'.ljust(25), fmt_asr(asr_s).rjust(6), fmt_asr(asr_a).rjust(6), fmt_asr(asr_o).rjust(6), fmt(bgr).rjust(5), fmt(xs).rjust(6), fmt(or_b).rjust(6), fmt(mt).rjust(6))
" 2>/dev/null
    fi
done

echo ""
echo "[+] Done"
