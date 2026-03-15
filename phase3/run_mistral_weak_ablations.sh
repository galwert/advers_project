#!/bin/bash
#SBATCH --job-name=mst_abl
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_mst_abl_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_mst_abl_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

TRAIN_SCRIPT="two_stage_defense_v2.py"
EVAL_SCRIPT="evaluate_v2.py"
BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"
GCG_DATA="$OUTDIR/gcg_strings_mistral.json"

COMMON="--alignment cka --use_borderline --precision fp32 \
        --target_layer_pct 0.5 --lora_r 32 --stage2_lr 2e-4 \
        --gcg_data_path $GCG_DATA --output_dir $OUTDIR"

# The current gamma-weak (gamma=0.3, alpha=0, eps=0) has MMLU=41.2% — catastrophic.
# Goal: find a gamma-dominant config that still preserves quality.
# Strategy: try lower gamma and/or add minimal KL regularization.

# ── Variant 1: gamma=0.15, alpha=0, eps=0 (half the gamma) ──
echo ""
echo "============================================================"
echo "  Variant 1: gamma=0.15, alpha=0, eps=0"
echo "============================================================"
if [ -f "$OUTDIR/eval_mst_abl_v1.json" ]; then
    echo "[SKIP] eval exists"
else
    python $TRAIN_SCRIPT \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" \
        --anchor llama2 \
        --gamma 0.15 --alpha 0 --beta 1.0 --epsilon 0 --delta 0 \
        --stage2_steps 300 \
        --cka_scope harmful_only \
        $COMMON \
        2>&1 | tee $OUTDIR/train_mst_abl_v1.log

    V1_ADAPTER=$(ls -dt $OUTDIR/defender_v2_cka_* | head -1)
    echo "[*] V1 adapter: $V1_ADAPTER"

    python $EVAL_SCRIPT \
        --adapter_path "$V1_ADAPTER" \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" --anchor llama2 \
        --precision fp32 --cka_per_group --verbose --baseline \
        --save_all_responses --responses_json $OUTDIR/responses_mst_abl_v1.json \
        --output_json $OUTDIR/eval_mst_abl_v1.json \
        2>&1 | tee $OUTDIR/eval_mst_abl_v1.log

    python $BENCH_SCRIPT \
        --adapter_path "$V1_ADAPTER" \
        --defender mistral \
        --precision fp32 \
        --output_json $OUTDIR/bench_mst_abl_v1.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_mst_abl_v1.log
fi

# ── Variant 2: gamma=0.3, alpha=0, eps=0.3 (original gamma + light KL) ──
echo ""
echo "============================================================"
echo "  Variant 2: gamma=0.3, alpha=0, eps=0.3"
echo "============================================================"
if [ -f "$OUTDIR/eval_mst_abl_v2.json" ]; then
    echo "[SKIP] eval exists"
else
    python $TRAIN_SCRIPT \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" \
        --anchor llama2 \
        --gamma 0.3 --alpha 0 --beta 1.0 --epsilon 0.3 --delta 0 \
        --stage2_steps 300 \
        --cka_scope harmful_only \
        $COMMON \
        2>&1 | tee $OUTDIR/train_mst_abl_v2.log

    V2_ADAPTER=$(ls -dt $OUTDIR/defender_v2_cka_* | head -1)
    echo "[*] V2 adapter: $V2_ADAPTER"

    python $EVAL_SCRIPT \
        --adapter_path "$V2_ADAPTER" \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" --anchor llama2 \
        --precision fp32 --cka_per_group --verbose --baseline \
        --save_all_responses --responses_json $OUTDIR/responses_mst_abl_v2.json \
        --output_json $OUTDIR/eval_mst_abl_v2.json \
        2>&1 | tee $OUTDIR/eval_mst_abl_v2.log

    python $BENCH_SCRIPT \
        --adapter_path "$V2_ADAPTER" \
        --defender mistral \
        --precision fp32 \
        --output_json $OUTDIR/bench_mst_abl_v2.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_mst_abl_v2.log
fi

# ── Variant 3: gamma=0.2, alpha=0, eps=0.2 (compromise) ──
echo ""
echo "============================================================"
echo "  Variant 3: gamma=0.2, alpha=0, eps=0.2"
echo "============================================================"
if [ -f "$OUTDIR/eval_mst_abl_v3.json" ]; then
    echo "[SKIP] eval exists"
else
    python $TRAIN_SCRIPT \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" \
        --anchor llama2 \
        --gamma 0.2 --alpha 0 --beta 1.0 --epsilon 0.2 --delta 0 \
        --stage2_steps 300 \
        --cka_scope harmful_only \
        $COMMON \
        2>&1 | tee $OUTDIR/train_mst_abl_v3.log

    V3_ADAPTER=$(ls -dt $OUTDIR/defender_v2_cka_* | head -1)
    echo "[*] V3 adapter: $V3_ADAPTER"

    python $EVAL_SCRIPT \
        --adapter_path "$V3_ADAPTER" \
        --defender "mistralai/Mistral-7B-Instruct-v0.2" --anchor llama2 \
        --precision fp32 --cka_per_group --verbose --baseline \
        --save_all_responses --responses_json $OUTDIR/responses_mst_abl_v3.json \
        --output_json $OUTDIR/eval_mst_abl_v3.json \
        2>&1 | tee $OUTDIR/eval_mst_abl_v3.log

    python $BENCH_SCRIPT \
        --adapter_path "$V3_ADAPTER" \
        --defender mistral \
        --precision fp32 \
        --output_json $OUTDIR/bench_mst_abl_v3.json \
        --verbose \
        2>&1 | tee $OUTDIR/bench_mst_abl_v3.log
fi

# ── Also benchmark gamma-strong (missing MT-Bench and MMLU) ──
echo ""
echo "============================================================"
echo "  Benchmark gamma-strong (gamma=1.5)"
echo "============================================================"
STRONG_ADAPTER=$(ls -dt $OUTDIR/defender_v2_cka_20260304_12* 2>/dev/null | head -1)
if [ -z "$STRONG_ADAPTER" ]; then
    # Find from training log
    STRONG_ADAPTER=$(grep "Saved adapter" $OUTDIR/train_mistral7b_abl_strong.log 2>/dev/null | tail -1 | grep -oP '/\S+')
fi
# Try to find by checking adapter directories for matching config
if [ -z "$STRONG_ADAPTER" ] || [ ! -d "$STRONG_ADAPTER" ]; then
    echo "[*] Looking for gamma-strong adapter..."
    for d in $OUTDIR/defender_v2_cka_*; do
        if [ -f "$d/adapter_config.json" ]; then
            gamma=$(python3 -c "import json; d=json.load(open('$d/training_metrics.json')); print(d.get('gamma', d.get('hyperparameters',{}).get('gamma','?')))" 2>/dev/null)
            alpha=$(python3 -c "import json; d=json.load(open('$d/training_metrics.json')); print(d.get('alpha', d.get('hyperparameters',{}).get('alpha','?')))" 2>/dev/null)
            if [ "$gamma" = "1.5" ] && [ "$alpha" = "0.0" ]; then
                STRONG_ADAPTER="$d"
                echo "[*] Found gamma-strong adapter: $d"
                break
            fi
        fi
    done
fi
# Also check saved_results/final_archive
if [ -z "$STRONG_ADAPTER" ] || [ ! -d "$STRONG_ADAPTER" ]; then
    if [ -d "./saved_results/final_archive/adapters/mistral7b_abl_strong" ]; then
        STRONG_ADAPTER="./saved_results/final_archive/adapters/mistral7b_abl_strong"
    fi
fi

if [ -n "$STRONG_ADAPTER" ] && [ -d "$STRONG_ADAPTER" ]; then
    if [ -f "$OUTDIR/bench_mistral7b_abl_strong.json" ]; then
        echo "[SKIP] bench exists"
    else
        echo "[*] Benchmarking gamma-strong from: $STRONG_ADAPTER"
        python $BENCH_SCRIPT \
            --adapter_path "$STRONG_ADAPTER" \
            --defender mistral \
            --precision fp32 \
            --output_json $OUTDIR/bench_mistral7b_abl_strong.json \
            --verbose \
            2>&1 | tee $OUTDIR/bench_mistral7b_abl_strong.log
    fi
else
    echo "[!] Could not find gamma-strong adapter"
fi

# ── Summary ──
echo ""
echo "============================================================"
echo "  MISTRAL WEAK ABLATION SUMMARY"
echo "============================================================"
for v in v1 v2 v3; do
    ef="$OUTDIR/eval_mst_abl_${v}.json"
    bf="$OUTDIR/bench_mst_abl_${v}.json"
    if [ -f "$ef" ]; then
        python3 -c "
import json
e = json.load(open('$ef'))
d = e.get('defended', e)
asr_s = d.get('asr_self', '?')
asr_a = d.get('asr_anchor', '?')
asr_o = d.get('asr_other', '?')
bgr = d.get('bgr', '?')
ppl = d.get('ppl', '?')
cka = d.get('cka_score', '?')
print(f'  $v: ASR={asr_s}/{asr_a}/{asr_o} BGR={bgr} PPL={ppl} CKA={cka}')
" 2>/dev/null
    else
        echo "  $v: eval MISSING"
    fi
    if [ -f "$bf" ]; then
        python3 -c "
import json
b = json.load(open('$bf'))
d = b.get('defended', b)
mt = d.get('mtbench_score', '?')
xs = d.get('xstest_refusal_rate', '?')
orb = d.get('orbench_refusal_rate', '?')
print(f'       MT={mt} XS={xs} OR={orb}')
" 2>/dev/null
    fi
done

echo ""
echo "[+] Done"
