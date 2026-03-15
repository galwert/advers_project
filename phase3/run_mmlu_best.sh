#!/bin/bash
#SBATCH --job-name=mmlu_best
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_mmlu_best_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_mmlu_best_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

run_mmlu() {
    local LABEL=$1
    local DEFENDER=$2
    local ADAPTER=$3

    local OUT="$OUTDIR/mmlu_${LABEL}.json"
    if [ -f "$OUT" ]; then
        echo "[SKIP] mmlu_${LABEL}.json already exists"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  MMLU: $LABEL (defender=$DEFENDER)"
    echo "============================================================"

    local ADAPTER_ARG=""
    if [ "$ADAPTER" != "none" ]; then
        ADAPTER_ARG="--adapter_path $ADAPTER"
    fi

    python $BENCH_SCRIPT \
        --defender $DEFENDER \
        $ADAPTER_ARG \
        --skip_mtbench \
        --precision fp16 \
        --no_baseline \
        --output_json "$OUT" \
        2>&1 | tee "$OUTDIR/mmlu_${LABEL}.log"

    python3 -c "
import json; d=json.load(open('$OUT'))
dd=d.get('defended',d)
mmlu=dd.get('mmlu_accuracy',None)
xs=dd.get('xstest_refusal_rate',None)
orb=dd.get('orbench_refusal_rate',None)
parts=[]
if mmlu is not None: parts.append(f'MMLU={mmlu*100:.1f}%')
else: parts.append('MMLU=MISSING')
print(f'  $LABEL: {\" \".join(parts)}')
" 2>/dev/null || echo "  Failed to parse $LABEL"
}

# ============================================================
# BASELINES (missing MMLU)
# ============================================================
run_mmlu "vicuna_baseline"  vicuna        none
run_mmlu "nemo_baseline"    mistral_nemo  none
run_mmlu "qwen14b_baseline" qwen-14b      none
run_mmlu "mistral_baseline" mistral       none

# ============================================================
# BEST DEFENDED CONFIGS
# ============================================================
run_mmlu "qwen_ab2_hx"      qwen         "$OUTDIR/defender_v2_cka_20260307_094710"
run_mmlu "vicuna_ab2_ax"     vicuna       "$OUTDIR/defender_v2_cka_20260307_115715"
run_mmlu "llama3_ab2_bx"     llama3       "$OUTDIR/defender_v2_cka_20260307_123357"
run_mmlu "yi9b_ab2_an"       yi9b         "$OUTDIR/defender_v2_cka_20260307_141547"
run_mmlu "nemo_ab2_ax"       mistral_nemo "$OUTDIR/defender_v2_cka_20260307_161510"
run_mmlu "qwen14b_q14_c"     qwen-14b     "$OUTDIR/defender_v2_cka_20260307_140824"

# Best Mistral configs (pick top 2 from completed results)
run_mmlu "mistral_anc_l2_hx"  mistral    "$OUTDIR/defender_v2_cka_20260309_110603"
run_mmlu "mistral_anc_p_hx"   mistral    "$OUTDIR/defender_v2_cka_20260309_123015"

echo ""
echo "============================================================"
echo "  MMLU RUNS COMPLETE"
echo "============================================================"

echo ""
printf "%-26s  %8s\n" "Config" "MMLU"
echo "--------------------------  --------"
for label in vicuna_baseline nemo_baseline qwen14b_baseline mistral_baseline \
             qwen_ab2_hx vicuna_ab2_ax llama3_ab2_bx yi9b_ab2_an nemo_ab2_ax \
             qwen14b_q14_c mistral_anc_l2_hx mistral_anc_p_hx; do
    f="$OUTDIR/mmlu_${label}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json; d=json.load(open('$f'))
dd=d.get('defended',d)
mmlu=dd.get('mmlu_accuracy',None)
print(f'  ${label:<26s}  {mmlu*100:.1f}%' if mmlu else f'  ${label:<26s}  MISSING')
" 2>/dev/null
    else
        printf "  %-26s  %8s\n" "$label" "--"
    fi
done
