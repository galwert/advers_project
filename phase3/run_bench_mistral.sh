#!/bin/bash
#SBATCH --job-name=bench_mist
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bench_mist_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bench_mist_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

run_bench() {
    local LABEL=$1
    local DEFENDER=$2
    local ADAPTER=$3

    local OUT="$OUTDIR/bench_${LABEL}.json"
    if [ -f "$OUT" ]; then
        echo "[SKIP] bench_${LABEL}.json already exists"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  BENCH: $LABEL (defender=$DEFENDER)"
    echo "============================================================"

    local ADAPTER_ARG=""
    if [ "$ADAPTER" != "none" ]; then
        ADAPTER_ARG="--adapter_path $ADAPTER"
    fi

    python $BENCH_SCRIPT \
        --defender $DEFENDER \
        $ADAPTER_ARG \
        --precision fp16 \
        --no_baseline \
        --output_json "$OUT" \
        2>&1 | tee "$OUTDIR/bench_${LABEL}.log"

    python3 -c "
import json; d=json.load(open('$OUT'))
dd=d.get('defended',d)
xs=dd.get('xstest_refusal_rate',None); orb=dd.get('orbench_refusal_rate',None)
mt=dd.get('mtbench_score',None); mmlu=dd.get('mmlu_accuracy',None)
parts=[]
if mmlu is not None: parts.append(f'MMLU={mmlu*100:.1f}%')
if xs is not None: parts.append(f'XS={xs*100:.1f}%')
if orb is not None: parts.append(f'OR={orb*100:.1f}%')
if mt is not None: parts.append(f'MT={mt:.2f}')
print(f'  $LABEL: {\" \".join(parts)}')
" 2>/dev/null || echo "  Failed to parse $LABEL"
}

# Mistral baseline (full benchmarks)
run_bench "mistral_baseline" mistral none

# Top Mistral defended configs (0/0/0% ASR, 0% BGR)
run_bench "mistral_anc_l2_hx"  mistral "$OUTDIR/defender_v2_cka_20260309_110603"
run_bench "mistral_safe_l2_hx" mistral "$OUTDIR/defender_v2_cka_20260309_140723"
run_bench "mistral_anc_p_hx"   mistral "$OUTDIR/defender_v2_cka_20260309_123015"
run_bench "mistral_anc_y_ax"   mistral "$OUTDIR/defender_v2_cka_20260309_130114"
run_bench "mistral_safe_p_ax"  mistral "$OUTDIR/defender_v2_cka_20260309_151606"

echo ""
echo "============================================================"
echo "  MISTRAL BENCHMARKS COMPLETE"
echo "============================================================"

printf "\n%-26s  %6s  %6s  %6s  %6s\n" "Config" "MMLU" "XS" "OR" "MT"
echo "--------------------------  ------  ------  ------  ------"
for label in mistral_baseline mistral_anc_l2_hx mistral_safe_l2_hx mistral_anc_p_hx mistral_anc_y_ax mistral_safe_p_ax; do
    f="$OUTDIR/bench_${label}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json; d=json.load(open('$f'))
dd=d.get('defended',d)
mmlu=dd.get('mmlu_accuracy',0)*100; xs=dd.get('xstest_refusal_rate',0)*100
orb=dd.get('orbench_refusal_rate',0)*100; mt=dd.get('mtbench_score',0)
print(f'  ${label:<26s}  {mmlu:5.1f}%  {xs:5.1f}%  {orb:5.1f}%  {mt:5.2f}')
" 2>/dev/null
    else
        printf "  %-26s  %6s  %6s  %6s  %6s\n" "$label" "--" "--" "--" "--"
    fi
done
