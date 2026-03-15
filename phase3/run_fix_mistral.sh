#!/bin/bash
#SBATCH --job-name=fix_mistr
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_fix_mistr_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_fix_mistr_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# ============================================================
# FIX 1: Mistral baseline benchmark (was failing due to CSV)
# ============================================================
echo ""
echo "============================================================"
echo "  FIX 1: Mistral baseline benchmarks"
echo "============================================================"

# Remove failed baseline log
rm -f "$OUTDIR/bench_mistral_baseline.log"

python $BENCH_SCRIPT \
    --defender mistral \
    --baseline_only \
    --precision fp16 \
    --output_json "$OUTDIR/bench_mistral_baseline.json" \
    2>&1 | tee "$OUTDIR/bench_mistral_baseline.log"

python3 -c "
import json; d=json.load(open('$OUTDIR/bench_mistral_baseline.json'))
dd=d.get('baseline',d)
xs=dd.get('xstest_refusal_rate',None); orb=dd.get('orbench_refusal_rate',None)
mt=dd.get('mtbench_score',None); mmlu=dd.get('mmlu_accuracy',dd.get('mmlu_score',None))
parts=[]
if xs is not None: parts.append(f'XS={xs*100:.1f}%')
if orb is not None: parts.append(f'OR={orb*100:.1f}%')
if mt is not None: parts.append(f'MT={mt:.2f}')
if mmlu is not None: parts.append(f'MMLU={mmlu*100:.1f}%')
print(f'  mistral_baseline: {\" \".join(parts)}')
" 2>/dev/null || echo "  Failed to parse mistral_baseline"

# ============================================================
# FIX 2: MMLU for all configs (cache was corrupted, now cleaned)
# ============================================================
echo ""
echo "============================================================"
echo "  FIX 2: Re-download MMLU dataset"
echo "============================================================"

# Force re-download by running a quick test
python3 -c "
from datasets import load_dataset
ds = load_dataset('cais/mmlu', 'world_religions', split='test')
print(f'MMLU world_religions loaded: {len(ds)} samples')
" 2>&1

echo ""
echo "============================================================"
echo "  FIX 3: MMLU for str_l2_hx (bench may have completed without MMLU)"
echo "============================================================"

# Run MMLU-only for all 8 defended configs + baseline
# (XS/OR/MT already done for most, just need MMLU added)

run_mmlu() {
    local LABEL=$1
    local ADAPTER=$2

    # Check if bench file already has mmlu
    if [ -f "$OUTDIR/bench_${LABEL}.json" ]; then
        HAS_MMLU=$(python3 -c "
import json; d=json.load(open('$OUTDIR/bench_${LABEL}.json'))
dd=d.get('defended',d.get('baseline',d))
print('yes' if dd.get('mmlu_accuracy') or dd.get('mmlu_score') else 'no')
" 2>/dev/null)
        if [ "$HAS_MMLU" = "yes" ]; then
            echo "[SKIP] $LABEL already has MMLU"
            return
        fi
    fi

    echo ""
    echo "--- MMLU: $LABEL ---"

    local ADAPTER_ARGS=""
    if [ "$ADAPTER" != "none" ]; then
        ADAPTER_ARGS="--adapter_path $ADAPTER"
    fi

    local BASELINE_FLAG=""
    if [ "$ADAPTER" = "none" ]; then
        BASELINE_FLAG="--baseline_only"
    else
        BASELINE_FLAG="--no_baseline"
    fi

    # Run MMLU-only
    python $BENCH_SCRIPT \
        --defender mistral \
        $ADAPTER_ARGS \
        $BASELINE_FLAG \
        --precision fp16 \
        --skip_mtbench \
        --output_json "$OUTDIR/mmlu_${LABEL}.json" \
        2>&1

    # Extract MMLU score and merge into existing bench file
    python3 -c "
import json, os
mmlu_f = '$OUTDIR/mmlu_${LABEL}.json'
bench_f = '$OUTDIR/bench_${LABEL}.json'

if not os.path.exists(mmlu_f):
    print(f'  ERROR: {mmlu_f} not found')
    exit(1)

md = json.load(open(mmlu_f))
key = 'defended' if 'defended' in md else ('baseline' if 'baseline' in md else None)
if not key:
    print('  ERROR: no defended/baseline key in mmlu file')
    exit(1)

mmlu_val = md[key].get('mmlu_accuracy', md[key].get('mmlu_score'))
if mmlu_val is None:
    print('  ERROR: no MMLU score found')
    exit(1)

print(f'  {\"$LABEL\"}: MMLU={mmlu_val*100:.1f}%')

# Merge into bench file if it exists
if os.path.exists(bench_f):
    bd = json.load(open(bench_f))
    bkey = 'defended' if 'defended' in bd else ('baseline' if 'baseline' in bd else None)
    if bkey:
        bd[bkey]['mmlu_accuracy'] = mmlu_val
        json.dump(bd, open(bench_f, 'w'), indent=2)
        print(f'  Merged MMLU into {bench_f}')
" 2>/dev/null || echo "  Failed to extract MMLU for $LABEL"
}

# Baseline MMLU (already in FIX 1 above, but in case it failed there)
run_mmlu "mistral_baseline" "none"

# All 8 defended configs
run_mmlu "mistral_anc_l2_hx"  "$OUTDIR/defender_v2_cka_20260309_110603"
run_mmlu "mistral_safe_l2_hx" "$OUTDIR/defender_v2_cka_20260309_140723"
run_mmlu "mistral_anc_p_hx"   "$OUTDIR/defender_v2_cka_20260309_123015"
run_mmlu "mistral_anc_y_ax"   "$OUTDIR/defender_v2_cka_20260309_130114"
run_mmlu "mistral_safe_p_ax"  "$OUTDIR/defender_v2_cka_20260309_151606"
run_mmlu "mistral_gen_l2_ax"  "$OUTDIR/defender_v2_cka_20260309_170430"
run_mmlu "mistral_gen_q_ax"   "$OUTDIR/defender_v2_cka_20260309_173545"
run_mmlu "mistral_str_l2_hx"  "$OUTDIR/defender_v2_cka_20260309_103050"

# ============================================================
# FIX 4: str_l2_hx full bench (if it didn't complete in the other job)
# ============================================================
if [ ! -f "$OUTDIR/bench_mistral_str_l2_hx.json" ]; then
    echo ""
    echo "============================================================"
    echo "  FIX 4: Full bench for mistral_str_l2_hx"
    echo "============================================================"
    python $BENCH_SCRIPT \
        --defender mistral \
        --adapter_path "$OUTDIR/defender_v2_cka_20260309_103050" \
        --precision fp16 \
        --no_baseline \
        --output_json "$OUTDIR/bench_mistral_str_l2_hx.json" \
        2>&1 | tee "$OUTDIR/bench_mistral_str_l2_hx.log"

    python3 -c "
import json; d=json.load(open('$OUTDIR/bench_mistral_str_l2_hx.json'))
dd=d.get('defended',d)
xs=dd.get('xstest_refusal_rate',None); orb=dd.get('orbench_refusal_rate',None)
mt=dd.get('mtbench_score',None)
parts=[]
if xs is not None: parts.append(f'XS={xs*100:.1f}%')
if orb is not None: parts.append(f'OR={orb*100:.1f}%')
if mt is not None: parts.append(f'MT={mt:.2f}')
print(f'  mistral_str_l2_hx: {\" \".join(parts)}')
" 2>/dev/null || echo "  Failed to parse mistral_str_l2_hx"
fi

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "  FINAL SUMMARY"
echo "============================================================"

python3 -c "
import json, os, glob

OUTDIR = '$OUTDIR'

# Load baseline
base_xs, base_or, base_mt, base_mmlu = None, None, None, None
try:
    bd = json.load(open(f'{OUTDIR}/bench_mistral_baseline.json'))
    bb = bd.get('baseline', bd)
    base_xs = bb.get('xstest_refusal_rate')
    base_or = bb.get('orbench_refusal_rate')
    base_mt = bb.get('mtbench_score')
    base_mmlu = bb.get('mmlu_accuracy', bb.get('mmlu_score'))
    print(f'Baseline: XS={base_xs*100:.1f}% OR={base_or*100:.1f}% MT={base_mt:.2f} MMLU={base_mmlu*100:.1f}%' if all(v is not None for v in [base_xs,base_or,base_mt,base_mmlu]) else f'Baseline: partial')
except:
    print('Baseline: MISSING')

configs = [
    'mistral_anc_l2_hx', 'mistral_safe_l2_hx', 'mistral_anc_p_hx',
    'mistral_anc_y_ax', 'mistral_safe_p_ax', 'mistral_gen_l2_ax',
    'mistral_gen_q_ax', 'mistral_str_l2_hx'
]

# Get ASR from eval files
print()
print(f'{\"Config\":<28} {\"ASR(s/a/o)\":<16} {\"XS(Δ)\":<12} {\"OR(Δ)\":<12} {\"MT(Δ)\":<12} {\"MMLU(Δ)\":<12}')
print('='*92)

for c in configs:
    # ASR
    asr_s = '?'
    try:
        ed = json.load(open(f'{OUTDIR}/eval_{c}.json'))
        dd = ed.get('defended', ed)
        s = dd.get('asr_self',0); a = dd.get('asr_anchor',0); o = dd.get('asr_other',0)
        if isinstance(s,float) and s<=1: s*=100
        if isinstance(a,float) and a<=1: a*=100
        if isinstance(o,float) and o<=1: o*=100
        asr_s = f'{s:.0f}/{a:.0f}/{o:.0f}%'
    except: pass

    # Bench
    xs_s, or_s, mt_s, mmlu_s = '-', '-', '-', '-'
    try:
        bd = json.load(open(f'{OUTDIR}/bench_{c}.json'))
        bdd = bd.get('defended', bd)
        xs = bdd.get('xstest_refusal_rate')
        orb = bdd.get('orbench_refusal_rate')
        mt = bdd.get('mtbench_score')
        mmlu = bdd.get('mmlu_accuracy', bdd.get('mmlu_score'))

        if xs is not None:
            d = f'({(xs-base_xs)*100:+.1f})' if base_xs else ''
            xs_s = f'{xs*100:.1f}%{d}'
        if orb is not None:
            d = f'({(orb-base_or)*100:+.1f})' if base_or else ''
            or_s = f'{orb*100:.1f}%{d}'
        if mt is not None:
            d = f'({mt-base_mt:+.2f})' if base_mt else ''
            mt_s = f'{mt:.2f}{d}'
        if mmlu is not None:
            d = f'({(mmlu-base_mmlu)*100:+.1f})' if base_mmlu else ''
            mmlu_s = f'{mmlu*100:.1f}%{d}'
    except: pass

    print(f'{c:<28} {asr_s:<16} {xs_s:<12} {or_s:<12} {mt_s:<12} {mmlu_s:<12}')
"

echo ""
echo "[+] Fix script complete"
