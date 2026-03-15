#!/bin/bash
#SBATCH --job-name=bch_mistr2
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bch_mistr2_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bch_mistr2_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

run_bench() {
    local LABEL=$1
    local ADAPTER=$2

    local OUT="$OUTDIR/bench_${LABEL}.json"
    if [ -f "$OUT" ]; then
        echo "[SKIP] bench_${LABEL}.json already exists"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  BENCH: $LABEL"
    echo "============================================================"

    python $BENCH_SCRIPT \
        --defender mistral \
        --adapter_path "$ADAPTER" \
        --precision fp16 \
        --no_baseline \
        --skip_mmlu \
        --output_json "$OUT" \
        2>&1 | tee "$OUTDIR/bench_${LABEL}.log"

    python3 -c "
import json; d=json.load(open('$OUT'))
dd=d.get('defended',d)
xs=dd.get('xstest_refusal_rate',None); orb=dd.get('orbench_refusal_rate',None)
mt=dd.get('mtbench_score',None)
parts=[]
if xs is not None: parts.append(f'XS={xs*100:.1f}%')
if orb is not None: parts.append(f'OR={orb*100:.1f}%')
if mt is not None: parts.append(f'MT={mt:.2f}')
print(f'  $LABEL: {\" \".join(parts)}')
" 2>/dev/null || echo "  Failed to parse $LABEL"
}

# 21 unbenchmarked Mistral configs (sorted by ASR)
run_bench "mistral_gen_q_hx" "$OUTDIR/defender_v2_cka_20260309_163034"
run_bench "mistral_safe_q_hx" "$OUTDIR/defender_v2_cka_20260309_143225"
run_bench "mistral_safe_y_ax" "$OUTDIR/defender_v2_cka_20260309_144636"
run_bench "mistral7b_r1a" "$OUTDIR/defender_v2_cka_20260304_104303"
run_bench "mistral_anc_l3_ax" "$OUTDIR/defender_v2_cka_20260309_134210"
run_bench "mistral_anc_q_hx" "$OUTDIR/defender_v2_cka_20260309_113358"
run_bench "mistral_safe_p_hx" "$OUTDIR/defender_v2_cka_20260309_141650"
run_bench "mistral_gen_y_hx" "$OUTDIR/defender_v2_cka_20260309_153824"
run_bench "mistral_safe_y_hx" "$OUTDIR/defender_v2_cka_20260309_135416"
run_bench "mistral_anc_l2_ax" "$OUTDIR/defender_v2_cka_20260309_123550"
run_bench "mistral_gen_l2_hx" "$OUTDIR/defender_v2_cka_20260309_160531"
run_bench "mistral_gen_p_hx" "$OUTDIR/defender_v2_cka_20260309_160033"
run_bench "mistral7b_abl_weak" "$OUTDIR/defender_v2_cka_20260304_120459"
run_bench "mistral_str_q_hx" "$OUTDIR/defender_v2_cka_20260309_182516"
run_bench "mistral7b_abl_strong" "$OUTDIR/defender_v2_cka_20260304_122704"
run_bench "mistral_anc_y_hx" "$OUTDIR/defender_v2_cka_20260309_120639"
run_bench "mistral_anc_l3_hx" "$OUTDIR/defender_v2_cka_20260309_120123"
run_bench "mistral_anc_p_ax" "$OUTDIR/defender_v2_cka_20260309_133139"
run_bench "mistral_anc_q_ax" "$OUTDIR/defender_v2_cka_20260309_130955"
run_bench "mistral_safe_q_ax" "$OUTDIR/defender_v2_cka_20260309_154040"
run_bench "mistral_safe_l2_ax" "$OUTDIR/defender_v2_cka_20260309_150613"

# Summary
echo ""
echo "============================================================"
echo "  SUMMARY (all Mistral configs vs baseline XS=8.0% OR=22.1% MT=6.38)"
echo "============================================================"

python3 << 'PYEOF2'
import json, glob, os

OUTDIR = "./7b_defense_wildguard_outputs"
BL_XS, BL_OR, BL_MT = 8.0, 22.1, 6.38

configs = []
for f in sorted(glob.glob(os.path.join(OUTDIR, "eval_mistral*.json"))):
    name = os.path.basename(f).replace("eval_","").replace(".json","")
    if "baseline" in name: continue
    d = json.load(open(f))
    if "defended" not in d: continue
    dd = d["defended"]
    asr_s = round(dd.get("asr_self",0)*100,1)
    asr_a = round(dd.get("asr_anchor",0)*100,1)
    asr_o = round(dd.get("asr_other",0)*100,1)
    if max(asr_s,asr_a,asr_o) > 20: continue
    bgr = round(dd.get("bgr",0)*100,1)
    if bgr > 5: continue

    bench_f = os.path.join(OUTDIR, f"bench_{name}.json")
    if not os.path.exists(bench_f): continue
    bd = json.load(open(bench_f))
    bdd = bd.get("defended", bd)
    xs_r = bdd.get("xstest_refusal_rate")
    or_r = bdd.get("orbench_refusal_rate")
    mt_r = bdd.get("mtbench_score")
    if xs_r is None or or_r is None or mt_r is None: continue

    xs, orb, mt = round(xs_r*100,1), round(or_r*100,1), round(mt_r,2)
    d_xs, d_or, d_mt = round(xs-BL_XS,1), round(orb-BL_OR,1), round(mt-BL_MT,2)
    cost = round(abs(d_xs)+abs(d_or)+abs(d_mt)*15, 1)

    configs.append((cost, name, asr_s, asr_a, asr_o, xs, d_xs, orb, d_or, mt, d_mt))

configs.sort()
print(f"{'Config':<28} {'ASR(s/a/o)':<15} {'XS(Δ)':<13} {'OR(Δ)':<13} {'MT(Δ)':<12} {'Cost'}")
print("-"*90)
for cost, name, s, a, o, xs, dxs, orb, dor, mt, dmt in configs:
    print(f"{name:<28} {s:.0f}/{a:.0f}/{o:.0f}%{'':<8} {xs:.1f}%({dxs:+.1f}){'':<3} {orb:.1f}%({dor:+.1f}){'':<3} {mt:.2f}({dmt:+.2f}){'':<2} {cost}")
PYEOF2

echo ""
echo "[+] Done"
