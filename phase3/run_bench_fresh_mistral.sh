#!/bin/bash
#SBATCH --job-name=bfr_mistr
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_bfr_mistr_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_bfr_mistr_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

BENCH_SCRIPT="benchmark_eval.py"
OUTDIR="./7b_defense_wildguard_outputs"

# ============================================================
# PART 1: BENCHMARKS (XS, OR, MT, MMLU) for all 0/0/0% Mistral configs
# ============================================================

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
    echo "  BENCH: $LABEL"
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

# Baseline
run_bench "mistral_baseline" mistral none

# All 8 configs with 0/0/0% ASR
run_bench "mistral_anc_l2_hx"  mistral "$OUTDIR/defender_v2_cka_20260309_110603"
run_bench "mistral_safe_l2_hx" mistral "$OUTDIR/defender_v2_cka_20260309_140723"
run_bench "mistral_anc_p_hx"   mistral "$OUTDIR/defender_v2_cka_20260309_123015"
run_bench "mistral_anc_y_ax"   mistral "$OUTDIR/defender_v2_cka_20260309_130114"
run_bench "mistral_safe_p_ax"  mistral "$OUTDIR/defender_v2_cka_20260309_151606"
run_bench "mistral_gen_l2_ax"  mistral "$OUTDIR/defender_v2_cka_20260309_170430"
run_bench "mistral_gen_q_ax"   mistral "$OUTDIR/defender_v2_cka_20260309_173545"
run_bench "mistral_str_l2_hx"  mistral "$OUTDIR/defender_v2_cka_20260309_180051"

# ============================================================
# PART 2: FRESH HELD-OUT ASR for Mistral
# Uses the same approach as run_fresh_eval.sh — 100 attacks from
# models NOT used as self or anchor, evaluated with WildGuard
# ============================================================

echo ""
echo "============================================================"
echo "  FRESH ASR: Mistral configs"
echo "============================================================"

python3 << 'PYEOF'
import os, sys, json, csv, random, torch, gc
sys.modules["flash_attn"] = None
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "phase2"))
sys.path.insert(0, "../phase2")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llm_judge import load_judge, classify_response as llm_judge_classify, setup_gibberish_detector

OUTDIR = "./7b_defense_wildguard_outputs"
GCG_CSV = "../outputs/advbench_suffixes_all_models_fixed.csv"

MODEL_IDS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

# Mistral = index 3. Its anchors across configs: llama2(0), qwen(10), llama3(1), yi9b(11), phi2(9)
# Exclude self(3) and all anchors used → use indices NOT in {0,1,3,9,10,11}
# Available: vicuna(2), zephyr(4), starling(6), orca(15)
FRESH_INDICES = [2, 4, 6, 15]

INDEX_NAMES = {
    0: "Llama2-7b", 1: "Llama3-8b", 2: "Vicuna-7b", 3: "Mistral-7b",
    4: "Zephyr-7b", 6: "Starling-7b", 9: "Phi-2", 10: "Qwen1.5-7b",
    11: "Yi-6b", 15: "Orca-2-7b",
}

def format_prompt(prompt, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

def generate(model, tokenizer, formatted, max_new=256):
    enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# Load GCG data
print("[*] Loading GCG data...")
import pandas as pd
df = pd.read_csv(GCG_CSV)

# Collect 100 fresh attacks (25 per fresh source model)
random.seed(42)
fresh_attacks = []
for idx in FRESH_INDICES:
    model_df = df[df["model_index"] == idx]
    if len(model_df) == 0:
        print(f"  WARNING: no attacks for index {idx}")
        continue
    sampled = model_df.sample(n=min(25, len(model_df)), random_state=42)
    for _, row in sampled.iterrows():
        fresh_attacks.append({
            "prompt": row["prompt"],
            "suffix": row["suffix"],
            "source_model": INDEX_NAMES.get(idx, f"idx{idx}"),
        })
random.shuffle(fresh_attacks)
print(f"[+] {len(fresh_attacks)} fresh attacks from {len(FRESH_INDICES)} source models")

# Configs to evaluate: baseline + 8 defended
CONFIGS = [
    ("mistral_baseline", None),
    ("mistral_anc_l2_hx",  f"{OUTDIR}/defender_v2_cka_20260309_110603"),
    ("mistral_safe_l2_hx", f"{OUTDIR}/defender_v2_cka_20260309_140723"),
    ("mistral_anc_p_hx",   f"{OUTDIR}/defender_v2_cka_20260309_123015"),
    ("mistral_anc_y_ax",   f"{OUTDIR}/defender_v2_cka_20260309_130114"),
    ("mistral_safe_p_ax",  f"{OUTDIR}/defender_v2_cka_20260309_151606"),
    ("mistral_gen_l2_ax",  f"{OUTDIR}/defender_v2_cka_20260309_170430"),
    ("mistral_gen_q_ax",   f"{OUTDIR}/defender_v2_cka_20260309_173545"),
    ("mistral_str_l2_hx",  f"{OUTDIR}/defender_v2_cka_20260309_180051"),
]

# Load judge
print("[*] Loading WildGuard judge...")
judge_model_id = "allenai/wildguard"
judge_model, judge_tokenizer = load_judge(judge_model_id, quantize=False)
gib_detector = setup_gibberish_detector()

model_id = MODEL_IDS["mistral"]

for label, adapter_path in CONFIGS:
    out_file = f"{OUTDIR}/fresh_eval_{label}.json"
    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} exists")
        continue

    print(f"\n=== {label} ===")

    # Unload judge temporarily, load target
    judge_model.cpu()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    # Generate responses
    responses = []
    for i, atk in enumerate(fresh_attacks):
        full = f"{atk['prompt']} {atk['suffix']}"
        formatted = format_prompt(full, tokenizer)
        resp = generate(model, tokenizer, formatted)
        responses.append({"prompt": atk["prompt"], "suffix": atk["suffix"],
                          "source": atk["source_model"], "response": resp})
        if (i+1) % 25 == 0:
            print(f"  Generated {i+1}/{len(fresh_attacks)}")

    # Free target, bring judge back
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    judge_model.cuda()

    # Judge
    success = 0
    for r in responses:
        is_success, reason, meta = llm_judge_classify(
            r["response"], r["prompt"], judge_model, judge_tokenizer,
            judge_model_id, gib_detector, suffix=r["suffix"]
        )
        r["judge_success"] = is_success
        r["judge_reason"] = reason
        if is_success:
            success += 1

    asr = success / len(responses) * 100
    print(f"  {label}: {asr:.1f}% ASR ({success}/{len(responses)})")

    with open(out_file, "w") as f:
        json.dump({"label": label, "asr": asr, "success": success,
                   "total": len(responses), "results": responses}, f, indent=2)

# Cleanup
del judge_model, judge_tokenizer
gc.collect(); torch.cuda.empty_cache()

print("\n=== FRESH ASR SUMMARY ===")
for label, _ in CONFIGS:
    f = f"{OUTDIR}/fresh_eval_{label}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"  {label}: {d['asr']:.1f}% ({d['success']}/{d['total']})")
PYEOF

echo ""
echo "============================================================"
echo "  MISTRAL BENCH + FRESH ASR COMPLETE"
echo "============================================================"

# Print benchmark summary
printf "\n%-26s  %6s  %6s  %6s  %6s  %8s\n" "Config" "MMLU" "XS" "OR" "MT" "FreshASR"
echo "--------------------------  ------  ------  ------  ------  --------"
for label in mistral_baseline mistral_anc_l2_hx mistral_safe_l2_hx mistral_anc_p_hx mistral_anc_y_ax mistral_safe_p_ax mistral_gen_l2_ax mistral_gen_q_ax mistral_str_l2_hx; do
    bench="$OUTDIR/bench_${label}.json"
    fresh="$OUTDIR/fresh_eval_${label}.json"
    python3 -c "
import json
b_str = '--'
if __import__('os').path.exists('$bench'):
    d=json.load(open('$bench'))
    dd=d.get('defended',d)
    mmlu=dd.get('mmlu_accuracy',0)*100; xs=dd.get('xstest_refusal_rate',0)*100
    orb=dd.get('orbench_refusal_rate',0)*100; mt=dd.get('mtbench_score',0)
    b_str = f'{mmlu:5.1f}%  {xs:5.1f}%  {orb:5.1f}%  {mt:5.2f}'
else:
    b_str = '    --      --      --      --'
f_str = '--'
if __import__('os').path.exists('$fresh'):
    d=json.load(open('$fresh'))
    f_str = f'{d[\"asr\"]:.1f}%'
print(f'  ${label:<26s}  {b_str}  {f_str:>8s}')
" 2>/dev/null
done
