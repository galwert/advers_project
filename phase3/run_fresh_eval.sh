#!/bin/bash
#SBATCH --job-name=fresh_eval
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_fresh_eval_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_fresh_eval_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs"

# ============================================================
# Step 1: Create fresh held-out GCG attack CSV (100 per defender)
# Uses attacks from models NOT used as self or anchor during training
# ============================================================
python3 -c "
import csv, random
random.seed(42)

# Read all attacks
attacks = []
with open('../outputs/advbench_suffixes_all_models_fixed.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        attacks.append(row)

# For each defender, pick 100 attacks from models that are NOT self or anchor
# This ensures zero overlap with training GCG data
configs = {
    # defender: (self_model_name, anchor_model_name)
    'qwen':         ('Qwen1.5-7b', 'Llama3-8b'),
    'vicuna':       ('Vicuna-7b', 'Qwen1.5-7b'),
    'llama3':       ('Llama3-8b', 'Qwen1.5-7b'),
    'yi9b':         ('Yi-6b', 'Mistral-7b'),
    'nemo':         ('Mistral-7b', 'Qwen1.5-7b'),
    'qwen-14b':     ('Qwen1.5-7b', 'Llama3-8b'),
}

with open('$OUTDIR/fresh_attacks.csv', 'w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=['defender','model_index','model','example_index','prompt','target','suffix','loss'])
    writer.writeheader()

    for defender, (self_name, anchor_name) in configs.items():
        # Exclude self and anchor model attacks
        pool = [a for a in attacks if a['model'] != self_name and a['model'] != anchor_name]
        selected = random.sample(pool, min(100, len(pool)))
        for row in selected:
            writer.writerow({
                'defender': defender,
                'model_index': row['model_index'],
                'model': row['model'],
                'example_index': row['example_index'],
                'prompt': row['prompt'],
                'target': row['target'],
                'suffix': row['suffix'],
                'loss': row['loss'],
            })
        print(f'{defender}: {len(selected)} fresh attacks from {len(set(a[\"model\"] for a in selected))} models (excluded {self_name}, {anchor_name})')

print('Saved to $OUTDIR/fresh_attacks.csv')
"

# ============================================================
# Step 2: Fresh eval script using held-out attacks
# ============================================================
python3 << 'PYEOF'
import json, csv, sys, os, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, '.')
from evaluate_v2 import (
    load_tokenizer, smart_format, clean_response,
    load_judge, setup_gibberish_detector
)
from llm_judge import classify_response as llm_judge_classify

OUTDIR = "./7b_defense_wildguard_outputs"
device = "cuda"

# Load fresh attacks
fresh = {}
with open(f"{OUTDIR}/fresh_attacks.csv") as f:
    for row in csv.DictReader(f):
        d = row['defender']
        if d not in fresh:
            fresh[d] = []
        fresh[d].append(row)

# Model configs: (defender_key, model_id, adapter_label, anchor_key)
# Best configs per model from ablation
CONFIGS = [
    # 7B models - best v2 ablation configs
    ("qwen",    "Qwen/Qwen1.5-7B-Chat",                  "qwen_ab2_ax",    "llama3"),
    ("qwen",    "Qwen/Qwen1.5-7B-Chat",                  "qwen_ab2_hx",    "llama3"),
    ("vicuna",  "lmsys/vicuna-7b-v1.5",                   "vicuna_ab2_ax",  "qwen"),
    ("vicuna",  "lmsys/vicuna-7b-v1.5",                   "vicuna_ab2_hx",  "qwen"),
    ("llama3",  "meta-llama/Meta-Llama-3-8B-Instruct",    "llama3_ab2_bx",  "qwen"),
    ("llama3",  "meta-llama/Meta-Llama-3-8B-Instruct",    "llama3_ab2_hx",  "qwen"),
    ("yi9b",    "01-ai/Yi-1.5-9B-Chat",                   "yi9b_ab2_an",    "mistral"),
    ("yi9b",    "01-ai/Yi-1.5-9B-Chat",                   "yi9b_ab2_ax",    "mistral"),
    ("nemo",    "mistralai/Mistral-Nemo-Instruct-2407",    "nemo_ab2_an",    "qwen"),
    ("nemo",    "mistralai/Mistral-Nemo-Instruct-2407",    "nemo_ab2_ax",    "qwen"),
    # 14B models
    ("qwen-14b","Qwen/Qwen1.5-14B-Chat",                  "qwen14b_q14_b",  "llama3"),
    ("qwen-14b","Qwen/Qwen1.5-14B-Chat",                  "qwen14b_q14_c",  "llama3"),
]

# Also run baselines
BASELINES = [
    ("qwen",    "Qwen/Qwen1.5-7B-Chat"),
    ("vicuna",  "lmsys/vicuna-7b-v1.5"),
    ("llama3",  "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("yi9b",    "01-ai/Yi-1.5-9B-Chat"),
    ("nemo",    "mistralai/Mistral-Nemo-Instruct-2407"),
    ("qwen-14b","Qwen/Qwen1.5-14B-Chat"),
]

# Load judge
print("[*] Loading WildGuard judge...")
judge_model_id = "allenai/wildguard"
judge_model, judge_tokenizer = load_judge(judge_model_id, quantize=False)
gib_detector = setup_gibberish_detector(device=device)

SLOW = {"llama2","llama2-13b","vicuna","orca","mistral","zephyr","starling","yi","phi2","yi9b"}

def eval_fresh(model, tokenizer, defender_key, model_type, attacks, label):
    """Evaluate on 100 fresh held-out attacks."""
    model.eval()
    success = 0
    total = len(attacks)

    for i, atk in enumerate(attacks):
        prompt_with_suffix = atk['prompt'] + " " + atk['suffix']
        formatted = smart_format(prompt_with_suffix, model_type, tokenizer)
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        gen_ids = out[0][enc.input_ids.shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Judge with WildGuard (full pipeline: coherence + compliance + judge)
        is_success, reason, metadata = llm_judge_classify(
            response, atk['prompt'], judge_model, judge_tokenizer, judge_model_id, gib_detector, suffix=atk['suffix']
        )
        if is_success:
            success += 1

    asr = success / total * 100
    print(f"  {label}: {success}/{total} = {asr:.1f}% ASR (fresh held-out)")
    return asr, total, success

results = {}

# Run baselines
for defender_key, model_id in BASELINES:
    out_file = f"{OUTDIR}/fresh_eval_{defender_key}_baseline.json"
    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} exists")
        d = json.load(open(out_file))
        results[f"{defender_key}_baseline"] = d
        continue

    print(f"\n{'='*60}")
    print(f"  BASELINE: {defender_key} ({model_id})")
    print(f"{'='*60}")

    model_type = defender_key.replace("-14b","").replace("-13b","")
    use_fast = model_type not in SLOW
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    attacks = fresh[defender_key]
    asr, total, success = eval_fresh(model, tokenizer, defender_key, model_type, attacks, f"{defender_key}_baseline")

    result = {"label": f"{defender_key}_baseline", "asr": asr, "total": total, "success": success}
    results[f"{defender_key}_baseline"] = result
    json.dump(result, open(out_file, 'w'), indent=2)

    del model
    torch.cuda.empty_cache()

# Run defended configs
for defender_key, model_id, adapter_label, anchor_key in CONFIGS:
    out_file = f"{OUTDIR}/fresh_eval_{adapter_label}.json"
    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} exists")
        d = json.load(open(out_file))
        results[adapter_label] = d
        continue

    # Find adapter path
    import subprocess
    adapter = subprocess.run(
        f"grep -oP 'defender_v2_cka_\\d+_\\d+' {OUTDIR}/train_{adapter_label}.log | head -1",
        shell=True, capture_output=True, text=True
    ).stdout.strip()

    if not adapter or not os.path.exists(f"{OUTDIR}/{adapter}"):
        print(f"[ERROR] No adapter for {adapter_label}")
        continue

    adapter_path = f"{OUTDIR}/{adapter}"

    print(f"\n{'='*60}")
    print(f"  DEFENDED: {adapter_label} ({adapter_path})")
    print(f"{'='*60}")

    model_type = defender_key.replace("-14b","").replace("-13b","")
    use_fast = model_type not in SLOW
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)

    attacks = fresh[defender_key]
    asr, total, success = eval_fresh(model, tokenizer, defender_key, model_type, attacks, adapter_label)

    # Get baseline for delta
    bl_asr = results.get(f"{defender_key}_baseline", {}).get("asr", 0)

    result = {"label": adapter_label, "asr": asr, "total": total, "success": success,
              "baseline_asr": bl_asr, "delta": asr - bl_asr}
    results[adapter_label] = result
    json.dump(result, open(out_file, 'w'), indent=2)

    del model
    torch.cuda.empty_cache()

# Print summary
print(f"\n{'='*60}")
print(f"  FRESH HELD-OUT EVAL SUMMARY")
print(f"{'='*60}")
print(f"{'Config':<24} {'ASR':>6} {'Baseline':>9} {'Delta':>7}")
print("-"*50)

for defender_key, _ in BASELINES:
    bl = results.get(f"{defender_key}_baseline", {})
    print(f"  {defender_key+' baseline':<24} {bl.get('asr',0):>5.1f}%")

print()
for _, _, adapter_label, _ in CONFIGS:
    r = results.get(adapter_label, {})
    if r:
        print(f"  {adapter_label:<24} {r.get('asr',0):>5.1f}% {r.get('baseline_asr',0):>8.1f}% {r.get('delta',0):>+6.1f}")

PYEOF

echo ""
echo "============================================================"
echo "  FRESH HELD-OUT EVAL COMPLETE"
echo "============================================================"
