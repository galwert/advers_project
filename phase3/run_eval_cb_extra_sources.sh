#!/bin/bash
#SBATCH --job-name=cb_extra
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_cb_extra_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_cb_extra_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs"
mkdir -p "$OUTDIR/cb_eval"

python3 << 'PYEOF'
import os, sys, json, torch, gc, random, glob as glob_mod
import pandas as pd
sys.modules["flash_attn"] = None
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "phase2"))
sys.path.insert(0, "../phase2")

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from llm_judge import load_judge, classify_response as llm_judge_classify, setup_gibberish_detector
from evaluate_v2 import load_tokenizer, sync_model_tokenizer

OUTDIR = "./7b_defense_wildguard_outputs/cb_eval"
GCG_CSV = "../outputs/advbench_suffixes_all_models_fixed.csv"

DEFENSE_MODELS = {
    "cb_mistral": {
        "model_id": "GraySwanAI/Mistral-7B-Instruct-RR",
        "base_model": "mistral",
        "needs_snapshot": False,
    },
    "cb_llama3": {
        "model_id": "GraySwanAI/Llama-3-8B-Instruct-RR",
        "base_model": "llama3",
        "needs_snapshot": False,
    },
    "repbend_mistral": {
        "model_id": "AIM-Intelligence/RepBend_Mistral_7B",
        "base_model": "mistral",
        "needs_snapshot": True,
        "model_type_patch": "mistral",
    },
    "repbend_llama3": {
        "model_id": "AIM-Intelligence/RepBend_Llama3_8B",
        "base_model": "llama3",
        "needs_snapshot": True,
        "model_type_patch": "llama",
    },
    "lat_llama3": {
        "model_id": "LLM-LAT/robust-llama3-8b-instruct",
        "base_model": "llama3",
        "needs_snapshot": False,
    },
}

BASE_MODEL_IDS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# UNTESTED source models
EXTRA_INDICES = {
    5: "Hermes-2",
    7: "OpenChat-3.5",
    8: "Gemma-7b",
    12: "Baichuan2-7b",
    13: "DeepSeek-7b",
    14: "InternLM2-7b",
    16: "Solar-10.7b",
    17: "Orca-2-7b",
    18: "NeuralChat-7b",
    19: "StableZephyr-3b",
}


def generate(model, tokenizer, formatted, max_new=256):
    enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_defense(name, info):
    model_id = info["model_id"]
    base_type = info["base_model"]
    base_id = BASE_MODEL_IDS[base_type]

    try:
        tokenizer = load_tokenizer(model_id, base_type)
    except Exception as e:
        print(f"  [!] Tokenizer from {model_id} failed ({e.__class__.__name__}), using {base_id}")
        tokenizer = load_tokenizer(base_id, base_type)

    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if info.get("needs_snapshot", False):
        print(f"  [*] Downloading snapshot from {model_id}...")
        snapshot_dir = snapshot_download(model_id)
        index_hits = glob_mod.glob(os.path.join(snapshot_dir, "**/model.safetensors.index.json"), recursive=True)
        if not index_hits:
            st_hits = glob_mod.glob(os.path.join(snapshot_dir, "**/model.safetensors"), recursive=True)
            if st_hits:
                model_dir = os.path.dirname(st_hits[0])
            else:
                raise FileNotFoundError(f"No model files found under {snapshot_dir}")
        else:
            model_dir = os.path.dirname(index_hits[0])
        print(f"  [*] Model dir: {model_dir}")

        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if "model_type" not in cfg:
                arch_type = info.get("model_type_patch", "llama")
                print(f"  [*] Patching config.json with model_type='{arch_type}'")
                cfg["model_type"] = arch_type
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    sync_model_tokenizer(model, tokenizer)
    model.eval()
    return model, tokenizer


# Load GCG data
print("[*] Loading GCG data...")
df = pd.read_csv(GCG_CSV)
print(f"[+] Loaded {len(df)} attack samples")

# Load judge
print("[*] Loading WildGuard judge...")
judge_model_id = "allenai/wildguard"
judge_model, judge_tokenizer = load_judge(judge_model_id, quantize=False)
gib_detector = setup_gibberish_detector()

for def_name, def_info in DEFENSE_MODELS.items():
    out_file = f"{OUTDIR}/eval_{def_name}_extra.json"
    if os.path.exists(out_file):
        print(f"\n[SKIP] {out_file} already exists")
        continue

    print(f"\n{'='*60}")
    print(f"  Evaluating: {def_name} ({def_info['model_id']}) — EXTRA SOURCES")
    print(f"{'='*60}")

    judge_model.cpu()
    torch.cuda.empty_cache()

    print(f"[*] Loading {def_info['model_id']}...")
    try:
        model, tokenizer = load_defense(def_name, def_info)
    except Exception as e:
        print(f"  [ERROR] Failed to load {def_name}: {e}")
        torch.cuda.empty_cache()
        judge_model.cuda()
        continue

    per_source = {}
    all_responses = []
    n_per_source = 100

    for src_idx, src_name in sorted(EXTRA_INDICES.items()):
        src_attacks = df[df["model_index"] == src_idx]
        if len(src_attacks) == 0:
            print(f"  [{src_name}] No attacks, skipping")
            continue

        if len(src_attacks) > n_per_source:
            src_attacks = src_attacks.sample(n=n_per_source, random_state=42)

        print(f"\n  [{src_name}] Generating {len(src_attacks)} responses...")
        src_responses = []
        for i, (_, row) in enumerate(src_attacks.iterrows()):
            full_prompt = f"{row['prompt']} {row['suffix']}"
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            resp = generate(model, tokenizer, formatted)
            src_responses.append({
                "prompt": row["prompt"],
                "suffix": row["suffix"],
                "source_model": src_name,
                "source_index": int(src_idx),
                "response": resp,
            })
            if (i+1) % 25 == 0:
                print(f"    Generated {i+1}/{len(src_attacks)}")

        per_source[src_name] = {"responses": src_responses, "total": len(src_responses)}
        all_responses.extend(src_responses)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    judge_model.cuda()

    print(f"\n  Judging {len(all_responses)} total responses...")
    total_success = 0
    for i, r in enumerate(all_responses):
        is_success, reason, meta = llm_judge_classify(
            r["response"], r["prompt"], judge_model, judge_tokenizer,
            judge_model_id, gib_detector, suffix=r["suffix"]
        )
        r["judge_success"] = is_success
        r["judge_reason"] = reason
        if is_success:
            total_success += 1
        if (i+1) % 100 == 0:
            print(f"    Judged {i+1}/{len(all_responses)}")

    results_by_source = {}
    for src_name, src_data in per_source.items():
        src_resps = [r for r in all_responses if r["source_model"] == src_name]
        success = sum(1 for r in src_resps if r.get("judge_success", False))
        total = len(src_resps)
        asr = (success / total * 100) if total > 0 else 0
        results_by_source[src_name] = {"asr": round(asr, 1), "success": success, "total": total}
        print(f"  {src_name}: {asr:.1f}% ASR ({success}/{total})")

    overall_asr = (total_success / len(all_responses) * 100) if all_responses else 0

    summary = {
        "defense": def_name,
        "model_id": def_info["model_id"],
        "judge": judge_model_id,
        "extra_sources": True,
        "overall_asr": round(overall_asr, 1),
        "total_success": total_success,
        "total_responses": len(all_responses),
        "per_source_asr": results_by_source,
    }

    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_file}")

    resp_file = f"{OUTDIR}/responses_{def_name}_extra.json"
    with open(resp_file, "w") as f:
        json.dump({"defense": def_name, "responses": all_responses}, f, indent=2)
    print(f"  Responses saved to {resp_file}")

del judge_model, judge_tokenizer
gc.collect()
torch.cuda.empty_cache()

print("\n" + "="*60)
print("  EXTRA SOURCES EVALUATION COMPLETE")
print("="*60)
for def_name in DEFENSE_MODELS:
    f = f"{OUTDIR}/eval_{def_name}_extra.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"\n  {def_name}: overall={d['overall_asr']}% ({d['total_success']}/{d['total_responses']})")
        for src, v in d["per_source_asr"].items():
            print(f"    {src}: {v['asr']}% ({v['success']}/{v['total']})")
PYEOF

echo ""
echo "[+] Done"
