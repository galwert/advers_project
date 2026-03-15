#!/bin/bash
#SBATCH --job-name=eval_defns
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_eval_defns_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_eval_defns_%j.err

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

# Import tokenizer helpers from evaluate_v2
from evaluate_v2 import load_tokenizer, sync_model_tokenizer

OUTDIR = "./7b_defense_wildguard_outputs/cb_eval"
GCG_CSV = "../outputs/advbench_suffixes_all_models_fixed.csv"

# All competing defense models to evaluate
DEFENSE_MODELS = {
    # Circuit Breakers
    "cb_mistral": {
        "model_id": "GraySwanAI/Mistral-7B-Instruct-RR",
        "base_model": "mistral",
        "self_index": 3,
        "needs_snapshot": False,
    },
    "cb_llama3": {
        "model_id": "GraySwanAI/Llama-3-8B-Instruct-RR",
        "base_model": "llama3",
        "self_index": 1,
        "needs_snapshot": False,
    },
    # RepBend
    "repbend_mistral": {
        "model_id": "AIM-Intelligence/RepBend_Mistral_7B",
        "base_model": "mistral",
        "self_index": 3,
        "needs_snapshot": True,
        "model_type_patch": "mistral",
    },
    "repbend_llama3": {
        "model_id": "AIM-Intelligence/RepBend_Llama3_8B",
        "base_model": "llama3",
        "self_index": 1,
        "needs_snapshot": True,
        "model_type_patch": "llama",
    },
    # LAT (Llama3 only — no Mistral variant available)
    "lat_llama3": {
        "model_id": "LLM-LAT/robust-llama3-8b-instruct",
        "base_model": "llama3",
        "self_index": 1,
        "needs_snapshot": False,
    },
}

BASE_MODEL_IDS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

INDEX_TO_MODEL = {
    0: "Llama2-7b", 1: "Llama3-8b", 2: "Vicuna-7b", 3: "Mistral-7b", 4: "Zephyr-7b",
    6: "Starling-7b", 9: "Phi-2", 10: "Qwen1.5-7b", 11: "Yi-6b", 15: "Orca-2-7b",
}

ALL_ATTACK_INDICES = [0, 1, 2, 3, 4, 6, 9, 10, 11, 15]


def generate(model, tokenizer, formatted, max_new=256):
    enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_defense(name, info):
    """Load a defense model, handling snapshot downloads and config patching."""
    model_id = info["model_id"]
    base_type = info["base_model"]
    base_id = BASE_MODEL_IDS[base_type]

    # Load tokenizer — try defense model first, fall back to base
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
        # Broken HF repo: download snapshot, patch config, load locally
        print(f"  [*] Downloading snapshot from {model_id}...")
        snapshot_dir = snapshot_download(model_id)

        # Find model files
        index_hits = glob_mod.glob(os.path.join(snapshot_dir, "**/model.safetensors.index.json"), recursive=True)
        if not index_hits:
            # Try single safetensors file
            st_hits = glob_mod.glob(os.path.join(snapshot_dir, "**/model.safetensors"), recursive=True)
            if st_hits:
                model_dir = os.path.dirname(st_hits[0])
            else:
                raise FileNotFoundError(f"No model files found under {snapshot_dir}")
        else:
            model_dir = os.path.dirname(index_hits[0])
        print(f"  [*] Model dir: {model_dir}")

        # Patch config.json if missing model_type
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
    out_file = f"{OUTDIR}/eval_{def_name}.json"
    if os.path.exists(out_file):
        print(f"\n[SKIP] {out_file} already exists")
        continue

    print(f"\n{'='*60}")
    print(f"  Evaluating: {def_name} ({def_info['model_id']})")
    print(f"{'='*60}")

    # Move judge to CPU, load target
    judge_model.cpu()
    torch.cuda.empty_cache()

    print(f"[*] Loading {def_info['model_id']}...")
    try:
        model, tokenizer = load_defense(def_name, def_info)
    except Exception as e:
        print(f"  [ERROR] Failed to load {def_name}: {e}")
        torch.cuda.empty_cache()
        judge_model.cuda()
        # Save error file so we don't retry
        with open(out_file.replace(".json", "_error.txt"), "w") as f:
            f.write(f"Failed to load: {e}\n")
        continue

    # Evaluate per-source ASR
    per_source = {}
    all_responses = []
    n_per_source = 100

    for src_idx in ALL_ATTACK_INDICES:
        src_name = INDEX_TO_MODEL.get(src_idx, f"idx{src_idx}")
        src_attacks = df[df["model_index"] == src_idx]
        if len(src_attacks) == 0:
            print(f"  [{src_name}] No attacks, skipping")
            continue

        # Sample up to n_per_source
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

    # Free target, bring judge back
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    judge_model.cuda()

    # Judge all responses
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

    # Compute per-source ASR
    results_by_source = {}
    for src_name, src_data in per_source.items():
        src_resps = [r for r in all_responses if r["source_model"] == src_name]
        success = sum(1 for r in src_resps if r.get("judge_success", False))
        total = len(src_resps)
        asr = (success / total * 100) if total > 0 else 0
        results_by_source[src_name] = {"asr": round(asr, 1), "success": success, "total": total}

        is_self = ""
        if src_data["responses"] and src_data["responses"][0]["source_index"] == def_info["self_index"]:
            is_self = " (SELF)"
        print(f"  {src_name}{is_self}: {asr:.1f}% ASR ({success}/{total})")

    # Self ASR
    self_idx = def_info["self_index"]
    self_name = INDEX_TO_MODEL.get(self_idx, f"idx{self_idx}")
    self_asr = results_by_source.get(self_name, {}).get("asr", None)

    # Cross ASR (all non-self sources)
    cross_results = {k: v for k, v in results_by_source.items() if k != self_name}
    cross_success = sum(v["success"] for v in cross_results.values())
    cross_total = sum(v["total"] for v in cross_results.values())
    cross_asr = (cross_success / cross_total * 100) if cross_total > 0 else 0

    overall_asr = (total_success / len(all_responses) * 100) if all_responses else 0

    summary = {
        "defense": def_name,
        "model_id": def_info["model_id"],
        "base_model": def_info["base_model"],
        "judge": judge_model_id,
        "self_asr": round(self_asr, 1) if self_asr is not None else None,
        "cross_asr": round(cross_asr, 1),
        "overall_asr": round(overall_asr, 1),
        "total_success": total_success,
        "total_responses": len(all_responses),
        "per_source_asr": results_by_source,
    }

    print(f"\n  SUMMARY for {def_name}:")
    print(f"    Self ASR:    {self_asr:.1f}%" if self_asr is not None else "    Self ASR:    N/A")
    print(f"    Cross ASR:   {cross_asr:.1f}%")
    print(f"    Overall ASR: {overall_asr:.1f}%")

    # Save full results
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_file}")

    # Save all responses separately
    resp_file = f"{OUTDIR}/responses_{def_name}.json"
    with open(resp_file, "w") as f:
        json.dump({"defense": def_name, "model_id": def_info["model_id"],
                    "responses": all_responses}, f, indent=2)
    print(f"  Responses saved to {resp_file}")

# Cleanup
del judge_model, judge_tokenizer
gc.collect()
torch.cuda.empty_cache()

# Final summary
print("\n" + "="*60)
print("  COMPETING DEFENSES EVALUATION COMPLETE")
print("="*60)

for def_name in DEFENSE_MODELS:
    f = f"{OUTDIR}/eval_{def_name}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"\n  {def_name} ({d['model_id']}):")
        print(f"    Self ASR:    {d['self_asr']:.1f}%" if d['self_asr'] is not None else f"    Self ASR:    N/A")
        print(f"    Cross ASR:   {d['cross_asr']:.1f}%")
        print(f"    Overall ASR: {d['overall_asr']:.1f}%")
        print(f"    Per-source:")
        for src, v in d["per_source_asr"].items():
            print(f"      {src}: {v['asr']:.1f}% ({v['success']}/{v['total']})")
    else:
        err = f.replace(".json", "_error.txt")
        if os.path.exists(err):
            print(f"\n  {def_name}: FAILED (see {err})")
        else:
            print(f"\n  {def_name}: NOT RUN")
PYEOF

echo ""
echo "[+] Done"
