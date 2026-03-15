#!/bin/bash
#SBATCH --job-name=div_comp
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=7b_defense_wildguard_outputs/slurm_div_comp_%j.out
#SBATCH --error=7b_defense_wildguard_outputs/slurm_div_comp_%j.err

source /home/wertheizer/miniforge3/etc/profile.d/conda.sh
conda activate defense_env
cd /home/wertheizer/advers_project/phase3

export HF_TOKEN="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"
export TQDM_DISABLE=1

OUTDIR="./7b_defense_wildguard_outputs"
N=50

# ── CB and LAT models: load directly from HF ──────────────────────────────
# These are standard HF models, diverse_attacks_13b.py handles them fine.
# Skip TAP (expensive + redundant with PAIR). Use: autodan, semantic_rewrite, embedding_pgd, pair
# Attacker: Mistral-7B (default). Target 7-8B + attacker 7B = ~28GB fp16, fits on L40.

declare -A MODELS
MODELS[cb_mistral]="GraySwanAI/Mistral-7B-Instruct-RR"
MODELS[cb_llama3]="GraySwanAI/Llama-3-8B-Instruct-RR"
MODELS[lat_llama3]="LLM-LAT/robust-llama3-8b-instruct"

for name in cb_mistral cb_llama3 lat_llama3; do
    mid="${MODELS[$name]}"
    out="$OUTDIR/diverse_${name}.json"
    if [ -f "$out" ]; then
        echo "[SKIP] $out exists"
        continue
    fi
    echo ""
    echo "============================================================"
    echo "  Diverse attacks: $name ($mid)"
    echo "============================================================"
    python diverse_attacks_13b.py \
        --model_id "$mid" \
        --precision fp16 --n_prompts $N \
        --attacks autodan semantic_rewrite embedding_pgd pair \
        --attacker_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
        --output "$out" \
        2>&1 | tee "$OUTDIR/diverse_${name}.log"
done

# ── RepBend models: need snapshot download + config patching ───────────────
# diverse_attacks_13b.py can't handle broken HF repos, so we download the
# snapshot first, patch config.json, then pass the local path as --model_id.

python3 << 'PYEOF'
import os, sys, json, glob as glob_mod, subprocess
from huggingface_hub import snapshot_download

OUTDIR = "./7b_defense_wildguard_outputs"
N = "50"

REPBEND_MODELS = {
    "repbend_mistral": {
        "model_id": "AIM-Intelligence/RepBend_Mistral_7B",
        "model_type_patch": "mistral",
    },
    "repbend_llama3": {
        "model_id": "AIM-Intelligence/RepBend_Llama3_8B",
        "model_type_patch": "llama",
    },
}

for name, info in REPBEND_MODELS.items():
    out_file = f"{OUTDIR}/diverse_{name}.json"
    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} exists")
        continue

    print(f"\n{'='*60}")
    print(f"  Diverse attacks: {name} ({info['model_id']})")
    print(f"{'='*60}")

    # Download and find model dir
    print(f"[*] Downloading snapshot: {info['model_id']}")
    snapshot_dir = snapshot_download(info["model_id"])

    index_hits = glob_mod.glob(os.path.join(snapshot_dir, "**/model.safetensors.index.json"), recursive=True)
    if not index_hits:
        st_hits = glob_mod.glob(os.path.join(snapshot_dir, "**/model.safetensors"), recursive=True)
        if st_hits:
            model_dir = os.path.dirname(st_hits[0])
        else:
            print(f"[ERROR] No model files found under {snapshot_dir}")
            continue
    else:
        model_dir = os.path.dirname(index_hits[0])

    # Patch config.json if needed
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if "model_type" not in cfg:
            cfg["model_type"] = info["model_type_patch"]
            print(f"[*] Patching config.json with model_type='{info['model_type_patch']}'")
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"[*] Using local model dir: {model_dir}")

    # Run diverse_attacks_13b.py with local path
    cmd = [
        sys.executable, "diverse_attacks_13b.py",
        "--model_id", model_dir,
        "--precision", "fp16",
        "--n_prompts", N,
        "--attacks", "autodan", "semantic_rewrite", "embedding_pgd", "pair",
        "--attacker_model_id", "mistralai/Mistral-7B-Instruct-v0.2",
        "--output", out_file,
    ]
    log_file = f"{OUTDIR}/diverse_{name}.log"
    print(f"[*] Running: {' '.join(cmd)}")

    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=36000)

    if result.returncode != 0:
        print(f"[ERROR] {name} failed (rc={result.returncode})")
    else:
        print(f"[+] {name} done, saved to {out_file}")

PYEOF

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DIVERSE ATTACKS ON COMPETING DEFENSES — SUMMARY"
echo "============================================================"
for name in cb_mistral cb_llama3 lat_llama3 repbend_mistral repbend_llama3; do
    f="$OUTDIR/diverse_${name}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
print(f'  $name:')
for k,v in d.items():
    if 'asr' in k.lower() and isinstance(v, (int,float)):
        print(f'    {k}: {v}')
" 2>/dev/null
    else
        echo "  $name: MISSING"
    fi
done

echo ""
echo "[+] Done"
