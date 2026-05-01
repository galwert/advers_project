#!/usr/bin/env bash
# Train a defender, then immediately evaluate the freshly-trained adapter.
#
# Usage:
#     bash scripts/train_and_eval.sh <config_name> [extra train.py flags ...]
#
# <config_name> resolves to configs/<config_name>.yaml. The 5 paper-pick
# configs ship with the repo (llama3, mistral, vicuna, qwen14b, phi3).
# Use `custom` after copying configs/custom.yaml.example to configs/custom.yaml.
#
# Examples:
#     bash scripts/train_and_eval.sh mistral
#     bash scripts/train_and_eval.sh custom
#     bash scripts/train_and_eval.sh mistral --gamma 0.5

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name> [extra train.py flags ...]"
    echo "  Built-in configs: llama3 | mistral | vicuna | qwen14b | phi3 | custom"
    exit 1
fi

CONFIG_NAME="$1"
shift

# Step 1: train
bash "$(dirname "$0")/train_one.sh" "$CONFIG_NAME" "$@"

# Step 2: locate the freshly-saved adapter and evaluate it
RUNS_DIR="runs/${CONFIG_NAME}"
LATEST_ADAPTER=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 2 -type d -name "defender_v2_cka_*" -printf '%T@ %p\n' 2>/dev/null \
                 | sort -nr | head -n1 | cut -d' ' -f2-)

if [[ -z "$LATEST_ADAPTER" ]] || [[ ! -f "$LATEST_ADAPTER/adapter_model.safetensors" ]]; then
    # Fallback: maybe the script saved directly to runs/<config_name>/adapter
    if [[ -f "$RUNS_DIR/adapter/adapter_model.safetensors" ]]; then
        LATEST_ADAPTER="$RUNS_DIR/adapter"
    else
        echo "[error] no adapter found under $RUNS_DIR/. Did training finish?"
        exit 1
    fi
fi

echo
echo "[train_and_eval] training done. Now evaluating: $LATEST_ADAPTER"
echo

bash "$(dirname "$0")/eval_one.sh" "$CONFIG_NAME" "$LATEST_ADAPTER"
