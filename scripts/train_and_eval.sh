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

# Step 2: evaluate the freshly-saved adapter (eval_one.sh resolves --latest
# to the newest runs/<config>/defender_v2_* directory by mtime).
echo
echo "[train_and_eval] training done. Evaluating freshly-trained adapter (--latest)..."
echo
bash "$(dirname "$0")/eval_one.sh" "$CONFIG_NAME" --latest
