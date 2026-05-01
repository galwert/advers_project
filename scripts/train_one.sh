#!/usr/bin/env bash
# Train one AnchorRep defender from a YAML config.
#
# Usage:
#     bash scripts/train_one.sh <config_name> [extra train.py flags ...]
#
# <config_name> resolves to configs/<config_name>.yaml. The 5 paper-pick
# configs ship with the repo (llama3, mistral, vicuna, qwen14b, phi3).
# To train with custom hyperparameters, copy configs/custom.yaml.example to
# configs/custom.yaml, edit it, then run: bash scripts/train_one.sh custom
#
# Any extra arguments after the config name are forwarded verbatim to train.py,
# allowing one-off overrides without editing the YAML:
#     bash scripts/train_one.sh mistral --gamma 0.5 --stage2_steps 800

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name> [extra train.py flags ...]"
    echo "  Built-in configs: llama3 | mistral | vicuna | qwen14b | phi3 | custom"
    echo
    echo "  Examples:"
    echo "    bash scripts/train_one.sh mistral"
    echo "    bash scripts/train_one.sh custom"
    echo "    bash scripts/train_one.sh mistral --gamma 0.5 --stage2_steps 400"
    exit 1
fi

CONFIG_NAME="$1"
shift
CONFIG="configs/${CONFIG_NAME}.yaml"
OUTPUT_DIR="runs/${CONFIG_NAME}"

if [[ ! -f "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG"
    if [[ "$CONFIG_NAME" == "custom" ]]; then
        echo "  Copy configs/custom.yaml.example -> configs/custom.yaml and edit before running."
    fi
    echo "Available configs:"
    ls configs/ | grep '\.yaml$'
    exit 1
fi

echo "[train_one] config:     $CONFIG"
echo "[train_one] output dir: $OUTPUT_DIR"
if [[ $# -gt 0 ]]; then
    echo "[train_one] extra args: $*"
fi
echo

python -m anchor_rep.run_with_config \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo
echo "[train_one] done. Adapter saved under: $OUTPUT_DIR/"
