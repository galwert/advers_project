#!/usr/bin/env bash
# Train one AnchorRep defender from a paper-pick YAML config.
#
# Usage:
#     bash scripts/train_one.sh <defender>
#
# Where <defender> is one of: llama3, mistral, vicuna, qwen14b, phi3.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <defender>"
    echo "  defender: llama3 | mistral | vicuna | qwen14b | phi3"
    exit 1
fi

DEFENDER="$1"
CONFIG="configs/${DEFENDER}.yaml"
OUTPUT_DIR="runs/${DEFENDER}"

if [[ ! -f "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG"
    echo "Available configs:"
    ls configs/
    exit 1
fi

echo "[train_one] config:     $CONFIG"
echo "[train_one] output dir: $OUTPUT_DIR"
echo

python -m anchor_rep.run_with_config \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR"

echo
echo "[train_one] done. Adapter saved to: $OUTPUT_DIR/adapter/"
