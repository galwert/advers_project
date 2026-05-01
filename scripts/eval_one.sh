#!/usr/bin/env bash
# Evaluate one AnchorRep defender on cross-model GCG transfer.
#
# Usage:
#     bash scripts/eval_one.sh <defender> [adapter]
#
# Where <defender> is one of: llama3, mistral, vicuna, qwen14b, phi3.
# [adapter] (optional) is a HuggingFace repo id or local adapter path.
#           If omitted, the released adapter from the AnchorRep collection is used.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <defender> [adapter]"
    echo "  defender: llama3 | mistral | vicuna | qwen14b | phi3"
    echo "  adapter:  HF repo id or local path (optional)"
    exit 1
fi

DEFENDER="$1"
DEFAULT_ADAPTER=""
DEFAULT_BASE=""

case "$DEFENDER" in
    llama3)
        DEFAULT_ADAPTER="anonsubmission12345/AnchorRep-Llama-3-8B-Instruct"
        DEFAULT_BASE="meta-llama/Meta-Llama-3-8B-Instruct"
        ;;
    mistral)
        DEFAULT_ADAPTER="anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2"
        DEFAULT_BASE="mistralai/Mistral-7B-Instruct-v0.2"
        ;;
    vicuna)
        DEFAULT_ADAPTER="anonsubmission12345/AnchorRep-Vicuna-7B-v1.5"
        DEFAULT_BASE="lmsys/vicuna-7b-v1.5"
        ;;
    qwen14b)
        DEFAULT_ADAPTER="anonsubmission12345/AnchorRep-Qwen1.5-14B-Chat"
        DEFAULT_BASE="Qwen/Qwen1.5-14B-Chat"
        ;;
    phi3)
        DEFAULT_ADAPTER="anonsubmission12345/AnchorRep-Phi-3-medium-4k-instruct"
        DEFAULT_BASE="microsoft/Phi-3-medium-4k-instruct"
        ;;
    *)
        echo "[error] unknown defender: $DEFENDER"
        echo "  Choices: llama3 | mistral | vicuna | qwen14b | phi3"
        exit 1
        ;;
esac

ADAPTER="${2:-$DEFAULT_ADAPTER}"
OUTPUT_DIR="logs/cross_model_transfer/${DEFENDER}_repro"

echo "[eval_one] base:    $DEFAULT_BASE"
echo "[eval_one] adapter: $ADAPTER"
echo "[eval_one] output:  $OUTPUT_DIR"
echo

python eval/cross_model_transfer.py \
    --base-model "$DEFAULT_BASE" \
    --adapter "$ADAPTER" \
    --suffixes-csv attack_artifacts/advbench_suffixes_all_models.csv \
    --output-dir "$OUTPUT_DIR"

echo
echo "[eval_one] done. Results in: $OUTPUT_DIR"
