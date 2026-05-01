#!/usr/bin/env bash
# Reproduce the main results table (tab:comparison) end-to-end.
#
# Runs cross-model transfer evaluation for all 5 defenders against the released
# adapters from the AnchorRep HuggingFace Collection. Total runtime: 4 to 6 hours
# on a single L40S (48 GB).
#
# Usage:
#     bash scripts/reproduce_main.sh

set -euo pipefail

DEFENDERS=(llama3 mistral vicuna qwen14b phi3)

mkdir -p logs/cross_model_transfer

echo "[reproduce_main] running cross-model transfer for ${#DEFENDERS[@]} defenders ..."
echo

for d in "${DEFENDERS[@]}"; do
    echo "================================================================"
    echo "  Evaluating defender: $d"
    echo "================================================================"
    bash scripts/eval_one.sh "$d"
    echo
done

echo
echo "[reproduce_main] all defenders evaluated."
echo "Per-defender summaries are in logs/cross_model_transfer/{defender}_repro/summary.csv"
echo
echo "To compare against the paper-original outputs, run:"
echo "    diff logs/cross_model_transfer/{defender}_repro/summary.csv logs/cross_model_transfer/{defender}_per_source_summary.json"
