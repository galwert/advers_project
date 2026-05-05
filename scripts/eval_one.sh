#!/usr/bin/env bash
# Evaluate one AnchorRep defender on cross-model GCG transfer.
#
# Usage:
#     bash scripts/eval_one.sh <defender> [adapter|--latest]
#
# <defender>: llama3 | mistral | vicuna | qwen14b | phi3
#             (must have a matching configs/<defender>.yaml)
#
# Adapter resolution (highest precedence first):
#   1. Explicit 2nd arg = HuggingFace repo id OR local path.
#        bash scripts/eval_one.sh mistral my-org/my-fork-adapter
#        bash scripts/eval_one.sh mistral runs/mistral/defender_v2_cka_20260505_120000
#   2. --latest = newest local adapter under runs/<defender>/defender_v2_*.
#        bash scripts/eval_one.sh mistral --latest
#   3. Default = configs/<defender>.yaml field eval.hf_adapter
#        (the released anonymous AnchorRep adapter).
#
# Base model and defender label (target_model column) are always read from
# configs/<defender>.yaml so the YAML is the single source of truth.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <defender> [adapter|--latest]"
    echo "  defender: llama3 | mistral | vicuna | qwen14b | phi3"
    echo "  adapter:  HF repo id OR local path OR --latest (optional)"
    exit 1
fi

DEFENDER="$1"
CONFIG="configs/${DEFENDER}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG"
    echo "Available configs:"
    ls configs/ 2>/dev/null | grep '\.yaml$' || true
    exit 1
fi

# Read base / hf_adapter / defender_label from YAML using PyYAML.
# Helper prints "<base>|<hf_adapter>|<defender_label>" so we get all three in one parse.
read DEFAULT_BASE DEFAULT_ADAPTER DEFENDER_LABEL <<<"$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
base = cfg["defender"]["base_model"]
ev = cfg.get("eval") or {}
adapter = ev.get("hf_adapter", "")
label = ev.get("defender_label", cfg["defender"]["name"])
print(base, adapter, label)
PY
)"

if [[ -z "$DEFAULT_BASE" ]]; then
    echo "[error] $CONFIG missing defender.base_model"
    exit 1
fi

# Resolve adapter precedence
ADAPTER_ARG="${2:-}"
if [[ "$ADAPTER_ARG" == "--latest" ]]; then
    LATEST="$(ls -1dt runs/"${DEFENDER}"/defender_v2_* 2>/dev/null | head -1 || true)"
    if [[ -z "$LATEST" ]]; then
        echo "[error] --latest requested but no adapters found under runs/${DEFENDER}/defender_v2_*"
        echo "  Train one first: bash scripts/train_one.sh ${DEFENDER}"
        exit 1
    fi
    ADAPTER="$LATEST"
elif [[ -n "$ADAPTER_ARG" ]]; then
    ADAPTER="$ADAPTER_ARG"
else
    if [[ -z "$DEFAULT_ADAPTER" ]]; then
        echo "[error] $CONFIG has no eval.hf_adapter and no adapter override given"
        echo "  Pass an adapter explicitly or use --latest after training."
        exit 1
    fi
    ADAPTER="$DEFAULT_ADAPTER"
fi

OUTPUT_DIR="logs/cross_model_transfer/${DEFENDER}_repro"

echo "[eval_one] config:  $CONFIG"
echo "[eval_one] base:    $DEFAULT_BASE"
echo "[eval_one] adapter: $ADAPTER"
echo "[eval_one] label:   $DEFENDER_LABEL"
echo "[eval_one] output:  $OUTPUT_DIR"
echo

python eval/cross_model_transfer.py \
    --base-model "$DEFAULT_BASE" \
    --adapter "$ADAPTER" \
    --suffixes-csv attack_artifacts/advbench_suffixes_all_models.csv \
    --output-dir "$OUTPUT_DIR" \
    --defender-label "$DEFENDER_LABEL"

# Score the defended responses through the canonical WildGuard pipeline
# (Stages 0-5 from Appendix app:judge). This is what tab:comparison reports;
# the simple refusal-keyword stat printed by cross_model_transfer.py is only
# a quick sanity check.
echo
echo "[eval_one] judging defended responses through WildGuard pipeline..."
python eval/judge_pipeline.py \
    --input  "$OUTPUT_DIR/defended_model_results.csv" \
    --output "$OUTPUT_DIR/defended_judged.csv" \
    --judge  allenai/wildguard

echo
echo "[eval_one] aggregating Self / Anchor / Other / Transfer ASR..."
python eval/aggregate_asr.py \
    --input    "$OUTPUT_DIR/defended_judged.csv" \
    --defender "$DEFENDER" \
    --summary  "$OUTPUT_DIR/defended_judged_summary.json"

echo
echo "[eval_one] done. Results in: $OUTPUT_DIR"
echo "  - base_model_results.csv       : raw baseline responses"
echo "  - defended_model_results.csv   : raw defended responses"
echo "  - defended_judged.csv          : per-prompt WildGuard pipeline verdict"
echo "  - defended_judged_summary.json : Self / Anchor / Other / Transfer ASR (paper-comparable)"
