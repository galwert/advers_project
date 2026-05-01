#!/usr/bin/env bash
# Evaluate a defender against cross-model GCG transfer WITHOUT training.
# Defaults to the released HuggingFace adapter; pass a local path to evaluate
# a different adapter you have on disk.
#
# Usage:
#     bash scripts/eval_only.sh <defender> [adapter_path_or_repo]
#
# Examples:
#     bash scripts/eval_only.sh mistral
#         -> evaluates the released adapter:
#            anonsubmission12345/AnchorRep-Mistral-7B-Instruct-v0.2
#
#     bash scripts/eval_only.sh mistral runs/mistral/adapter
#         -> evaluates a locally-trained adapter
#
#     bash scripts/eval_only.sh mistral other-user/SomeOtherAdapter
#         -> evaluates any HF adapter compatible with the Mistral base
#
# This is a thin wrapper around scripts/eval_one.sh; the two are aliases.

set -euo pipefail

exec bash "$(dirname "$0")/eval_one.sh" "$@"
