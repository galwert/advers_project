"""Load a YAML config and invoke anchor_rep.train with the right CLI flags.

Usage:
    python -m anchor_rep.run_with_config --config configs/mistral.yaml \
        --output-dir runs/mistral

Any extra arguments after --output-dir are forwarded to train.py verbatim,
allowing config overrides without editing the YAML file:

    python -m anchor_rep.run_with_config --config configs/mistral.yaml \
        --output-dir runs/mistral_g05 --gamma 0.5
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def yaml_to_argv(cfg: dict) -> list[str]:
    """Translate a YAML config dict into a list of argv flags for train.py."""
    argv: list[str] = []

    defender = cfg.get("defender", {})
    if "name" in defender:
        argv += ["--defender", str(defender["name"])]
    if "precision" in defender:
        argv += ["--precision", str(defender["precision"])]

    anchor = cfg.get("anchor", {})
    if "name" in anchor:
        argv += ["--anchor", str(anchor["name"])]
    if "precision" in anchor:
        argv += ["--anchor_precision", str(anchor["precision"])]

    losses = cfg.get("losses", {})
    for k in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"):
        if k in losses:
            argv += [f"--{k}", str(losses[k])]

    training = cfg.get("training", {})
    if "cka_scope" in training:
        argv += ["--cka_scope", str(training["cka_scope"])]
    if "target_layer_pct" in training:
        argv += ["--target_layer_pct", str(training["target_layer_pct"])]
    if "stage2_steps" in training:
        argv += ["--stage2_steps", str(training["stage2_steps"])]
    if "stage2_lr" in training:
        argv += ["--stage2_lr", str(training["stage2_lr"])]
    if "seed" in training:
        argv += ["--seed", str(training["seed"])]

    lora = cfg.get("lora", {})
    if "rank" in lora:
        argv += ["--lora_r", str(lora["rank"])]

    borderline = cfg.get("borderline", {})
    if "source" in borderline:
        argv += ["--borderline_source", str(borderline["source"])]
    if "count" in borderline:
        argv += ["--n_borderline", str(borderline["count"])]
    argv += ["--use_borderline"]

    data = cfg.get("data", {})
    if "harmful_prompts_file" in data:
        argv += ["--harmful_prompts_file", str(data["harmful_prompts_file"])]
    if "gcg_data_path" in data:
        argv += ["--gcg_data_path", str(data["gcg_data_path"])]

    return argv


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=str,
                        help="Path to YAML config (e.g., configs/mistral.yaml).")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Where to write the trained adapter.")
    args, extra = parser.parse_known_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        sys.exit(f"[error] config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    argv = yaml_to_argv(cfg)
    argv += ["--output_dir", args.output_dir]
    argv += extra

    print(f"[run_with_config] resolved argv:")
    for i in range(0, len(argv), 2):
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            print(f"    {argv[i]} {argv[i+1]}")
        else:
            print(f"    {argv[i]}")
    print()

    sys.argv = ["anchor_rep.train"] + argv
    from anchor_rep.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
