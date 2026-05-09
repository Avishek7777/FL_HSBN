"""
main.py

Entry point for HSBN Feedback-Driven FL.

Usage:
    # Single experiment
    python main.py --config configs/dirichlet_05.yaml

    # Full ablation (all three alpha values)
    python -m experiments.run_ablation

Config merging:
    Experiment configs are minimal — they only override what changes.
    load_config() deep-merges the experiment config on top of base.yaml
    so base.yaml remains the single source of truth for all hyperparameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import torch
from fl.runner import FLRunner


def load_config(experiment_path: str) -> dict:
    """
    Load base.yaml then deep-merge experiment config on top.
    Only keys present in the experiment config are overridden.

    Args:
        experiment_path: path to experiment YAML (e.g. configs/dirichlet_05.yaml)

    Returns:
        merged config dict
    """
    base_path = os.path.join(
        os.path.dirname(experiment_path), "base.yaml"
    )

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    with open(experiment_path) as f:
        override = yaml.safe_load(f)

    if override:
        for section, values in override.items():
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(values)
            else:
                cfg[section] = values

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="HSBN Feedback-Driven Federated Learning"
    )
    parser.add_argument(
        "--config",
        type    = str,
        required= True,
        help    = "Path to experiment config YAML (e.g. configs/dirichlet_05.yaml)"
    )
    parser.add_argument(
        "--device",
        type    = str,
        default = "cuda" if torch.cuda.is_available() else "cpu",
        help    = "Compute device (default: cuda if available)"
    )
    parser.add_argument(
        "--rounds",
        type    = int,
        default = None,
        help    = "Override num_rounds from config (useful for quick tests)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Optional round override for quick testing
    if args.rounds is not None:
        cfg["fl"]["num_rounds"] = args.rounds

    exp_name = cfg.get("experiment", {}).get("name", "unnamed")
    print(f"\n{'='*60}")
    print(f"  HSBN Feedback-Driven FL")
    print(f"  Experiment : {exp_name}")
    print(f"  Device     : {args.device}")
    print(f"  Rounds     : {cfg['fl']['num_rounds']}")
    print(f"  Clients    : {cfg['fl']['num_clients']} "
          f"({int(cfg['fl']['client_fraction']*100)}% per round)")
    print(f"  Alpha (D)  : {cfg['data']['dirichlet_alpha']}")
    print(f"  d1         : {cfg['common']['d1']}")
    print(f"{'='*60}\n")

    runner = FLRunner(cfg, device=args.device)

    import json
    os.makedirs("results", exist_ok=True)
    history = runner.run()

    out_path = f"results/{exp_name}_history.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()