"""
experiments/run_ablation.py

Runs all three Dirichlet ablations sequentially.

Usage:
    python -m experiments.run_ablation
    python -m experiments.run_ablation --rounds 10   # quick smoke test
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import torch
from main import load_config
from fl.runner import FLRunner


CIFAR100_CONFIGS = [
    "configs/dirichlet_01.yaml",
    "configs/dirichlet_05.yaml",
    "configs/dirichlet_10.yaml",
]

CIFAR10_CONFIGS = [
    "configs/cifar10_alpha01.yaml",
    "configs/cifar10_alpha05.yaml",
    "configs/cifar10_alpha10.yaml",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_all(config_files: list, rounds_override: int = None):
    os.makedirs("results", exist_ok=True)
    all_results = {}

    for config_path in config_files:
        cfg      = load_config(config_path)
        exp_name = cfg.get("experiment", {}).get(
            "name", os.path.basename(config_path).replace(".yaml", "")
        )

        if rounds_override is not None:
            cfg["fl"]["num_rounds"] = rounds_override

        print(f"\n{'='*60}")
        print(f"  Experiment : {exp_name}")
        print(f"  Dataset    : {cfg['data'].get('name', 'cifar100')}")
        print(f"  Alpha (D)  : {cfg['data']['dirichlet_alpha']}")
        print(f"  Rounds     : {cfg['fl']['num_rounds']}")
        print(f"{'='*60}\n")

        runner  = FLRunner(cfg, device=DEVICE)
        history = runner.run()

        out_path = f"results/{exp_name}_history.json"
        with open(out_path, "w") as f:
            json.dump(history, f, indent=2)

        accs      = [r["top1_acc"] for r in history]
        best_acc  = max(accs)
        final_acc = accs[-1]

        all_results[exp_name] = {
            "dataset"  : cfg["data"].get("name", "cifar100"),
            "alpha"    : cfg["data"]["dirichlet_alpha"],
            "best_acc" : best_acc,
            "final_acc": final_acc,
        }

        print(f"\n  {exp_name} done.")
        print(f"  Best:  {best_acc:.4f}")
        print(f"  Final: {final_acc:.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<25} {'Alpha':>6} {'Best':>8} {'Final':>8}")
    print(f"  {'-'*52}")
    for name, res in all_results.items():
        print(
            f"  {name:<25} "
            f"{res['alpha']:>6.1f} "
            f"{res['best_acc']:>8.4f} "
            f"{res['final_acc']:>8.4f}"
        )
    print(f"{'='*60}\n")

    summary_path = "results/ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="HSBN FL Ablation")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cifar10"],
                        help="Dataset to run ablation on")
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    configs = CIFAR100_CONFIGS if args.dataset == "cifar100" else CIFAR10_CONFIGS
    run_all(configs, rounds_override=args.rounds)


if __name__ == "__main__":
    main()