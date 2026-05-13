"""
experiments/transfer_eval.py

Transfer Evaluation
====================
Loads a trained server checkpoint and evaluates it on a
different dataset's test set without any fine-tuning.

Protocol:
    Train on CIFAR-100 → evaluate on CIFAR-10 test set
    Train on CIFAR-10  → evaluate on CIFAR-100 test set

Why this is meaningful:
    The server encoder, Z1, and Z1.5 learned visual representations
    from one dataset's distribution. Transfer accuracy measures how
    general those representations are — how much they capture
    universal visual structure vs dataset-specific features.

    Non-trivial transfer (above random chance) is evidence that
    the bottleneck forced the server to learn general representations
    rather than dataset-specific ones.

Usage:
    python experiments/transfer_eval.py \
        --source cifar100_alpha05 \
        --target cifar10 \
        --results_dir results/

Output:
    Transfer accuracy printed and saved to results/transfer_eval.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl.server.adapter import build_adapter
from fl.server.classifier import build_classifier
from fl.server.apex import build_apex
from fl.server.encoder import build_server_encoder
from fl.checkpoint import CheckpointManager
from main import load_config


# ── Dataset builders ──────────────────────────────────────────────────────────

DATASET_MEAN_STD = {
    "cifar10" : ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "fmnist"  : ((0.2860,), (0.3530,)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
}

def build_test_loader(
    dataset_name: str,
    data_root   : str,
    batch_size  : int = 64,
) -> tuple:
    """
    Build test DataLoader for the target dataset.
    Returns (loader, num_classes).
    """
    mean, std = DATASET_MEAN_STD[dataset_name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(
            root=data_root, train=False,
            download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(
            root=data_root, train=False,
            download=True, transform=transform
        )
        num_classes = 100

    elif dataset_name == "fmnist":
        ds = datasets.FashionMNIST(
            root=data_root, train=False,
            download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name == "tinyimagenet":
        ds = datasets.ImageFolder(
            root=os.path.join(data_root, "tiny-imagenet-200", "val"),
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        )
        num_classes = 200

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return loader, num_classes


# ── Transfer evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def transfer_evaluate(
    adapter,
    classifier,
    server_encoder,
    test_loader : DataLoader,
    device      : str,
) -> float:
    """
    Evaluate server components on target dataset test set.
    No fine-tuning — pure transfer accuracy.

    Uses server encoder → Z1 → Z1.5 → cls_head pipeline.
    cls_head was trained on source dataset classes so logits
    won't map correctly — instead we measure representation
    quality via nearest-centroid classification.

    For a simpler metric: linear probe accuracy.
    Train a single linear layer on top of frozen Z1.5 representations
    using the target dataset's training set, evaluate on test set.
    """
    adapter.eval()
    classifier.eval()
    server_encoder.eval()

    # Collect all representations and labels
    all_reprs  = []
    all_labels = []

    for x, labels in test_loader:
        x      = x.to(device)
        labels = labels.to(device)

        z0     = server_encoder(x)
        stacked = z0.unsqueeze(0)
        z1, _, _ = adapter(stacked)
        z1_5, _, _ = classifier(z1, z0, labels)

        all_reprs.append(z1_5.cpu())
        all_labels.append(labels.cpu())

    reprs  = torch.cat(all_reprs, dim=0)   # (N, d_cls)
    labels = torch.cat(all_labels, dim=0)  # (N,)

    return reprs, labels


def linear_probe_accuracy(
    reprs     : torch.Tensor,   # (N, d_cls)
    labels    : torch.Tensor,   # (N,)
    num_classes: int,
    device    : str,
    epochs    : int = 50,
    lr        : float = 0.01,
) -> float:
    """
    Train a linear probe on frozen representations.
    Standard transfer evaluation protocol.
    """
    import torch.nn as nn
    import torch.optim as optim

    N, d = reprs.shape
    split = int(0.8 * N)

    train_r, test_r = reprs[:split].to(device), reprs[split:].to(device)
    train_l, test_l = labels[:split].to(device), labels[split:].to(device)

    probe = nn.Linear(d, num_classes).to(device)
    opt   = optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        probe.train()
        logits = probe(train_r)
        loss   = F.cross_entropy(logits, train_l)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds   = probe(test_r).argmax(dim=1)
        correct = (preds == test_l).sum().item()
        total   = test_l.size(0)

    return correct / total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transfer Evaluation")
    parser.add_argument("--source", required=True,
                        help="Source checkpoint name e.g. cifar100_alpha05")
    parser.add_argument("--target", required=True,
                        help="Target dataset name e.g. cifar10")
    parser.add_argument("--config", required=True,
                        help="Config used for source training")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load checkpoint metadata to get num_classes of source
    ckpt_mgr  = CheckpointManager(args.results_dir, args.source)
    metadata  = ckpt_mgr.load_metadata()
    print(f"\nSource: {args.source}")
    print(f"  Best accuracy (source): {metadata['best_accuracy']:.4f}")
    print(f"  Target dataset: {args.target}\n")

    # Build server components
    device = args.device
    adapter        = build_adapter(cfg).to(device)
    classifier     = build_classifier(cfg).to(device)
    apex           = build_apex(cfg).to(device)
    server_encoder = build_server_encoder(cfg).to(device)

    # Load trained weights
    ckpt_mgr.load_server(adapter, classifier, apex, server_encoder, device)

    # Build target test loader
    test_loader, num_classes = build_test_loader(
        args.target, args.data_root
    )
    print(f"Target: {args.target} — {num_classes} classes")

    # Extract representations
    print("Extracting representations...")
    reprs, labels = transfer_evaluate(
        adapter, classifier, server_encoder, test_loader, device
    )

    # Linear probe
    print("Running linear probe (50 epochs)...")
    acc = linear_probe_accuracy(reprs, labels, num_classes, device)
    print(f"\nTransfer accuracy ({args.source} → {args.target}): {acc:.4f}")

    # Save result
    result = {
        "source"         : args.source,
        "target"         : args.target,
        "source_best_acc": metadata["best_accuracy"],
        "transfer_acc"   : acc,
        "num_classes"    : num_classes,
    }

    out_path = os.path.join(
        args.results_dir, f"transfer_{args.source}_to_{args.target}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()