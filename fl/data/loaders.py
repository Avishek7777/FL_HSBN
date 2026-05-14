"""
fl/data/loaders.py

Dataset-aware DataLoader builders.
Supports: CIFAR-10, CIFAR-100, FashionMNIST, TinyImageNet

drop_last=True on client loaders is intentional:
    Z1 adapter receives stacked (N, B, d1) tensors.
    All clients must submit the same batch size B.
    Dropping the last incomplete batch avoids shape mismatches.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple


# ── Per-dataset normalization stats ───────────────────────────────────────────

DATASET_STATS = {
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std" : (0.2675, 0.2565, 0.2761),
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std" : (0.2470, 0.2435, 0.2616),
    },
    "fmnist": {
        "mean": (0.2860,),
        "std" : (0.3530,),
    },
    "tinyimagenet": {
        "mean": (0.4802, 0.4481, 0.3975),
        "std" : (0.2302, 0.2265, 0.2262),
    },
}


# ── Transform builders ────────────────────────────────────────────────────────

def get_transforms(
    dataset_name: str,
    train       : bool,
    input_size  : int = 32,
) -> transforms.Compose:
    """
    Build dataset-appropriate transforms.
    Handles grayscale (FMNIST) and variable input sizes.
    """
    stats = DATASET_STATS[dataset_name]
    mean  = stats["mean"]
    std   = stats["std"]

    # Resize only if needed (TinyImageNet is 64x64 natively)
    resize = [transforms.Resize((input_size, input_size))] \
             if dataset_name == "tinyimagenet" else []

    if train:
        augment = [
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ] if dataset_name != "fmnist" else [
            transforms.RandomHorizontalFlip(),
        ]
        return transforms.Compose(
            resize + augment + [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            resize + [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


# ── Dataset builders ──────────────────────────────────────────────────────────

def load_dataset(
    dataset_name: str,
    data_root   : str,
    train       : bool,
    input_size  : int = 32,
):
    """
    Load a dataset by name. Returns a torch Dataset.
    """
    transform = get_transforms(dataset_name, train, input_size)

    if dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=data_root, train=train,
            download=True, transform=transform,
        )
    elif dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=data_root, train=train,
            download=True, transform=transform,
        )
    elif dataset_name == "fmnist":
        return datasets.FashionMNIST(
            root=data_root, train=train,
            download=True, transform=transform,
        )
    elif dataset_name == "tinyimagenet":
        split = "train" if train else "val"
        return datasets.ImageFolder(
            root=os.path.join(data_root, "tiny-imagenet-200", split),
            transform=transform,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: cifar100, cifar10, fmnist, tinyimagenet"
        )


# ── Client loaders ────────────────────────────────────────────────────────────

def build_client_loaders(
    partition   : Dict[int, np.ndarray],
    data_root   : str,
    batch_size  : int,
    dataset_name: str = "cifar100",
    input_size  : int = 32,
    num_workers : int = 2,
) -> Dict[int, DataLoader]:
    """
    Build a DataLoader for each client from their partition indices.
    drop_last=True ensures consistent batch size for Z1 stack alignment.
    """
    dataset = load_dataset(dataset_name, data_root, train=True,
                           input_size=input_size)
    loaders = {}
    for cid, indices in partition.items():
        subset = Subset(dataset, indices.tolist())
        loaders[cid] = DataLoader(
            subset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
    return loaders


# ── Public loader ─────────────────────────────────────────────────────────────

def build_public_loader(
    public_indices: np.ndarray,
    data_root     : str,
    batch_size    : int,
    dataset_name  : str = "cifar100",
    input_size    : int = 32,
    num_workers   : int = 2,
) -> DataLoader:
    """
    Build a DataLoader for the server's public IID split.
    """
    dataset = load_dataset(dataset_name, data_root, train=True,
                           input_size=input_size)
    subset  = Subset(dataset, public_indices.tolist())
    return DataLoader(
        subset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )


# ── Global test loader ────────────────────────────────────────────────────────

def build_global_test_loader(
    data_root   : str,
    batch_size  : int,
    dataset_name: str = "cifar100",
    input_size  : int = 32,
    num_workers : int = 2,
) -> DataLoader:
    """
    Build a DataLoader for global test evaluation.
    """
    dataset = load_dataset(dataset_name, data_root, train=False,
                           input_size=input_size)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )