"""
fl/data/dirichlet.py

Dirichlet partitioning with public split.

Data split:
    CIFAR-100 train (50,000)
        ├── Public split  (5,000 — 10%, IID, balanced, server only)
        └── Private split (45,000 — partitioned across clients via Dirichlet)

The public split is carved out BEFORE Dirichlet partitioning so client
distributions are not contaminated by the balanced public samples.
50 samples per class across 100 classes — fully balanced IID.
"""

import numpy as np
from typing import Dict, Tuple


def carve_public_split(
    targets         : np.ndarray,
    samples_per_class: int = 50,
    seed            : int  = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carve out a balanced public split from the full dataset.

    Args:
        targets          : full array of class labels
        samples_per_class: how many samples per class in public split
                           50 × 100 classes = 5,000 total
        seed             : for reproducibility

    Returns:
        public_indices  : (5000,) balanced IID indices for server
        private_indices : (45000,) remaining indices for client partitioning
    """
    rng = np.random.default_rng(seed)
    num_classes   = int(targets.max()) + 1
    public_idx    = []
    private_idx   = []

    for c in range(num_classes):
        c_indices = np.where(targets == c)[0]
        rng.shuffle(c_indices)
        public_idx.extend(c_indices[:samples_per_class].tolist())
        private_idx.extend(c_indices[samples_per_class:].tolist())

    return np.array(public_idx), np.array(private_idx)


def dirichlet_partition(
    targets     : np.ndarray,
    num_clients : int,
    alpha       : float,
    seed        : int = 42,
) -> Dict[int, np.ndarray]:
    """
    Partition dataset indices across clients using Dirichlet distribution.
    Should be called on private_indices only, after carving public split.

    Args:
        targets     : array of class labels (private split only)
        num_clients : number of FL clients
        alpha       : Dirichlet concentration parameter
                      0.1 → severe heterogeneity
                      0.5 → moderate
                      1.0 → mild (near-IID)
        seed        : for reproducibility

    Returns:
        dict mapping client_id → array of dataset indices
    """
    rng = np.random.default_rng(seed)
    num_classes   = int(targets.max()) + 1
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    client_indices: Dict[int, list] = {i: [] for i in range(num_clients)}

    for c_indices in class_indices:
        rng.shuffle(c_indices)
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        splits      = (proportions * len(c_indices)).astype(int)
        splits[-1]  = len(c_indices) - splits[:-1].sum()

        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(
                c_indices[start: start + count].tolist()
            )
            start += count

    return {cid: np.array(idxs) for cid, idxs in client_indices.items()}


def partition_stats(
    partition  : Dict[int, np.ndarray],
    targets    : np.ndarray,
    num_classes: int,
) -> Dict[int, np.ndarray]:
    """
    Returns per-client class distribution as counts.
    Useful for verifying heterogeneity level.
    """
    stats = {}
    for cid, idxs in partition.items():
        counts     = np.bincount(targets[idxs], minlength=num_classes)
        stats[cid] = counts
    return stats