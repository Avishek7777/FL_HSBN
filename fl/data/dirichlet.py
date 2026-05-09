# fl/data/dirichlet.py

import numpy as np
from typing import List, Dict


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Partition dataset indices across clients using a Dirichlet distribution.

    Args:
        targets      : array of class labels for the full dataset
        num_clients  : number of FL clients
        alpha        : Dirichlet concentration parameter
                       0.1 → severe heterogeneity
                       0.5 → moderate
                       1.0 → mild (near-IID)
        seed         : for reproducibility

    Returns:
        dict mapping client_id → array of dataset indices
    """
    rng = np.random.default_rng(seed)
    num_classes = int(targets.max()) + 1
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c_indices in class_indices:
        rng.shuffle(c_indices)
        # Sample proportions from Dirichlet for this class
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        # Convert to integer splits
        splits = (proportions * len(c_indices)).astype(int)
        # Fix rounding so total == len(c_indices)
        splits[-1] = len(c_indices) - splits[:-1].sum()

        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(
                c_indices[start : start + count].tolist()
            )
            start += count

    return {cid: np.array(idxs) for cid, idxs in client_indices.items()}


def partition_stats(
    partition: Dict[int, np.ndarray],
    targets: np.ndarray,
    num_classes: int,
) -> Dict[int, np.ndarray]:
    """
    Returns per-client class distribution as a (num_clients, num_classes) array.
    Useful for logging and verifying heterogeneity level.
    """
    stats = {}
    for cid, idxs in partition.items():
        counts = np.bincount(targets[idxs], minlength=num_classes)
        stats[cid] = counts
    return stats