"""
fl/server/encoder.py

Server-Side Public Data Encoder
=================================
A lightweight CNN that processes public IID data into d1-dimensional
representations before passing them through Z1 and Z2.

Why a separate encoder:
    Client data enters Z1 via per-client bottlenecks (client-owned).
    Public data has no associated client — it needs its own pathway
    into the common d1 space. This encoder is that pathway.

    The encoder is server-owned and trained server-side. It gives Z1
    a continuous IID signal that is completely independent of which
    clients participated this round or how skewed their distributions are.

Two training pathways into Z1:
    Client pathway:  x → client_model → bottleneck → z0_compressed → Z1
    Public pathway:  x → server_encoder → z0_public → Z1

Both produce (B, d1) tensors. Z1 sees them identically.

Architecture:
    Small CNN — fast, low memory, just enough capacity to produce
    meaningful d1-dim representations from 32x32 CIFAR images.
    Intentionally simpler than client architectures — this is a
    server-side utility, not a competitive model.
"""

import torch
import torch.nn as nn


class ServerEncoder(nn.Module):
    """
    Lightweight CNN encoder for server-side public data.

    Args:
        d1      : output dimension — must match common space dim
        dropout : regularization
    """

    def __init__(self, d1: int, dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 8x8 -> 4x4

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),           # -> (B, 256, 1, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Project to d1 — same target space as client bottlenecks
        self.proj = nn.Linear(256, d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: public data batch (B, 3, 32, 32)

        Returns:
            z: (B, d1) — ready to stack with client z0_compressed
        """
        z = self.encoder(x)
        z = z.flatten(1)                            # (B, 256)
        z = self.dropout(z)
        return self.proj(z)                         # (B, d1)


def build_server_encoder(cfg: dict) -> ServerEncoder:
    return ServerEncoder(
        d1      = cfg["common"]["d1"],
        dropout = cfg["server"].get("encoder_dropout", 0.1),
    )