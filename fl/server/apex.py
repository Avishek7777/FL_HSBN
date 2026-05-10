"""
fl/server/apex.py

Z2 — Server Apex (MLP)
========================
Single responsibility: coarse abstraction and top-down feedback generation.

Now receives z1_5 from Z1.5 (classifier bridge) instead of z1 directly.
z1_5 is classification-optimized — Z2 has meaningful structure to work with.

Gradient flow:
    coarse_loss → Z2 → z1_5 → Z1.5 → z1 → Z1 → z0_compressed → bottleneck

Input:  z1_5         (B, d_cls) — from Z1.5 classifier bridge
Output: z2           (B, d2)
        coarse_logits (B, num_coarse)
        coarse_loss   scalar
        downward_message (B, d_cls) — projects back up to d_cls for Z1.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Z2Apex(nn.Module):
    """
    Server-side Z2 Apex MLP.

    Args:
        d_cls              : input dim from Z1.5
        d2                 : apex compressed dim
        hidden_dim         : MLP hidden size
        num_coarse_classes : coarse classes (20 for CIFAR-100)
    """

    def __init__(
        self,
        d_cls             : int,
        d2                : int,
        hidden_dim        : int,
        num_coarse_classes: int,
    ):
        super().__init__()

        self.d_cls = d_cls
        self.d2    = d2

        # Compression: d_cls → d2
        self.encoder = nn.Sequential(
            nn.Linear(d_cls, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d2),
            nn.ReLU(),
        )

        # Coarse classification
        self.cls_head = nn.Linear(d2, num_coarse_classes)

        # Downward message: projects z2 back up to d_cls
        # Flows back through Z1.5 → Z1 → clients
        self.upward_proj = nn.Sequential(
            nn.Linear(d2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_cls),
        )

    def forward(
        self,
        z1_5         : torch.Tensor,  # (B, d_cls) — from Z1.5
        coarse_labels: torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z2            : (B, d2)
            coarse_logits : (B, num_coarse)
            coarse_loss   : scalar
        """
        z2            = self.encoder(z1_5)            # (B, d2)
        coarse_logits = self.cls_head(z2)             # (B, num_coarse)
        coarse_loss   = F.cross_entropy(coarse_logits, coarse_labels)

        return z2, coarse_logits, coarse_loss

    def downward_message(
        self,
        z2: torch.Tensor,             # (B, d2)
    ) -> torch.Tensor:                # (B, d_cls)
        """
        Project z2 back up to d_cls.
        Flows: Z2 → Z1.5.downward_message → Z1.adapter.downward_message → clients
        """
        return self.upward_proj(z2)


def build_apex(cfg: dict) -> Z2Apex:
    apex_cfg = cfg["apex"]
    return Z2Apex(
        d_cls              = cfg["classifier"]["d_cls"],
        d2                 = cfg["common"]["d2"],
        hidden_dim         = apex_cfg["hidden_dim"],
        num_coarse_classes = cfg["data"]["num_coarse_classes"],
    )