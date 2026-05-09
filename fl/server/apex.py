"""
fl/server/apex.py

Z2 — Server Apex (MLP)
========================
Most compressed, most abstract level. Operates on pooled z1
from the adapter — the globally coherent representation.

Responsibilities:
    1. Coarse classification on the global representation
    2. Generate top-down message that flows back through Z1 to clients

The downward message from Z2 is the global semantic signal —
it carries information about what abstract concepts the global
model is attending to, which Z1 then personalizes per client
before forwarding to each client's bottleneck.

Architecture:
    Simple MLP — Z2 should be a distillation point, not a
    complex processor. Complexity lives in Z1's cross-attention.
    Z2's job is to compress z1 into the most abstract summary
    and generate a coherent top-down signal.

Input:  z1_pooled  (B, d1)   — mean-pooled across clients from Z1
Output: z2         (B, d2)
        coarse logits (B, num_coarse_classes)
        downward message (B, d1) — projects back up to d1 for Z1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Z2Apex(nn.Module):
    """
    Server-side Z2 Apex MLP.

    Args:
        d1                  : input dim from Z1 (common space dim)
        d2                  : apex compressed dim
        hidden_dim          : MLP hidden layer size
        num_coarse_classes  : number of coarse classes
    """

    def __init__(
        self,
        d1                : int,
        d2                : int,
        hidden_dim        : int,
        num_coarse_classes: int,
    ):
        super().__init__()

        self.d1 = d1
        self.d2 = d2

        # Compression MLP: d1 -> d2
        self.encoder = nn.Sequential(
            nn.Linear(d1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d2),
            nn.ReLU(),
        )

        # Coarse classification head
        self.cls_head = nn.Linear(d2, num_coarse_classes)

        # Downward message: projects z2 back up to d1
        # This is what flows back to Z1 and then to clients
        # Larger projection = richer feedback signal
        self.upward_proj = nn.Sequential(
            nn.Linear(d2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d1),
        )

    def forward(
        self,
        z1_pooled    : torch.Tensor,   # (B, d1)
        coarse_labels: torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z1_pooled    : mean-pooled z1 across clients  (B, d1)
            coarse_labels: coarse class targets           (B,)

        Returns:
            z2           : apex representation            (B, d2)
            coarse_logits: classification output          (B, num_coarse)
            coarse_loss  : scalar
        """
        z2 = self.encoder(z1_pooled)                 # (B, d2)
        coarse_logits = self.cls_head(z2)            # (B, num_coarse)
        coarse_loss   = F.cross_entropy(coarse_logits, coarse_labels)

        return z2, coarse_logits, coarse_loss

    def downward_message(
        self,
        z2: torch.Tensor,              # (B, d2)
    ) -> torch.Tensor:                 # (B, d1)
        """
        Project z2 back up to d1 — the global top-down signal.
        Z1 receives this and uses it to generate per-client feedback.
        """
        return self.upward_proj(z2)    # (B, d1)


def build_apex(cfg: dict) -> Z2Apex:
    apex_cfg = cfg["apex"]
    return Z2Apex(
        d1                 = cfg["common"]["d1"],
        d2                 = cfg["common"]["d2"],
        hidden_dim         = apex_cfg["hidden_dim"],
        num_coarse_classes = cfg["data"]["num_coarse_classes"],
    )