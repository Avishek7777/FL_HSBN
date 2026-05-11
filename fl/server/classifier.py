"""
fl/server/classifier.py

Z1.5 — Classification Bridge
==============================
Sits between Z1 (alignment) and Z2 (coarse abstraction).
Single responsibility: fine classification from aligned representations.

Why this exists:
    Z1's job is alignment — variance minimization via cross-client attention.
    Asking Z1 to also classify creates conflicting gradients.
    Z1.5 receives already-aligned z1 and learns purely from classification
    signal. Z2 then receives z1_5 — semantically meaningful — and has
    real structure to do coarse classification on.

Gradient flow (unbroken chain):
    coarse_loss → Z2 → z1_5 → Z1.5 (cls_loss) → z1 → Z1 → z0_compressed
                                                          → bottleneck
                                                          → local_model

Input:  z1        (N, B, d1) — aligned client representations from Z1
        z0_public (B, d1)    — server encoder output for public data
Output: z1_5      (B, d_cls) — classification-optimized representation
        cls_loss  scalar
        logits    (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Z1_5Classifier(nn.Module):
    """
    Classification bridge between Z1 and Z2.

    Args:
        d1          : input dim from Z1 common space
        d_cls       : internal classification representation dim
        num_classes : fine classes (100 for CIFAR-100)
        dropout     : regularization
    """

    def __init__(
        self,
        d1         : int,
        d_cls      : int,
        num_classes: int,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.d1    = d1
        self.d_cls = d_cls

        # Projects pooled z1 into classification space
        self.pool_proj = nn.Sequential(
            nn.Linear(d1, d_cls),
            nn.LayerNorm(d_cls),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fine classification head
        self.cls_head = nn.Linear(d_cls, num_classes)

        # Downward message — projects d_cls back to d1 for feedback chain
        self.down_proj = nn.Sequential(
            nn.Linear(d_cls, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
        )

    def forward(
        self,
        z1         : torch.Tensor,    # (N, B, d1)
        z0_public  : torch.Tensor,    # (B, d1)
        fine_labels: torch.Tensor,    # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z1_5     : (B, d_cls)
            cls_loss : scalar
            logits   : (B, num_classes)
        """
        # Pool client representations across N clients
        z1_pooled = z1.mean(dim=0)                   # (B, d1)

        # Public data weighted heavily — Z1.5 learns classification from
        # the clean IID signal first. z1_pooled is detached — it conditions
        # without pulling classification gradients back into collapsed Z1.
        # As Z1 learns better representations, z1_pooled becomes more useful.
        z_mixed = 0.2 * z1_pooled.detach() + 0.8 * z0_public  # (B, d1)

        # Project to classification space
        z1_5   = self.pool_proj(z_mixed)             # (B, d_cls)
        logits = self.cls_head(z1_5)                 # (B, num_classes)

        # Z1.5 local objective — fine classification
        cls_loss = F.cross_entropy(logits, fine_labels)

        # Additionally supervise directly on public data alone
        # Ensures Z1.5 has its own strong local objective independent of Z1
        z1_5_pub   = self.pool_proj(z0_public)
        logits_pub = self.cls_head(z1_5_pub)
        pub_cls_loss = F.cross_entropy(logits_pub, fine_labels)

        total_cls_loss = cls_loss + 0.5 * pub_cls_loss

        return z1_5, total_cls_loss, logits

    def downward_message(
        self,
        z1_5: torch.Tensor,           # (B, d_cls)
    ) -> torch.Tensor:                # (B, d1)
        """Projects z1_5 back to d1 for the feedback chain."""
        return self.down_proj(z1_5)


def build_classifier(cfg: dict) -> Z1_5Classifier:
    cls_cfg = cfg["classifier"]
    return Z1_5Classifier(
        d1          = cfg["common"]["d1"],
        d_cls       = cls_cfg["d_cls"],
        num_classes = cfg["data"]["num_fine_classes"],
        dropout     = cls_cfg.get("dropout", 0.1),
    )