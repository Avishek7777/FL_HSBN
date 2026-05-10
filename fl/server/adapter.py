"""
fl/server/adapter.py

Z1 — Server Adapter (Transformer)
===================================
Single responsibility: alignment.

Receives compressed representations from all participating clients
and produces aligned representations via cross-client attention.
Variance minimization is the only objective here.

Classification has been moved to Z1.5 (classifier.py).
This resolves the conflicting gradient problem — variance loss
and classification loss were fighting each other. Now Z1 aligns
cleanly and Z1.5 classifies cleanly.

Gradient flow into Z1:
    cls_loss (Z1.5) → z1 → Z1 (variance_loss) → z0_compressed
                                                → bottleneck
                                                → local_model

Z1 receives gradients from both its own variance_loss AND from
Z1.5's cls_loss flowing back through z1. Both signals shape the
alignment Transformer — making it align in a way that is also
useful for downstream classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ClientCrossAttention(nn.Module):
    """
    Transformer encoder over client representations.
    Each client's token attends to all other clients' tokens.

    Input:  (N, B, d1)
    Output: (N, B, d1)
    """

    def __init__(
        self,
        d1        : int,
        num_heads : int,
        num_layers: int,
        dropout   : float,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d1,
            nhead           = num_heads,
            dim_feedforward = d1 * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
            #enable_nested_tensor = False,   # silence warning
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

    def forward(self, client_reprs: torch.Tensor) -> torch.Tensor:
        N, B, d1 = client_reprs.shape
        x = client_reprs.permute(1, 0, 2)           # (B, N, d1)
        x = self.transformer(x)                      # (B, N, d1)
        return x.permute(1, 0, 2)                    # (N, B, d1)


class Z1Adapter(nn.Module):
    """
    Z1 — Alignment only.

    Args:
        d1         : common representation dimension
        num_heads  : attention heads
        num_layers : Transformer encoder layers
        dropout    : attention dropout
        lambda_var : variance loss weight
    """

    def __init__(
        self,
        d1        : int,
        num_heads : int,
        num_layers: int,
        dropout   : float,
        lambda_var: float = 1.0,
    ):
        super().__init__()

        self.d1         = d1
        self.lambda_var = lambda_var

        self.cross_attn = ClientCrossAttention(
            d1         = d1,
            num_heads  = num_heads,
            num_layers = num_layers,
            dropout    = dropout,
        )
        self.norm = nn.LayerNorm(d1)

        # Downward message projection — for feedback chain
        self.feedback_proj = nn.Linear(d1, d1)

    def forward(
        self,
        client_reprs: torch.Tensor,   # (N, B, d1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            client_reprs : stacked z0_compressed from clients (N, B, d1)

        Returns:
            z1           : aligned representations  (N, B, d1)
            variance_loss: scalar — alignment objective
        """
        z1 = self.cross_attn(client_reprs)           # (N, B, d1)
        z1 = self.norm(z1)

        # Variance loss — pulls client representations together
        z1_mean       = z1.mean(dim=0, keepdim=True) # (1, B, d1)
        variance_loss = ((z1 - z1_mean) ** 2).mean()

        return z1, variance_loss

    def downward_message(
        self,
        z1: torch.Tensor,             # (N, B, d1)
    ) -> torch.Tensor:                # (N, B, d1)
        """Per-client feedback vectors from aligned z1."""
        return self.feedback_proj(z1)

    def total_loss(self, variance_loss: torch.Tensor) -> torch.Tensor:
        return self.lambda_var * variance_loss


def build_adapter(cfg: dict) -> Z1Adapter:
    adapter_cfg = cfg["adapter"]
    return Z1Adapter(
        d1         = cfg["common"]["d1"],
        num_heads  = adapter_cfg["num_heads"],
        num_layers = adapter_cfg["num_layers"],
        dropout    = adapter_cfg["dropout"],
        lambda_var = adapter_cfg.get("lambda_var", 1.0),
    )