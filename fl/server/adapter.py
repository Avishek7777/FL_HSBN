"""
fl/server/adapter.py

Z1 — Server Adapter (Transformer)
===================================
Receives compressed representations from all participating clients
(z0_compressed, all of dim d1) and produces refined representations z1.

Why Transformer:
    Z1 sees a SET of client representations each round. A Transformer
    over that set means each client's z1 is computed with awareness of
    what other clients submitted — implicit cross-client collaboration
    without any explicit coordination or weight sharing.

    This is fundamentally different from processing each client
    independently — the attention mechanism lets the adapter ask:
    "given what everyone else submitted, what should this client's
    representation look like in the common space?"

Dual Objective:
    1. Variance loss     — pulls client representations together
                           forces the common space to be consistent
                           across heterogeneous architectures
    2. Classification    — keeps the common space semantically meaningful
                           prevents variance collapse to a trivial constant

Forward pass:
    Input  : stacked z0_compressed from N clients  (N, B, d1)
    Output : z1 per client                         (N, B, d1)
             variance_loss scalar
             classification_loss scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class ClientCrossAttention(nn.Module):
    """
    Transformer encoder that attends over client representations.

    Takes a batch of client representations stacked as a sequence
    and applies multi-head self-attention so each client's token
    attends to all other clients' tokens.

    Input:  (N, B, d1)  — N clients, B batch size, d1 features
    Output: (N, B, d1)  — same shape, cross-client informed
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
            batch_first     = True,   # (B, seq, d) convention
            norm_first      = True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

    def forward(
        self,
        client_reprs: torch.Tensor,  # (N, B, d1)
    ) -> torch.Tensor:               # (N, B, d1)
        N, B, d1 = client_reprs.shape

        # Reshape: treat each sample's N client tokens as a sequence
        # (B, N, d1) — batch of sequences, each sequence = N client tokens
        x = client_reprs.permute(1, 0, 2)           # (B, N, d1)
        x = self.transformer(x)                      # (B, N, d1)
        return x.permute(1, 0, 2)                    # (N, B, d1)


class Z1Adapter(nn.Module):
    """
    Server-side Z1 Adapter.

    Assembles cross-client attention, variance objective,
    and classification objective into one module.

    Args:
        d1              : common representation dimension (e.g. 128)
        num_heads       : attention heads in Transformer
        num_layers      : Transformer encoder layers
        dropout         : attention dropout
        num_classes     : number of classes for classification objective
        lambda_var      : weight for variance loss
        lambda_cls      : weight for classification loss
    """

    def __init__(
        self,
        d1         : int,
        num_heads  : int,
        num_layers : int,
        dropout    : float,
        num_classes: int,
        lambda_var : float = 1.0,
        lambda_cls : float = 1.0,
    ):
        super().__init__()

        self.d1         = d1
        self.lambda_var = lambda_var
        self.lambda_cls = lambda_cls

        # Cross-client Transformer
        self.cross_attn = ClientCrossAttention(
            d1         = d1,
            num_heads  = num_heads,
            num_layers = num_layers,
            dropout    = dropout,
        )

        # Per-client layer norm after attention
        self.norm = nn.LayerNorm(d1)

        # Classification head on pooled z1
        self.cls_head = nn.Linear(d1, num_classes)

        # Downward message projection for top-down feedback
        # Projects z1 -> feedback vector (same dim d1)
        self.feedback_proj = nn.Linear(d1, d1)

    def forward(
        self,
        client_reprs : torch.Tensor,   # (N, B, d1) — stacked z0_compressed
        labels       : torch.Tensor,   # (N*B,) or (N, B) — fine labels
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            client_reprs : stacked compressed representations (N, B, d1)
            labels       : class labels for classification objective

        Returns:
            z1           : refined representations          (N, B, d1)
            variance_loss: scalar
            cls_loss     : scalar
        """
        N, B, d1 = client_reprs.shape

        # Cross-client attention — each client attends to all others
        z1 = self.cross_attn(client_reprs)           # (N, B, d1)
        z1 = self.norm(z1)

        # ── Variance loss ─────────────────────────────────────────────────
        # Mean representation across clients per sample
        # We want all clients to produce similar z1 for the same concepts
        z1_mean = z1.mean(dim=0, keepdim=True)       # (1, B, d1)
        variance_loss = ((z1 - z1_mean) ** 2).mean()

        # ── Classification loss ───────────────────────────────────────────
        # Pool across clients then classify
        z1_pooled = z1.mean(dim=0)                   # (B, d1)
        logits    = self.cls_head(z1_pooled)         # (B, num_classes)

        # Labels may be (N, B) — use first client's labels as global target
        if labels.dim() == 2:
            flat_labels = labels[0]                  # (B,)
        else:
            flat_labels = labels[:B]                 # (B,)

        cls_loss = F.cross_entropy(logits, flat_labels)

        return z1, variance_loss, cls_loss

    def downward_message(
        self,
        z1: torch.Tensor,              # (N, B, d1)
    ) -> torch.Tensor:                 # (N, B, d1)
        """
        Produces per-client feedback vectors from z1.
        Each client gets a personalized signal shaped by cross-client attention.
        """
        return self.feedback_proj(z1)  # (N, B, d1)

    def total_loss(
        self,
        variance_loss: torch.Tensor,
        cls_loss     : torch.Tensor,
    ) -> torch.Tensor:
        return self.lambda_var * variance_loss + self.lambda_cls * cls_loss


def build_adapter(cfg: dict) -> Z1Adapter:
    adapter_cfg = cfg["adapter"]
    return Z1Adapter(
        d1          = cfg["common"]["d1"],
        num_heads   = adapter_cfg["num_heads"],
        num_layers  = adapter_cfg["num_layers"],
        dropout     = adapter_cfg["dropout"],
        num_classes = cfg["data"]["num_fine_classes"],
        lambda_var  = adapter_cfg.get("lambda_var", 1.0),
        lambda_cls  = adapter_cfg.get("lambda_cls", 1.0),
    )