"""
fl/client/base_client.py

FL Client
==========
Each client owns:
    1. A local model (Z0) — any architecture from architectures/
    2. A per-client bottleneck channel — compresses z0 into common space d1
       This is the ONLY thing that crosses to the server (z0_compressed)

Training signal has two components:
    1. Local task loss     — supervised by local labeled data
    2. Feedback alignment  — alpha-weighted signal from server
       "here is what the global model expects your representation to look like"

Gradient flow:
    local_task_loss
        └──► through local_model
        └──► through bottleneck (partial)

    feedback_alignment_loss
        └──► through bottleneck
        └──► through local_model (via alpha gate)

What crosses the client-server boundary:
    Client → Server : z0_compressed  (B, d1)  — bottleneck output only
    Server → Client : feedback vector (B, d1)  — top-down signal only

Nothing about the local model's weights ever leaves the client.
Nothing about the server's weights ever comes to the client.
Only representations and feedback signals cross the boundary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from hsbn.channels.bottleneck import BandwidthBottleneck


class FLClient:
    """
    Federated Learning Client.

    Args:
        client_id       : unique client identifier
        local_model     : Z0 — feature extractor (any architecture)
        bottleneck      : per-client bandwidth bottleneck channel
        dataloader      : local Non-IID data partition
        arch_name       : name of architecture file (for logging)
        lr              : learning rate
        local_epochs    : number of local training epochs per round
        device          : compute device
        alpha           : feedback alignment loss weight
        beta            : bottleneck bandwidth budget (for annealing)
    """

    def __init__(
        self,
        client_id   : int,
        local_model : nn.Module,
        bottleneck  : BandwidthBottleneck,
        dataloader  : DataLoader,
        arch_name   : str,
        num_classes : int,
        lr          : float = 1e-3,
        local_epochs: int   = 5,
        device      : str   = "cpu",
        alpha       : float = 0.15,
    ):
        self.client_id    = client_id
        self.arch_name    = arch_name
        self.device       = device
        self.local_epochs = local_epochs
        self.alpha        = alpha
        self.num_classes  = num_classes

        self.local_model = local_model.to(device)
        self.bottleneck  = bottleneck.to(device)
        self.dataloader  = dataloader

        # Local classification head — maps z0_compressed to local logits
        # Sits on top of bottleneck output, entirely client-private
        d1 = bottleneck.out_dim
        self.local_head = nn.Sequential(
            nn.Linear(d1, d1 * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d1 * 2, num_classes),
        ).to(device)

        # Optimizer covers local_model (trainable params) + bottleneck + head
        trainable = (
            [p for p in self.local_model.parameters() if p.requires_grad]
            + list(self.bottleneck.parameters())
            + list(self.local_head.parameters())
        )
        self.optimizer = optim.Adam(trainable, lr=lr)

        # Feedback buffer — server writes here, client reads during training
        self._feedback_buffer: Optional[torch.Tensor] = None  # (B, d1)

    # =========================================================================
    # Boundary — what crosses client ↔ server
    # =========================================================================

    def get_compressed(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run local model + bottleneck.
        Returns z0_compressed to send to server, and kl_loss for local update.

        This is the ONLY output that leaves the client.

        Args:
            x: input batch (B, C, H, W)

        Returns:
            z0_compressed : (B, d1) — sent to server
            kl_loss       : scalar  — stays local, used in backward
        """
        self.local_model.eval()   # pretrained frozen layers stay eval
        self.bottleneck.train()

        x = x.to(self.device)
        z0 = self.local_model(x)                    # (B, out_dim)
        z0_compressed, kl_loss = self.bottleneck(z0)  # (B, d1)
        return z0_compressed, kl_loss

    def receive_feedback(self, feedback: torch.Tensor):
        """
        Receive top-down feedback vector from server.
        Stored in buffer, used during next local training step.

        Args:
            feedback: (B, d1) — personalized feedback from dispatcher
        """
        self._feedback_buffer = feedback.detach().to(self.device)

    # =========================================================================
    # Local Training
    # =========================================================================

    def train_round(self, global_round: int) -> dict:
        """
        Run local_epochs of training using local task loss + feedback signal.

        Args:
            global_round: current FL round (for logging)

        Returns:
            metrics dict
        """
        self.local_model.train()
        self.bottleneck.train()
        self.local_head.train()

        total_task_loss     = 0.0
        total_feedback_loss = 0.0
        total_kl_loss       = 0.0
        num_batches         = 0

        for _ in range(self.local_epochs):
            for batch in self.dataloader:
                x, labels = self._unpack_batch(batch)
                x      = x.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # ── Forward ──────────────────────────────────────────────
                z0 = self.local_model(x)                    # (B, out_dim)
                z0_compressed, kl_loss = self.bottleneck(z0)  # (B, d1)

                # Local task loss via classification head
                local_logits = self.local_head(z0_compressed)
                task_loss    = F.cross_entropy(local_logits, labels)

                # ── Feedback alignment loss ───────────────────────────────
                # If feedback is available, penalize distance between
                # z0_compressed and what the server expects
                if self._feedback_buffer is not None:
                    feedback = self._feedback_buffer
                    # Handle batch size mismatch (last batch may differ)
                    min_b = min(z0_compressed.size(0), feedback.size(0))
                    feedback_loss = F.mse_loss(
                        z0_compressed[:min_b],
                        feedback[:min_b],
                    )
                else:
                    feedback_loss = torch.tensor(0.0, device=self.device)

                # ── Combined loss ─────────────────────────────────────────
                loss = (
                    task_loss
                    + self.alpha * feedback_loss
                    + 0.01 * kl_loss        # small KL weight — regularizer
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.optimizer.param_groups[0]['params']
                     if p.grad is not None],
                    max_norm=1.0,
                )
                self.optimizer.step()

                total_task_loss     += task_loss.item()
                total_feedback_loss += feedback_loss.item()
                total_kl_loss       += kl_loss.item()
                num_batches         += 1

        n = max(num_batches, 1)
        return {
            "client_id"        : self.client_id,
            "arch"             : self.arch_name,
            "avg_task_loss"    : total_task_loss / n,
            "avg_feedback_loss": total_feedback_loss / n,
            "avg_kl_loss"      : total_kl_loss / n,
            "num_batches"      : num_batches,
        }

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def local_accuracy(self) -> float:
        """
        Compute local top-1 accuracy on client's own data partition.
        Useful for monitoring per-client performance across rounds.
        """
        self.local_model.eval()
        self.bottleneck.eval()
        self.local_head.eval()

        correct = total = 0
        for batch in self.dataloader:
            x, labels = self._unpack_batch(batch)
            x      = x.to(self.device)
            labels = labels.to(self.device)

            z0            = self.local_model(x)
            z0_compressed, _ = self.bottleneck(z0)
            logits        = self.local_head(z0_compressed)
            preds         = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return correct / total if total > 0 else 0.0

    # =========================================================================
    # Utilities
    # =========================================================================

    def _unpack_batch(
        self,
        batch: tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack dataloader batch.
        CIFAR-100 returns (x, fine_label) — fine label is the local task target.
        """
        x, labels = batch
        return x, labels

    def num_local_samples(self) -> int:
        return len(self.dataloader.dataset)