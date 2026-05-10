"""
fl/runner.py

FL Runner — Updated
====================
Changes from v1:
    1. Public data split  — server trains on IID balanced data every round
                            via a dedicated ServerEncoder → Z1 → Z2 pathway
    2. Multi-step server  — server updates `server_update_steps` times per
                            round, not just once (was severely undertrained)
    3. All client labels  — Z1 cls_loss uses ALL selected clients' labels,
                            not just client 0's skewed batch
    4. Server encoder     — lightweight CNN projects public data into d1
                            same space as client bottleneck outputs

Communication protocol (unchanged):
    Client → Server : z0_compressed  (B, d1)
    Server → Client : feedback vector (B, d1)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List

from fl.client.client_factory import build_all_clients
from fl.server.adapter import build_adapter
from fl.server.apex import build_apex
from fl.server.encoder import build_server_encoder
from fl.server.feedback import build_feedback_dispatcher
from fl.data.dirichlet import dirichlet_partition, carve_public_split
from fl.data.loaders import (
    build_client_loaders,
    build_public_loader,
    build_global_test_loader,
)
from fl.client.base_client import fine_to_coarse


class FLRunner:

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        fl_cfg     = cfg["fl"]
        data_cfg   = cfg["data"]
        server_cfg = cfg["server"]

        self.num_clients     = fl_cfg["num_clients"]
        self.num_rounds      = fl_cfg["num_rounds"]
        self.client_fraction = fl_cfg["client_fraction"]
        self.seed            = fl_cfg.get("seed", 42)
        self.arch_dir        = fl_cfg["arch_dir"]
        self.data_root       = data_cfg["root"]
        self.batch_size      = data_cfg["batch_size"]
        self.alpha           = data_cfg["dirichlet_alpha"]
        self.server_steps    = server_cfg.get("update_steps", 5)
        self.warmup_epochs   = fl_cfg.get("warmup_epochs", 1)

        # ── Data ──────────────────────────────────────────────────────────
        from torchvision.datasets import CIFAR100
        import torchvision.transforms as T

        _ds     = CIFAR100(
            root=self.data_root, train=True,
            download=True, transform=T.ToTensor()
        )
        targets = np.array(_ds.targets)

        # Carve public split BEFORE partitioning
        public_indices, private_indices = carve_public_split(
            targets,
            samples_per_class = data_cfg.get("public_samples_per_class", 50),
            seed              = self.seed,
        )
        private_targets = targets[private_indices]

        print(f"\nData split:")
        print(f"  Public  (server IID) : {len(public_indices):,} samples")
        print(f"  Private (clients)    : {len(private_indices):,} samples\n")

        # Dirichlet partition on private split only
        partition = dirichlet_partition(
            private_targets, self.num_clients, self.alpha, seed=self.seed
        )
        # Remap local indices back to global dataset indices
        partition = {
            cid: private_indices[local_idxs]
            for cid, local_idxs in partition.items()
        }

        self.client_loaders = build_client_loaders(
            partition, self.data_root, self.batch_size
        )
        self.client_sizes = {
            cid: len(idxs) for cid, idxs in partition.items()
        }
        self.public_loader = build_public_loader(
            public_indices, self.data_root, self.batch_size
        )
        self.public_iter = iter(self.public_loader)

        self.test_loader = build_global_test_loader(
            self.data_root, self.batch_size
        )

        # ── Clients ───────────────────────────────────────────────────────
        self.clients: Dict[int, object] = build_all_clients(
            num_clients  = self.num_clients,
            arch_dir     = self.arch_dir,
            dataloaders  = self.client_loaders,
            d1           = cfg["common"]["d1"],
            channel_cfg  = cfg["channel"],
            num_classes  = data_cfg["num_fine_classes"],
            lr           = fl_cfg["lr"],
            local_epochs = fl_cfg["local_epochs"],
            alpha        = cfg["feedback"]["alpha"],
            device       = device,
            global_seed  = self.seed,
        )

        # ── Server ────────────────────────────────────────────────────────
        self.adapter        = build_adapter(cfg).to(device)
        self.apex           = build_apex(cfg).to(device)
        self.server_encoder = build_server_encoder(cfg).to(device)
        self.dispatcher     = build_feedback_dispatcher(cfg)

        # Separate optimizers — encoder trains with adapter
        self.adapter_optimizer = optim.Adam(
            list(self.adapter.parameters())
            + list(self.server_encoder.parameters()),
            lr = server_cfg["adapter_lr"],
        )
        self.apex_optimizer = optim.Adam(
            self.apex.parameters(),
            lr = server_cfg["apex_lr"],
        )

        self.history: List[dict] = []

    # =========================================================================
    # Warmup
    # =========================================================================

    def _client_warmup(self):
        """Pre-train all clients locally before any server interaction."""
        print(f"Client warmup — {self.warmup_epochs} epoch(s)...")
        print("─" * 50)

        for cid, client in self.clients.items():
            client.local_model.train()
            client.bottleneck.train()
            client.local_head.train()

            total_loss  = 0.0
            num_batches = 0

            for _ in range(self.warmup_epochs):
                for batch in client.dataloader:
                    x, labels = batch
                    x      = x.to(self.device)
                    labels = labels.to(self.device)

                    client.optimizer.zero_grad()
                    z0                     = client.local_model(x)
                    z0_compressed, kl_loss = client.bottleneck(z0)
                    logits                 = client.local_head(z0_compressed)
                    loss = F.cross_entropy(logits, labels) + 0.01 * kl_loss
                    loss.backward()
                    client.optimizer.step()

                    total_loss  += loss.item()
                    num_batches += 1

            avg = total_loss / max(num_batches, 1)
            print(
                f"  Client {cid:02d} ({client.arch_name:<20}) "
                f"avg_loss={avg:.4f}"
            )

        print("─" * 50)
        print("Warmup complete.\n")

    # =========================================================================
    # Public batch helper
    # =========================================================================

    def _next_public_batch(self):
        """Get next public batch, reset iterator when exhausted."""
        try:
            return next(self.public_iter)
        except StopIteration:
            self.public_iter = iter(self.public_loader)
            return next(self.public_iter)

    # =========================================================================
    # Server update
    # =========================================================================

    def _server_update(
        self,
        stacked_compressed: torch.Tensor,  # (N, B, d1)
        stacked_labels    : torch.Tensor,  # (N, B)
        stacked_coarse    : torch.Tensor,  # (N, B)
    ):
        """
        Update Z1 and Z2 using client representations + public data.
        Runs server_update_steps times per round.
        """
        total_server_loss  = 0.0
        total_var_loss     = 0.0
        total_cls_loss     = 0.0
        total_coarse_loss  = 0.0
        total_pub_loss     = 0.0

        z1_final = z2_final = None

        for step in range(self.server_steps):
            self.adapter_optimizer.zero_grad()
            self.apex_optimizer.zero_grad()

            # ── Client representations pathway ────────────────────────────
            z1, variance_loss, cls_loss = self.adapter(
                stacked_compressed,
                stacked_labels,
            )
            z1_pooled     = z1.mean(dim=0)                   # (B, d1)
            coarse_global = stacked_coarse[0]                # (B,)
            z2, _, coarse_loss = self.apex(z1_pooled, coarse_global)

            # ── Public data pathway ───────────────────────────────────────
            pub_x, pub_labels = self._next_public_batch()
            pub_x      = pub_x.to(self.device)
            pub_labels = pub_labels.to(self.device)
            pub_coarse = fine_to_coarse(pub_labels)

            z0_pub            = self.server_encoder(pub_x)      # (B, d1)
            z0_pub_stacked    = z0_pub.unsqueeze(0)             # (1, B, d1)
            pub_labels_stacked = pub_labels.unsqueeze(0)        # (1, B)

            z1_pub, _, cls_pub = self.adapter(
                z0_pub_stacked, pub_labels_stacked
            )
            z1_pub_pooled = z1_pub.mean(dim=0)
            _, _, coarse_pub = self.apex(z1_pub_pooled, pub_coarse)

            pub_loss = cls_pub + coarse_pub

            # ── Combined loss ─────────────────────────────────────────────
            adapter_loss = self.adapter.total_loss(variance_loss, cls_loss)
            server_loss  = adapter_loss + coarse_loss + pub_loss

            server_loss.backward(
                retain_graph=(step < self.server_steps - 1)
            )
            torch.nn.utils.clip_grad_norm_(
                list(self.adapter.parameters())
                + list(self.server_encoder.parameters()),
                max_norm=1.0,
            )
            self.adapter_optimizer.step()
            self.apex_optimizer.step()

            total_server_loss += server_loss.item()
            total_var_loss    += variance_loss.item()
            total_cls_loss    += cls_loss.item()
            total_coarse_loss += coarse_loss.item()
            total_pub_loss    += pub_loss.item()

            if step == self.server_steps - 1:
                z1_final = z1.detach()
                z2_final = z2.detach()

        n = self.server_steps
        losses = {
            "server_loss"  : total_server_loss / n,
            "variance_loss": total_var_loss    / n,
            "cls_loss"     : total_cls_loss    / n,
            "coarse_loss"  : total_coarse_loss / n,
            "pub_loss"     : total_pub_loss    / n,
        }
        return z1_final, z2_final, losses

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self):
        self._client_warmup()

        rng = np.random.default_rng(self.seed)

        for round_idx in range(self.num_rounds):

            # ── 1. Sample clients ─────────────────────────────────────────
            n_selected = max(1, int(self.num_clients * self.client_fraction))
            selected   = rng.choice(
                self.num_clients, size=n_selected, replace=False
            ).tolist()

            # ── 2. Bottom-up pass ─────────────────────────────────────────
            compressed_list = []
            label_list      = []
            coarse_list     = []

            for cid in selected:
                client = self.clients[cid]
                batch  = next(iter(client.dataloader))
                x, fine_labels = batch
                x             = x.to(self.device)
                fine_labels   = fine_labels.to(self.device)
                coarse_labels = fine_to_coarse(fine_labels)

                z0_compressed, _ = client.get_compressed(x)
                compressed_list.append(z0_compressed)
                label_list.append(fine_labels)
                coarse_list.append(coarse_labels)

            stacked_compressed = torch.stack(compressed_list, dim=0)
            stacked_labels     = torch.stack(label_list, dim=0)
            stacked_coarse     = torch.stack(coarse_list, dim=0)

            # ── 3. Server update ──────────────────────────────────────────
            z1, z2, losses = self._server_update(
                stacked_compressed,
                stacked_labels,
                stacked_coarse,
            )

            # ── 4. Top-down feedback ──────────────────────────────────────
            with torch.no_grad():
                global_signal = self.apex.downward_message(z2)
                client_z1s    = self.adapter.downward_message(z1)
                feedback      = self.dispatcher.dispatch(
                    global_signal = global_signal,
                    client_z1s    = client_z1s,
                    client_ids    = selected,
                )

            for i, cid in enumerate(selected):
                self.clients[cid].receive_feedback(
                    self.dispatcher.get_client_feedback(feedback, i)
                )

            # ── 5. Client local training ──────────────────────────────────
            round_metrics = []
            for cid in selected:
                metrics = self.clients[cid].train_round(
                    global_round=round_idx
                )
                round_metrics.append(metrics)

            # ── 6. Evaluate ───────────────────────────────────────────────
            acc = self.evaluate()

            log = {
                "round"           : round_idx,
                "top1_acc"        : acc,
                **losses,
                "client_logs"     : round_metrics,
                "selected_clients": selected,
            }
            self.history.append(log)

            print(
                f"Round {round_idx:03d} | "
                f"Top-1: {acc:.4f} | "
                f"Server: {losses['server_loss']:.4f} "
                f"(var={losses['variance_loss']:.3f} "
                f"cls={losses['cls_loss']:.3f} "
                f"coarse={losses['coarse_loss']:.3f} "
                f"pub={losses['pub_loss']:.3f}) | "
                f"Clients: {selected}"
            )

        return self.history

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Global test accuracy via server encoder → Z1 → cls_head.
        Consistent across rounds — not tied to any client architecture.
        """
        self.adapter.eval()
        self.apex.eval()
        self.server_encoder.eval()

        correct = total = 0

        for x, fine_labels in self.test_loader:
            x           = x.to(self.device)
            fine_labels = fine_labels.to(self.device)

            z0_pub  = self.server_encoder(x)
            stacked = z0_pub.unsqueeze(0)
            labels  = fine_labels.unsqueeze(0)

            z1, _, _ = self.adapter(stacked, labels)
            z1_pooled = z1.mean(dim=0)
            logits    = self.adapter.cls_head(z1_pooled)
            preds     = logits.argmax(dim=1)

            correct += (preds == fine_labels).sum().item()
            total   += fine_labels.size(0)

        self.adapter.train()
        self.apex.train()
        self.server_encoder.train()

        return correct / total if total > 0 else 0.0