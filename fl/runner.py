"""
fl/runner.py

FL Runner — Full Gradient Chain
=================================
Every component learns from every other component.

Gradient chain (unbroken):
    coarse_loss → Z2 → z1_5 → Z1.5 (cls_loss) → z1 → Z1 (var_loss)
                                                       → z0_compressed
                                                       → bottleneck
                                                       → local_model

Server components and their single responsibilities:
    Z1  (adapter)    : alignment via cross-client attention + variance loss
    Z1.5 (classifier): fine classification from aligned representations
    Z2  (apex)       : coarse classification + top-down feedback generation

Three server optimizers:
    adapter_optimizer    : Z1 + server_encoder
    classifier_optimizer : Z1.5
    apex_optimizer       : Z2

Communication protocol:
    Client → Server : z0_compressed  (B, d1)  — WITH gradients
    Server → Client : feedback vector (B, d1)  — top-down signal
                      bottleneck gradient       — ∂loss/∂z0_compressed
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List

from fl.client.client_factory import build_all_clients
from fl.server.adapter import build_adapter
from fl.server.classifier import build_classifier
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

        public_indices, private_indices = carve_public_split(
            targets,
            samples_per_class = data_cfg.get("public_samples_per_class", 50),
            seed              = self.seed,
        )
        private_targets = targets[private_indices]

        print(f"\nData split:")
        print(f"  Public  (server IID) : {len(public_indices):,} samples")
        print(f"  Private (clients)    : {len(private_indices):,} samples\n")

        partition = dirichlet_partition(
            private_targets, self.num_clients, self.alpha, seed=self.seed
        )
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
        self.classifier     = build_classifier(cfg).to(device)
        self.apex           = build_apex(cfg).to(device)
        self.server_encoder = build_server_encoder(cfg).to(device)
        self.dispatcher     = build_feedback_dispatcher(cfg)

        # Three separate optimizers — one per responsibility
        self.adapter_optimizer = optim.Adam(
            list(self.adapter.parameters())
            + list(self.server_encoder.parameters()),
            lr = server_cfg["adapter_lr"],
        )
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
            lr = server_cfg["classifier_lr"],
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
        """Pre-train all clients locally before server interaction."""
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
        try:
            return next(self.public_iter)
        except StopIteration:
            self.public_iter = iter(self.public_loader)
            return next(self.public_iter)

    # =========================================================================
    # Server update — full gradient chain
    # =========================================================================

    def _server_update(
        self,
        clients_data  : list,          # list of (client, x, fine_labels)
        stacked_coarse: torch.Tensor,  # (N, B)
    ):
        """
        Full server update with unbroken gradient chain.

        Key difference from v1:
            z0_compressed is recomputed INSIDE this function with
            gradients enabled. This means server backward pass flows
            all the way back through client bottlenecks.

        Steps per update:
            1. Recompute z0_compressed with gradients (client bottlenecks)
            2. Z1 forward → z1, variance_loss
            3. Public data → server_encoder → z0_public
            4. Z1.5 forward → z1_5, cls_loss
            5. Z2 forward → z2, coarse_loss
            6. Total loss backward → updates all server components
                                  → gradients flow to client bottlenecks
        """
        total_var_loss    = 0.0
        total_cls_loss    = 0.0
        total_coarse_loss = 0.0
        total_server_loss = 0.0
        total_recon_loss  = 0.0

        z1_final = z2_final = z1_5_final = None

        for step in range(self.server_steps):
            self.adapter_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            self.apex_optimizer.zero_grad()

            # Zero client bottleneck grads too — they participate
            for client, _, _ in clients_data:
                client.optimizer.zero_grad()

            # ── Recompute z0_compressed WITH gradients ────────────────────
            compressed_list = []
            label_list      = []

            for client, x, fine_labels in clients_data:
                client.local_model.train()
                client.bottleneck.train()

                z0            = client.local_model(x)
                z0_c, kl_loss = client.bottleneck(z0)   # gradients flow here
                compressed_list.append(z0_c)
                label_list.append(fine_labels)

            stacked_compressed = torch.stack(compressed_list, dim=0)  # (N,B,d1)
            stacked_labels     = torch.stack(label_list, dim=0)       # (N,B)

            # ── Z1 forward — alignment + local recon objective ───────────
            z1, variance_loss, recon_loss = self.adapter(stacked_compressed)

            # ── Public data → server encoder ──────────────────────────────
            pub_x, pub_labels = self._next_public_batch()
            pub_x      = pub_x.to(self.device)
            pub_labels = pub_labels.to(self.device)
            pub_coarse = fine_to_coarse(pub_labels)

            z0_public = self.server_encoder(pub_x)                    # (B, d1)

            # ── Z1.5 forward — use pub_labels for supervision ─────────────
            # pub_labels match pub_x — correct supervision
            # stacked_labels[0] are client labels — wrong for public data
            z1_5, cls_loss, _ = self.classifier(
                z1, z0_public, pub_labels
            )

            # ── Z2 forward — use pub_coarse for supervision ───────────────
            z2, _, coarse_loss = self.apex(z1_5, pub_coarse)

            # ── Combined loss — all per-level objectives ──────────────────
            # lambda_var=0.1 — soft alignment, class structure survives
            # recon keeps Z1 from discarding information while aligning
            adapter_loss = self.adapter.total_loss(
                variance_loss, recon_loss, lambda_recon=0.5
            )
            server_loss = adapter_loss + cls_loss + coarse_loss

            # Backward — gradients flow through Z2 → Z1.5 → Z1 → bottlenecks
            server_loss.backward(
                retain_graph=(step < self.server_steps - 1)
            )

            # Clip server gradients
            for params in [
                list(self.adapter.parameters()),
                list(self.classifier.parameters()),
                list(self.apex.parameters()),
                list(self.server_encoder.parameters()),
            ]:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            # Update server components
            self.adapter_optimizer.step()
            self.classifier_optimizer.step()
            self.apex_optimizer.step()

            # Update client bottlenecks — gradients flowed back to them
            for client, _, _ in clients_data:
                torch.nn.utils.clip_grad_norm_(
                    client.bottleneck.parameters(), max_norm=1.0
                )
                client.optimizer.step()

            total_var_loss    += variance_loss.item()
            total_cls_loss    += cls_loss.item()
            total_coarse_loss += coarse_loss.item()
            total_server_loss += server_loss.item()
            total_recon_loss  += recon_loss.item()

            if step == self.server_steps - 1:
                z1_final   = z1.detach()
                z1_5_final = z1_5.detach()
                z2_final   = z2.detach()

        n = self.server_steps
        losses = {
            "server_loss"  : total_server_loss / n,
            "variance_loss": total_var_loss    / n,
            "recon_loss"   : total_recon_loss  / n,
            "cls_loss"     : total_cls_loss    / n,
            "coarse_loss"  : total_coarse_loss / n,
        }
        return z1_final, z1_5_final, z2_final, losses

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

            # ── 2. Prepare client data for this round ─────────────────────
            # Collect (client, x, labels) — NOT z0_compressed yet
            # z0_compressed is recomputed inside _server_update with grads
            clients_data  = []
            coarse_list   = []

            for cid in selected:
                client = self.clients[cid]
                batch  = next(iter(client.dataloader))
                x, fine_labels = batch
                x             = x.to(self.device)
                fine_labels   = fine_labels.to(self.device)
                coarse_labels = fine_to_coarse(fine_labels)

                clients_data.append((client, x, fine_labels))
                coarse_list.append(coarse_labels)

            stacked_coarse = torch.stack(coarse_list, dim=0)          # (N, B)

            # ── 3. Server update with full gradient chain ─────────────────
            z1, z1_5, z2, losses = self._server_update(
                clients_data,
                stacked_coarse,
            )

            # ── 4. Top-down feedback ──────────────────────────────────────
            # Full feedback chain: Z2 → Z1.5 → Z1 → clients
            with torch.no_grad():
                # Z2 → Z1.5
                apex_msg       = self.apex.downward_message(z2)       # (B, d_cls)
                # Z1.5 → Z1
                cls_msg        = self.classifier.downward_message(apex_msg)
                                                                       # (B, d1)
                # Z1 → per-client
                client_z1s     = self.adapter.downward_message(z1)    # (N, B, d1)

                # Blend global signal with per-client z1
                global_signal  = cls_msg                              # (B, d1)
                feedback       = self.dispatcher.dispatch(
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
                f"var={losses['variance_loss']:.3f} "
                f"recon={losses['recon_loss']:.3f} "
                f"cls={losses['cls_loss']:.3f} "
                f"coarse={losses['coarse_loss']:.3f} | "
                f"Clients: {selected}"
            )

        return self.history

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate via server encoder → Z1 → Z1.5 → cls_head.
        Consistent across rounds — not tied to any client architecture.
        """
        self.adapter.eval()
        self.classifier.eval()
        self.apex.eval()
        self.server_encoder.eval()

        correct = total = 0

        for x, fine_labels in self.test_loader:
            x           = x.to(self.device)
            fine_labels = fine_labels.to(self.device)

            # Server encoder → Z1
            z0_pub  = self.server_encoder(x)
            stacked = z0_pub.unsqueeze(0)                              # (1, B, d1)

            z1, _, _ = self.adapter(stacked)
            z1_5, _, logits = self.classifier(
                z1, z0_pub, fine_labels
            )

            preds   = logits.argmax(dim=1)
            correct += (preds == fine_labels).sum().item()
            total   += fine_labels.size(0)

        self.adapter.train()
        self.classifier.train()
        self.apex.train()
        self.server_encoder.train()

        return correct / total if total > 0 else 0.0