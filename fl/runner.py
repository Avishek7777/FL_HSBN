"""
fl/runner.py

FL Runner
==========
Orchestrates the full feedback-driven FL training loop.

No FedAvg. No weight sharing. Communication protocol:
    Client → Server : z0_compressed  (B, d1)  — bottleneck output only
    Server → Client : feedback vector (B, d1)  — top-down signal only

Round structure:
    1. Sample client fraction
    2. Each selected client runs a forward pass → z0_compressed
    3. Server stacks all z0_compressed → (N, B, d1)
    4. Z1 adapter forward → z1, variance_loss, cls_loss
    5. Z2 apex forward on pooled z1 → z2, coarse_loss
    6. Server backward → update Z1 and Z2 (separate optimizers)
    7. Top-down feedback generated → dispatched per client
    8. Each client trains locally using task loss + feedback alignment
    9. Evaluate on global test set
"""

import os
import copy
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List
import torch.nn.functional as F

from fl.client.client_factory import build_all_clients
from fl.server.adapter import build_adapter
from fl.server.apex import build_apex
from fl.server.feedback import build_feedback_dispatcher
from fl.data.dirichlet import dirichlet_partition
from fl.data.loaders import build_client_loaders, build_global_test_loader
from fl.client.base_client import fine_to_coarse


class FLRunner:
    """
    Feedback-driven FL training loop.

    Args:
        cfg    : full parsed config dict
        device : compute device
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        fl_cfg   = cfg["fl"]
        data_cfg = cfg["data"]
        server_cfg = cfg["server"]

        self.num_clients     = fl_cfg["num_clients"]
        self.num_rounds      = fl_cfg["num_rounds"]
        self.client_fraction = fl_cfg["client_fraction"]
        self.seed            = fl_cfg.get("seed", 42)
        self.arch_dir        = fl_cfg["arch_dir"]
        self.data_root       = data_cfg["root"]
        self.batch_size      = data_cfg["batch_size"]
        self.alpha           = data_cfg["dirichlet_alpha"]

        # ── Data ──────────────────────────────────────────────────────────
        from torchvision.datasets import CIFAR100
        import torchvision.transforms as T
        _ds = CIFAR100(
            root=self.data_root, train=True,
            download=True, transform=T.ToTensor()
        )
        targets   = np.array(_ds.targets)
        partition = dirichlet_partition(
            targets, self.num_clients, self.alpha, seed=self.seed
        )
        self.client_loaders = build_client_loaders(
            partition, self.data_root, self.batch_size
        )
        self.client_sizes = {
            cid: len(idxs) for cid, idxs in partition.items()
        }
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
        self.adapter    = build_adapter(cfg).to(device)
        self.apex       = build_apex(cfg).to(device)
        self.dispatcher = build_feedback_dispatcher(cfg)

        # Two separate optimizers — different lr for adapter vs apex
        self.adapter_optimizer = optim.Adam(
            self.adapter.parameters(),
            lr = server_cfg["adapter_lr"],
        )
        self.apex_optimizer = optim.Adam(
            self.apex.parameters(),
            lr = server_cfg["apex_lr"],
        )

        self.history: List[dict] = []

    # =========================================================================
    # Main Loop
    # =========================================================================

    def _client_warmup(self, warmup_epochs: int = 1):
        """
        Pre-train all clients locally before any server interaction.
        Gives local models enough signal to produce meaningful
        representations before Z1 attempts cross-client alignment.
        """
        print(f"\nClient warmup — {warmup_epochs} epoch(s) on local data...")
        for cid, client in self.clients.items():
            client.local_model.train()
            client.bottleneck.train()
            client.local_head.train()

            for _ in range(warmup_epochs):
                for batch in client.dataloader:
                    x, labels = batch
                    x      = x.to(self.device)
                    labels = labels.to(self.device)

                    client.optimizer.zero_grad()
                    z0 = client.local_model(x)
                    z0_compressed, kl_loss = client.bottleneck(z0)
                    logits = client.local_head(z0_compressed)
                    loss   = F.cross_entropy(logits, labels) + 0.01 * kl_loss
                    loss.backward()
                    client.optimizer.step()

            print(f"  Client {cid:02d} ({client.arch_name}) — warmup done")

        print("Warmup complete. Starting FL rounds.\n")

    def run(self):
        rng = np.random.default_rng(self.seed)
        self._client_warmup(warmup_epochs=cfg["fl"].get("warmup_epochs", 1))

        for round_idx in range(self.num_rounds):

            # ── 1. Sample clients ─────────────────────────────────────────
            n_selected = max(1, int(self.num_clients * self.client_fraction))
            selected   = rng.choice(
                self.num_clients, size=n_selected, replace=False
            ).tolist()

            # ── 2. Bottom-up: collect z0_compressed from each client ──────
            # Each client processes one batch from its local dataloader
            # We use iter() to get one batch per round, not exhaust the loader
            compressed_list = []
            label_list      = []
            coarse_list     = []

            for cid in selected:
                client = self.clients[cid]
                batch  = next(iter(client.dataloader))
                x, fine_labels = batch
                x           = x.to(self.device)
                fine_labels = fine_labels.to(self.device)
                coarse_labels = fine_to_coarse(fine_labels)

                # Client forward: local model + bottleneck
                # z0_compressed is the ONLY thing that leaves the client
                z0_compressed, _ = client.get_compressed(x)

                compressed_list.append(z0_compressed)   # (B, d1)
                label_list.append(fine_labels)           # (B,)
                coarse_list.append(coarse_labels)        # (B,)

            # Stack across clients → (N, B, d1)
            stacked_compressed = torch.stack(compressed_list, dim=0)
            stacked_labels     = torch.stack(label_list, dim=0)      # (N, B)
            stacked_coarse     = torch.stack(coarse_list, dim=0)     # (N, B)

            # ── 3 & 4. Server forward ─────────────────────────────────────
            self.adapter_optimizer.zero_grad()
            self.apex_optimizer.zero_grad()

            # Z1 adapter — cross-client Transformer
            z1, variance_loss, cls_loss = self.adapter(
                stacked_compressed,
                stacked_labels,
            )

            # Pool z1 across clients for Z2 input
            z1_pooled     = z1.mean(dim=0)               # (B, d1)
            coarse_labels_global = stacked_coarse[0]     # (B,) use first client

            # Z2 apex
            z2, coarse_logits, coarse_loss = self.apex(
                z1_pooled,
                coarse_labels_global,
            )

            # ── 5. Server backward ────────────────────────────────────────
            adapter_loss = self.adapter.total_loss(variance_loss, cls_loss)
            server_loss  = adapter_loss + coarse_loss

            server_loss.backward(retain_graph=True)
            self.adapter_optimizer.step()
            self.apex_optimizer.step()

            # ── 6. Top-down feedback ──────────────────────────────────────
            with torch.no_grad():
                # Z2 → global signal (B, d1)
                global_signal = self.apex.downward_message(z2)

                # Z1 → per-client signals (N, B, d1)
                client_z1s = self.adapter.downward_message(z1)

                # Dispatcher → personalized feedback per client (N, B, d1)
                feedback = self.dispatcher.dispatch(
                    global_signal = global_signal,
                    client_z1s    = client_z1s,
                    client_ids    = selected,
                )

            # Push feedback to each selected client
            for i, cid in enumerate(selected):
                client_feedback = self.dispatcher.get_client_feedback(
                    feedback, client_idx=i
                )
                self.clients[cid].receive_feedback(client_feedback)

            # ── 7. Client local training ──────────────────────────────────
            round_metrics = []
            for cid in selected:
                metrics = self.clients[cid].train_round(
                    global_round=round_idx
                )
                round_metrics.append(metrics)

            # ── 8. Evaluate ───────────────────────────────────────────────
            acc = self.evaluate()

            log = {
                "round"          : round_idx,
                "top1_acc"       : acc,
                "server_loss"    : server_loss.item(),
                "variance_loss"  : variance_loss.item(),
                "cls_loss"       : cls_loss.item(),
                "coarse_loss"    : coarse_loss.item(),
                "client_logs"    : round_metrics,
                "selected_clients": selected,
            }
            self.history.append(log)

            print(
                f"Round {round_idx:03d} | "
                f"Top-1: {acc:.4f} | "
                f"Server: {server_loss.item():.4f} "
                f"(var={variance_loss.item():.3f} "
                f"cls={cls_loss.item():.3f} "
                f"coarse={coarse_loss.item():.3f}) | "
                f"Clients: {selected}"
            )

        return self.history

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Global test accuracy.

        For evaluation we need client representations — we use all clients
        and pool their compressed representations through the server.
        Since we can't run all clients on the test set (they have private
        local models), we evaluate using a single representative client
        per round, rotating through all clients.

        Specifically: evaluate using client 0's local model + bottleneck
        as a fixed probe, passed through the server Z1+Z2 pipeline.
        This gives a stable, comparable metric across rounds.
        """
        self.adapter.eval()
        self.apex.eval()
        self.clients[0].local_model.eval()
        self.clients[0].bottleneck.eval()

        correct = total = 0

        for x, fine_labels in self.test_loader:
            x           = x.to(self.device)
            fine_labels = fine_labels.to(self.device)

            # Client 0 compresses
            z0 = self.clients[0].local_model(x)
            z0_compressed, _ = self.clients[0].bottleneck(z0)

            # Expand to (N=1, B, d1) for adapter
            stacked = z0_compressed.unsqueeze(0)             # (1, B, d1)
            labels  = fine_labels.unsqueeze(0)               # (1, B)

            # Server forward
            z1, _, _ = self.adapter(stacked, labels)
            z1_pooled = z1.mean(dim=0)                       # (B, d1)

            # Classification via adapter's cls_head
            logits = self.adapter.cls_head(z1_pooled)        # (B, num_classes)
            preds  = logits.argmax(dim=1)

            correct += (preds == fine_labels).sum().item()
            total   += fine_labels.size(0)

        self.adapter.train()
        self.apex.train()

        return correct / total if total > 0 else 0.0