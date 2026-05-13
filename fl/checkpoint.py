"""
fl/checkpoint.py

Checkpoint Manager
===================
Saves and loads complete run state at the end of each dataset x alpha run.

Saved per run:
    Server components:
        server_adapter.pt
        server_classifier.pt
        server_apex.pt
        server_encoder.pt

    Client components (per client):
        clients/client_{id:02d}_model.pt
        clients/client_{id:02d}_bottleneck.pt
        clients/client_{id:02d}_head.pt

    Metadata:
        metadata.json — config, arch assignments, final accuracy,
                        training history summary

Structure:
    results/checkpoints/{dataset}_{experiment}/
        ├── server_adapter.pt
        ├── server_classifier.pt
        ├── server_apex.pt
        ├── server_encoder.pt
        ├── clients/
        │   ├── client_00_model.pt
        │   ├── client_00_bottleneck.pt
        │   ├── client_00_head.pt
        │   └── ...
        └── metadata.json

Transfer evaluation:
    Load server components from one checkpoint.
    Run evaluate() on a different dataset's test loader.
    No fine-tuning — pure transfer accuracy.
"""

import os
import json
import torch
from typing import Dict


class CheckpointManager:
    """
    Manages saving and loading of complete FL run state.

    Args:
        base_dir    : root directory for all checkpoints
        experiment  : unique name for this run e.g. 'cifar100_alpha01'
    """

    def __init__(self, base_dir: str, experiment: str):
        self.experiment  = experiment
        self.checkpoint_dir = os.path.join(
            base_dir, "checkpoints", experiment
        )
        self.clients_dir = os.path.join(self.checkpoint_dir, "clients")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.clients_dir, exist_ok=True)

    # =========================================================================
    # Save
    # =========================================================================

    def save(
        self,
        adapter,
        classifier,
        apex,
        server_encoder,
        clients       : Dict,
        history       : list,
        cfg           : dict,
    ):
        """
        Save complete run state at end of training.

        Args:
            adapter        : Z1 adapter module
            classifier     : Z1.5 classifier module
            apex           : Z2 apex module
            server_encoder : server encoder module
            clients        : dict {client_id: FLClient}
            history        : full training history list
            cfg            : experiment config dict
        """
        print(f"\nSaving checkpoint: {self.experiment}")

        # ── Server components ─────────────────────────────────────────────
        torch.save(
            adapter.state_dict(),
            os.path.join(self.checkpoint_dir, "server_adapter.pt")
        )
        torch.save(
            classifier.state_dict(),
            os.path.join(self.checkpoint_dir, "server_classifier.pt")
        )
        torch.save(
            apex.state_dict(),
            os.path.join(self.checkpoint_dir, "server_apex.pt")
        )
        torch.save(
            server_encoder.state_dict(),
            os.path.join(self.checkpoint_dir, "server_encoder.pt")
        )

        # ── Client components ─────────────────────────────────────────────
        arch_assignments = {}
        for cid, client in clients.items():
            torch.save(
                client.local_model.state_dict(),
                os.path.join(self.clients_dir, f"client_{cid:02d}_model.pt")
            )
            torch.save(
                client.bottleneck.state_dict(),
                os.path.join(self.clients_dir, f"client_{cid:02d}_bottleneck.pt")
            )
            torch.save(
                client.local_head.state_dict(),
                os.path.join(self.clients_dir, f"client_{cid:02d}_head.pt")
            )
            arch_assignments[cid] = {
                "arch"   : client.arch_name,
                "out_dim": client.local_model.out_dim,
            }

        # ── Metadata ──────────────────────────────────────────────────────
        accs     = [r["top1_acc"] for r in history]
        metadata = {
            "experiment"       : self.experiment,
            "dataset"          : cfg.get("data", {}).get("name", "unknown"),
            "dirichlet_alpha"  : cfg["data"]["dirichlet_alpha"],
            "num_rounds"       : cfg["fl"]["num_rounds"],
            "num_clients"      : cfg["fl"]["num_clients"],
            "best_accuracy"    : max(accs),
            "best_round"       : accs.index(max(accs)),
            "final_accuracy"   : accs[-1],
            "arch_assignments" : arch_assignments,
            "common"           : cfg["common"],
            "channel"          : cfg["channel"],
        }

        with open(os.path.join(self.checkpoint_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to: {self.checkpoint_dir}")
        print(f"  Best accuracy: {metadata['best_accuracy']:.4f} "
              f"(round {metadata['best_round']})")
        print(f"  Final accuracy: {metadata['final_accuracy']:.4f}")

    # =========================================================================
    # Load server components
    # =========================================================================

    def load_server(
        self,
        adapter,
        classifier,
        apex,
        server_encoder,
        device: str = "cpu",
    ):
        """
        Load server component weights from checkpoint.
        Used for transfer evaluation — load trained server,
        evaluate on a different dataset's test loader.

        Args:
            adapter        : Z1 adapter module (instantiated, empty weights)
            classifier     : Z1.5 classifier module
            apex           : Z2 apex module
            server_encoder : server encoder module
            device         : compute device
        """
        adapter.load_state_dict(torch.load(
            os.path.join(self.checkpoint_dir, "server_adapter.pt"),
            map_location=device,
        ))
        classifier.load_state_dict(torch.load(
            os.path.join(self.checkpoint_dir, "server_classifier.pt"),
            map_location=device,
        ))
        apex.load_state_dict(torch.load(
            os.path.join(self.checkpoint_dir, "server_apex.pt"),
            map_location=device,
        ))
        server_encoder.load_state_dict(torch.load(
            os.path.join(self.checkpoint_dir, "server_encoder.pt"),
            map_location=device,
        ))
        print(f"Loaded server checkpoint: {self.experiment}")

    # =========================================================================
    # Metadata
    # =========================================================================

    def load_metadata(self) -> dict:
        path = os.path.join(self.checkpoint_dir, "metadata.json")
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def list_checkpoints(base_dir: str) -> list:
        """List all available checkpoints."""
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            return []
        return sorted(os.listdir(ckpt_dir))


def build_checkpoint_manager(cfg: dict) -> CheckpointManager:
    """
    Build a CheckpointManager from config.
    Experiment name is derived from dataset name + alpha value.
    """
    dataset  = cfg.get("data", {}).get("name", "dataset")
    alpha    = cfg["data"]["dirichlet_alpha"]
    alpha_str = str(alpha).replace(".", "")
    experiment = f"{dataset}_alpha{alpha_str}"

    return CheckpointManager(
        base_dir   = cfg.get("results_dir", "results"),
        experiment = experiment,
    )