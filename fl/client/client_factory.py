"""
fl/client/client_factory.py

Client Factory
===============
Responsible for:
    1. Sampling an architecture from the architectures/ folder
    2. Reading its out_dim
    3. Building a per-client BandwidthBottleneck wired to that out_dim → d1
    4. Assembling and returning a complete FLClient

The architecture files know nothing about FL, bottlenecks, or d1.
The factory is the only place where those two worlds connect.

Architecture contract:
    Each file in architectures/ must expose:
        class LocalModel(nn.Module):
            out_dim: int = ...       # class attribute
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                ...                  # returns (B, out_dim)
"""

import os
import importlib.util
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from hsbn.channels.bottleneck import build_channel
from fl.client.base_client import FLClient


def load_architecture(
    arch_dir : str,
    arch_file: str,
) -> nn.Module:
    """
    Dynamically import a LocalModel from an architecture file.

    Args:
        arch_dir  : path to architectures/ folder
        arch_file : filename (e.g. 'small_cnn.py')

    Returns:
        instantiated LocalModel
    """
    path = os.path.join(arch_dir, arch_file)
    spec = importlib.util.spec_from_file_location("local_arch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "LocalModel"):
        raise AttributeError(
            f"{arch_file} must define a class named 'LocalModel'"
        )
    if not hasattr(module.LocalModel, "out_dim"):
        raise AttributeError(
            f"LocalModel in {arch_file} must define class attribute 'out_dim'"
        )

    return module.LocalModel()


def sample_architecture(
    arch_dir: str,
    seed    : int,
) -> str:
    """
    Randomly select an architecture filename from arch_dir.

    Args:
        arch_dir : path to architectures/ folder
        seed     : for reproducible selection

    Returns:
        selected filename (e.g. 'resnet18.py')
    """
    files = sorted([
        f for f in os.listdir(arch_dir)
        if f.endswith(".py") and not f.startswith("_")
    ])
    if not files:
        raise RuntimeError(
            f"No architecture files found in {arch_dir}. "
            f"Add .py files with a LocalModel class."
        )

    rng = np.random.default_rng(seed)
    return rng.choice(files)


def build_client(
    client_id     : int,
    arch_dir      : str,
    dataloader    : DataLoader,
    d1            : int,
    channel_cfg   : dict,
    num_classes   : int,
    lr            : float,
    local_epochs  : int,
    alpha         : float,
    device        : str,
    arch_seed     : int,
) -> Tuple[FLClient, str]:
    """
    Build a complete FLClient.

    Steps:
        1. Sample architecture file using arch_seed
        2. Instantiate LocalModel — reads its out_dim
        3. Build BandwidthBottleneck: out_dim → d1
        4. Assemble FLClient

    Args:
        client_id   : unique client id
        arch_dir    : path to architectures/ folder
        dataloader  : client's local DataLoader
        d1          : common space dimension (server's input dim)
        channel_cfg : bottleneck channel config (beta, gamma)
        num_classes : local task num classes
        lr          : local learning rate
        local_epochs: local epochs per round
        alpha       : feedback alignment loss weight
        device      : compute device
        arch_seed   : seed for architecture sampling (client_id + global_seed)

    Returns:
        (FLClient, arch_name)
    """
    # 1. Sample architecture
    arch_file = sample_architecture(arch_dir, seed=arch_seed)

    # 2. Instantiate local model
    local_model = load_architecture(arch_dir, arch_file)
    in_dim      = local_model.out_dim                # variable per architecture

    # 3. Build per-client bottleneck: local out_dim → common d1
    bottleneck = build_channel(
        in_dim      = in_dim,
        out_dim     = d1,
        channel_cfg = channel_cfg,
    )

    # 4. Assemble client
    client = FLClient(
        client_id    = client_id,
        local_model  = local_model,
        bottleneck   = bottleneck,
        dataloader   = dataloader,
        arch_name    = arch_file.replace(".py", ""),
        num_classes  = num_classes,
        lr           = lr,
        local_epochs = local_epochs,
        device       = device,
        alpha        = alpha,
    )

    return client, arch_file


def build_all_clients(
    num_clients  : int,
    arch_dir     : str,
    dataloaders  : dict,
    d1           : int,
    channel_cfg  : dict,
    num_classes  : int,
    lr           : float,
    local_epochs : int,
    alpha        : float,
    device       : str,
    global_seed  : int,
) -> dict:
    """
    Build all clients. Each gets a randomly sampled architecture.
    Seed is deterministic per client: global_seed + client_id.

    Args:
        num_clients : total number of clients
        dataloaders : dict {client_id: DataLoader}
        ...rest     : passed through to build_client

    Returns:
        dict {client_id: FLClient}
    """
    clients = {}
    arch_log = {}

    for cid in range(num_clients):
        client, arch_name = build_client(
            client_id    = cid,
            arch_dir     = arch_dir,
            dataloader   = dataloaders[cid],
            d1           = d1,
            channel_cfg  = channel_cfg,
            num_classes  = num_classes,
            lr           = lr,
            local_epochs = local_epochs,
            alpha        = alpha,
            device       = device,
            arch_seed    = global_seed + cid,
        )
        clients[cid]  = client
        arch_log[cid] = arch_name

    # Print architecture assignment summary
    print("\nClient architecture assignments:")
    print("─" * 40)
    for cid, arch in arch_log.items():
        out_dim = clients[cid].local_model.out_dim
        print(f"  Client {cid:02d} │ {arch:<25} │ out_dim={out_dim}")
    print("─" * 40)

    return clients