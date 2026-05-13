"""
architectures/residual.py

Residual CNN — from scratch, dataset-aware
-------------------------------------------
Output: (B, 256)
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class LocalModel(nn.Module):
    out_dim: int = 256

    def __init__(self, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        return self.proj(z.flatten(1))