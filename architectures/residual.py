"""
architectures/residual.py

Residual CNN — from scratch
----------------------------
ResNet-style skip connections without the full ResNet weight budget.
Tests whether skip connection representations compress differently
through the bottleneck compared to plain CNNs.
Output: (B, 256)
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class LocalModel(nn.Module):
    out_dim: int = 256

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 32x32 -> 16x16
        )
        self.layer1 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.MaxPool2d(2),                        # 16x16 -> 8x8
        )
        self.layer2 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),           # 8x8 -> 1x1
        )
        self.proj = nn.Linear(256, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = z.flatten(1)
        return self.proj(z)                         # (B, 256)