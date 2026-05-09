"""
architectures/small_cnn.py

Small CNN — from scratch
------------------------
3 conv layers with progressive channel expansion.
Lightweight, simulates a low-resource client.
Output: (B, 256) — must exceed d1=128 for bottleneck compression.
"""

import torch
import torch.nn as nn


class LocalModel(nn.Module):
    out_dim: int = 256

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),           # 8x8 -> 1x1
        )
        self.proj = nn.Linear(128, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.flatten(1)
        return self.proj(z)                         # (B, 256)