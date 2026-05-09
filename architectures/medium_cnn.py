"""
architectures/medium_cnn.py

Medium CNN — from scratch
--------------------------
4 conv layers, wider channels, dropout regularization.
Mid-resource client.
Output: (B, 256)
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
            nn.MaxPool2d(2),                        # 8x8 -> 4x4

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),           # 4x4 -> 1x1
        )
        self.dropout = nn.Dropout(0.3)
        self.proj = nn.Linear(256, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.flatten(1)
        z = self.dropout(z)
        return self.proj(z)                         # (B, 256)