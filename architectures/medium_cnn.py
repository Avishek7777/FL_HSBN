"""
architectures/medium_cnn.py

Medium CNN — from scratch, dataset-aware
-----------------------------------------
Output: (B, 256)
"""

import torch
import torch.nn as nn


class LocalModel(nn.Module):
    out_dim: int = 256

    def __init__(self, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.3)
        self.proj    = nn.Linear(256, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x).flatten(1)
        z = self.dropout(z)
        return self.proj(z)