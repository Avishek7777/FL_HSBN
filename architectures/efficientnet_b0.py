"""
architectures/efficientnet_b0.py

EfficientNet-B0 — pretrained, frozen backbone
----------------------------------------------
Strong mid-tier pretrained model. Compound scaling makes its
representations structurally different from plain CNNs —
good stress test for the bottleneck alignment.
Backbone weights frozen — only projection layer trains.
Output: (B, 512)
"""

import torch
import torch.nn as nn
from torchvision import models


class LocalModel(nn.Module):
    out_dim: int = 512

    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Keep feature extractor, drop classifier
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # EfficientNet-B0 features output 1280 channels
        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.backbone(x)                    # (B, 1280, H, W)
            z = self.pool(z).flatten(1)             # (B, 1280)
        return self.proj(z)                         # (B, 512)