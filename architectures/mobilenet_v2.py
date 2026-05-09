"""
architectures/mobilenet_v2.py

MobileNetV2 — pretrained, frozen backbone
------------------------------------------
Lightweight pretrained model, simulates a resource-constrained client
that has a pretrained model but limited compute for fine-tuning.
Backbone weights frozen — only projection layer trains.
Output: (B, 256)
"""

import torch
import torch.nn as nn
from torchvision import models


class LocalModel(nn.Module):
    out_dim: int = 256

    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        # Remove classifier head — keep feature extractor only
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable projection into common out_dim
        # MobileNetV2 features output 1280 channels
        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.backbone(x)                    # (B, 1280, H, W)
            z = self.pool(z).flatten(1)             # (B, 1280)
        return self.proj(z)                         # (B, 256)