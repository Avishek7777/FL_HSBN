"""
architectures/efficientnet_b0.py

EfficientNet-B0 — pretrained, last block unfrozen
--------------------------------------------------
Backbone mostly frozen. Last two feature blocks unfrozen
for CIFAR-100 adaptation.
Output: (B, 512)
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class LocalModel(nn.Module):
    out_dim: int = 512

    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.backbone = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last two blocks
        for param in self.backbone[7].parameters():
            param.requires_grad = True
        for param in self.backbone[8].parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        z = self.backbone(x)
        z = self.pool(z).flatten(1)                 # (B, 1280)
        return self.proj(z)                         # (B, 512)