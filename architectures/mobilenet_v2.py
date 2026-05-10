"""
architectures/mobilenet_v2.py

MobileNetV2 — pretrained, last block unfrozen
----------------------------------------------
Backbone mostly frozen for efficiency.
Last inverted residual block (features[17], features[18]) unfrozen
so the model can adapt MobileNet's ImageNet representations
toward CIFAR-100 structure.
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
        self.backbone = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last two blocks — enough to adapt to CIFAR-100
        for param in self.backbone[17].parameters():
            param.requires_grad = True
        for param in self.backbone[18].parameters():
            param.requires_grad = True

        # Trainable projection
        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.pool(z).flatten(1)                 # (B, 1280)
        return self.proj(z)                         # (B, 256)