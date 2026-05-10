"""
architectures/resnet18.py

ResNet18 — pretrained, last block unfrozen
-------------------------------------------
Backbone mostly frozen. layer4 unfrozen for CIFAR-100 adaptation.
ResNet18 is the standard FL baseline backbone — keeping it mostly
frozen preserves the comparison point while allowing adaptation.
Output: (B, 512)
"""

import torch
import torch.nn as nn
from torchvision import models


class LocalModel(nn.Module):
    out_dim: int = 512

    def __init__(self):
        super().__init__()

        backbone        = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        # Remove final FC
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze all first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4 (index 7 in the sequential)
        for param in self.backbone[7].parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x).flatten(1)             # (B, 512)
        return self.proj(z)                         # (B, 512)