"""
architectures/resnet18.py

ResNet18 — pretrained, frozen backbone
----------------------------------------
Standard FL baseline backbone. ResNet18 is the most commonly used
pretrained model in FL literature — including it grounds comparisons
with prior work. Backbone frozen, only projection trains.
Output: (B, 512)
"""

import torch
import torch.nn as nn
from torchvision import models


class LocalModel(nn.Module):
    out_dim: int = 512

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        # Remove final FC layer — use everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ResNet18 outputs 512-dim after avgpool
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.backbone(x).flatten(1)         # (B, 512)
        return self.proj(z)                         # (B, 512)