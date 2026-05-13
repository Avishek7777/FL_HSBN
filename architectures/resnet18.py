"""
architectures/resnet18.py

ResNet18 — pretrained, dataset-aware
--------------------------------------
Learned channel adapter for grayscale datasets.
layer4 unfrozen for dataset adaptation.
Output: (B, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LocalModel(nn.Module):
    out_dim: int = 512

    def __init__(self, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.in_channels = in_channels

        # Learned channel adapter for grayscale → RGB
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.channel_adapter = None

        backbone        = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze all
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.backbone[7].parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        if x.shape[-1] < 32:
            x = F.interpolate(
                x, size=(32, 32),
                mode='bilinear', align_corners=False
            )

        z = self.backbone(x).flatten(1)             # (B, 512)
        return self.proj(z)