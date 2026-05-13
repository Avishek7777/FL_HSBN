"""
architectures/mobilenet_v2.py

MobileNetV2 — pretrained, dataset-aware
-----------------------------------------
Learned channel adapter for grayscale datasets (FMNIST).
Upsampling to 224x224 for pretrained backbone compatibility.
Last two blocks unfrozen for dataset adaptation.
Output: (B, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LocalModel(nn.Module):
    out_dim: int = 256

    def __init__(self, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.input_size  = input_size

        # Learned channel adapter for grayscale → RGB (Option B)
        # Identity for RGB inputs (in_channels=3)
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.channel_adapter = None

        backbone      = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        self.backbone = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last two blocks
        for param in self.backbone[17].parameters():
            param.requires_grad = True
        for param in self.backbone[18].parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel adaptation (grayscale → RGB if needed)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        # Upsample to 224x224 for pretrained backbone
        if x.shape[-1] != 224:
            x = F.interpolate(
                x, size=(224, 224),
                mode='bilinear', align_corners=False
            )

        z = self.backbone(x)
        z = self.pool(z).flatten(1)                 # (B, 1280)
        return self.proj(z)                         # (B, 256)