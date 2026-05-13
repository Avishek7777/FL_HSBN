"""
architectures/attention_cnn.py

Attention CNN — from scratch, dataset-aware
--------------------------------------------
Output: (B, 512)
"""

import torch
import torch.nn as nn


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq      = x.flatten(2).transpose(1, 2)
        out, _   = self.attn(seq, seq, seq)
        out      = self.norm(seq + out)
        return out.transpose(1, 2).reshape(B, C, H, W)


class LocalModel(nn.Module):
    out_dim: int = 512

    def __init__(self, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.attention = SelfAttention2d(512, num_heads=4)
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        self.proj      = nn.Linear(512, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv_stem(x)
        z = self.attention(z)
        z = self.pool(z).flatten(1)
        return self.proj(z)