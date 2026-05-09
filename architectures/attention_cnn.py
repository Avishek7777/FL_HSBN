"""
architectures/attention_cnn.py

Attention CNN — from scratch
------------------------------
CNN backbone with a single self-attention layer injected before pooling.
Hybrid architecture — tests whether attention-augmented representations
have different compression characteristics through the bottleneck.
Output: (B, 512)
"""

import torch
import torch.nn as nn


class SelfAttention2d(nn.Module):
    """
    Lightweight spatial self-attention over CNN feature maps.
    Flattens spatial dims, applies multi-head attention, restores shape.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = num_heads,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims → sequence
        seq = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        attn_out, _ = self.attn(seq, seq, seq)
        out = self.norm(seq + attn_out)              # residual + norm
        # Restore spatial shape
        return out.transpose(1, 2).reshape(B, C, H, W)


class LocalModel(nn.Module):
    out_dim: int = 512

    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 16x16 -> 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                        # 8x8 -> 4x4

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
                                                    # 4x4 feature map fed to attention
        )
        self.attention = SelfAttention2d(channels=512, num_heads=4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv_stem(x)
        z = self.attention(z)
        z = self.pool(z)
        z = z.flatten(1)
        return self.proj(z)                         # (B, 512)