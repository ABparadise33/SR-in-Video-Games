from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class InceptionDWConv(nn.Module):
    """Inception-style depthwise projection used by the RBSFormer paper."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        splits = [channels // 4] * 4
        splits[-1] += channels - sum(splits)
        self.splits = splits
        self.dw3 = nn.Conv2d(splits[0], splits[0], 3, 1, 1, groups=splits[0])
        self.dw1x11 = nn.Conv2d(splits[1], splits[1], (1, 11), 1, (0, 5), groups=splits[1])
        self.dw11x1 = nn.Conv2d(splits[2], splits[2], (11, 1), 1, (5, 0), groups=splits[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3 = torch.split(x, self.splits, dim=1)
        return torch.cat([self.dw3(x0), self.dw1x11(x1), self.dw11x1(x2), x3], dim=1)


class Project(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(channels, channels, 1), InceptionDWConv(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EXCA(nn.Module):
    """Enhanced cross-covariance attention adapted from RAW RBSFormer to RGB SR."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm = LayerNorm2d(channels)
        self.q = Project(channels)
        self.k = Project(channels)
        self.v = Project(channels)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        q = self.q(y).reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = self.k(y).reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v(y).reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).reshape(b, c, h, w)
        return x + self.out(out)


class EGFN(nn.Module):
    """Enhanced gated feed-forward network."""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, 1)
        self.dw = InceptionDWConv(hidden * 2)
        self.out_proj = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(self.in_proj(self.norm(x)))
        y1, y2 = y.chunk(2, dim=1)
        y = F.gelu(y1) * y2
        return x + self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, ffn_expansion: float) -> None:
        super().__init__()
        self.attn = EXCA(channels, num_heads)
        self.ffn = EGFN(channels, ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))


class RBSFormerRGB(nn.Module):
    """RGB x4 adaptation of RBSFormer for the Kaggle game SR task."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 60,
        num_blocks: int = 8,
        num_heads: int = 6,
        ffn_expansion: float = 2.0,
        scale: int = 4,
        residual_upsample: bool = True,
    ) -> None:
        super().__init__()
        if scale not in {2, 3, 4}:
            raise ValueError("RBSFormerRGB currently supports scale 2, 3, or 4.")
        self.scale = scale
        self.residual_upsample = residual_upsample
        self.conv_first = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.body = nn.Sequential(
            *[TransformerBlock(channels, num_heads, ffn_expansion) for _ in range(num_blocks)]
        )
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, out_channels * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.conv_first(x)
        deep = self.conv_body(self.body(shallow))
        out = self.upsample(shallow + deep)
        if self.residual_upsample:
            base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
            out = out + base
        return out
