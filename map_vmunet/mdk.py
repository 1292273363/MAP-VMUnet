"""Dynamic multi-kernel encoder enhancement for BHWC features.

The depthwise branches and channel shuffle draw on MK-UNet (BSD-3-Clause).
MAP-VMUnet adds sample-dependent branch weighting and high-frequency
calibration. See THIRD_PARTY_NOTICES.md.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelShuffle(nn.Module):
    """Shuffle the channels of a BCHW feature map."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        if channels <= 0 or groups <= 0:
            raise ValueError("channels and groups must be positive")
        self.groups = math.gcd(channels, min(channels, groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return x
        batch, channels, height, width = x.shape
        x = x.reshape(batch, self.groups, channels // self.groups, height, width)
        return x.transpose(1, 2).contiguous().reshape(batch, channels, height, width)


class HighFrequencyBoundaryCue(nn.Module):
    """Predict the local-detail gate for a BCHW feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_frequency = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return torch.sigmoid(self.detail(high_frequency))


class DynamicMultiKernelBlock(nn.Module):
    """Local multi-scale compensation block for a VM-UNet encoder stage.

    The block expands channels, evaluates depth-wise kernel branches, selects
    them with sample-wise GAP+GMP weights, shuffles channels, and projects back
    to the input width. A high-frequency gate calibrates the projected feature.
    The final learnable residual scale is deliberately initialised near zero so
    pretrained VMamba feature statistics are only perturbed gradually.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        kernels: Sequence[int] = (1, 3, 5),
        use_dilated_kernel: bool = False,
        dilated_kernel_dilation: int = 2,
        shuffle_groups: int = 4,
        init_gamma: float = 1e-3,
        selection_reduction: int = 4,
    ):
        super().__init__()
        if channels <= 0 or expansion <= 0:
            raise ValueError("channels and expansion must be positive")
        if selection_reduction <= 0:
            raise ValueError("selection_reduction must be positive")
        if not kernels:
            raise ValueError("kernels must contain at least one odd kernel size")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("All MDK kernel sizes must be positive odd integers")

        self.expanded_channels = channels * expansion

        self.expand = nn.Sequential(
            nn.Conv2d(channels, self.expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expanded_channels),
            nn.GELU(),
        )

        branch_specs = [(kernel, 1) for kernel in kernels]
        if use_dilated_kernel:
            if dilated_kernel_dilation <= 1:
                raise ValueError("dilated_kernel_dilation must be greater than one")
            branch_specs.append((3, int(dilated_kernel_dilation)))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.expanded_channels,
                        self.expanded_channels,
                        kernel_size=kernel,
                        padding=dilation * (kernel // 2),
                        dilation=dilation,
                        groups=self.expanded_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expanded_channels),
                    nn.GELU(),
                )
                for kernel, dilation in branch_specs
            ]
        )

        hidden_channels = max(
            len(branch_specs), self.expanded_channels // selection_reduction
        )
        self.kernel_selector = nn.Sequential(
            nn.Conv2d(self.expanded_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, len(branch_specs), kernel_size=1),
        )
        self.shuffle = ChannelShuffle(
            self.expanded_channels,
            groups=shuffle_groups,
        )
        self.project = nn.Sequential(
            nn.Conv2d(self.expanded_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.boundary_cue = HighFrequencyBoundaryCue(channels)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        residual = x

        expanded = self.expand(x)
        pooled = F.adaptive_avg_pool2d(expanded, 1)
        pooled = pooled + F.adaptive_max_pool2d(expanded, 1)
        alpha = torch.softmax(self.kernel_selector(pooled), dim=1)

        branch_features = torch.stack(
            [branch(expanded) for branch in self.branches], dim=1
        )
        fused = torch.sum(branch_features * alpha.unsqueeze(2), dim=1)
        fused = self.project(self.shuffle(fused))

        boundary_gate = self.boundary_cue(residual)
        fused = fused + fused * boundary_gate
        output = residual + self.gamma * fused
        return output.permute(0, 2, 3, 1).contiguous()
