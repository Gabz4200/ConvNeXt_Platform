"""Pooling components for 2D spatial feature maps."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LearnedWeightedGAP(nn.Module):
    """Learned Weighted Global Average Pooling (GAP) layer for 2D spatial feature maps.

    Computes spatial attention weights via a 2D convolution and performs weighted
    pooling over spatial dimensions. Optionally concatenates standard uniform GAP.

    :param in_features: Number of input channels.
    :param kernel_size: Kernel size for the spatial weighter convolution.
    :param num_output: Number of spatial attention maps to generate.
    :param concat_gap: Whether to append standard uniform GAP output to pooled features.
    """

    def __init__(
        self,
        in_features: int,
        kernel_size: int = 1,
        num_output: int = 1,
        concat_gap: bool = True,
    ) -> None:
        """Initialize `LearnedWeightedGAP` module."""
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer, got {kernel_size}.")

        self.in_features = in_features
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.concat_gap = concat_gap

        padding = kernel_size // 2

        # Produce `num_output` spatial attention maps.
        self.weighter_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=num_output,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform spatial weighted global average pooling on the input tensor.

        :param x: Input feature tensor of shape `(B, C, H, W)`.
        :return: Pooled feature tensor. If `num_output == 1` and `concat_gap == False`, shape
            is `(B, C)`. If `num_output == 1` and `concat_gap == True`, shape is `(B, 2C)`.
            If `num_output > 1`, shape is `(B, num_output, C)` or `(B, num_output + 1, C)`.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got tensor with shape {tuple(x.shape)}."
            )

        # Compute spatial attention maps normalized so each map sums to 1.
        logits = self.weighter_conv(x)
        weights = F.softmax(logits.flatten(start_dim=2), dim=-1).view_as(logits)

        # Aggregate spatial feature channels using attention weights.
        pooled = torch.einsum("bohw, bchw -> boc", weights, x)

        # Squeeze output map dimension when only 1 attention map is requested.
        if self.num_output == 1:
            pooled = pooled.squeeze(1)

        # Concatenate unweighted global average pooling features if requested.
        if self.concat_gap:
            gap = x.mean(dim=[2, 3])
            gap_target = gap if self.num_output == 1 else gap.unsqueeze(1)
            pooled = torch.cat([pooled, gap_target], dim=1)

        return pooled

    def get_config(self) -> dict[str, Any]:
        """Return module configuration dictionary.

        :return: Dictionary containing initialization parameters.
        """
        return {
            "in_features": self.in_features,
            "kernel_size": self.kernel_size,
            "num_output": self.num_output,
            "concat_gap": self.concat_gap,
        }

    def extra_repr(self) -> str:
        """Return extra representation string for logging.

        :return: Module string representation.
        """
        return (
            f"in_features={self.in_features}, "
            f"kernel_size={self.kernel_size}, "
            f"num_output={self.num_output}, "
            f"concat_gap={self.concat_gap}"
        )

