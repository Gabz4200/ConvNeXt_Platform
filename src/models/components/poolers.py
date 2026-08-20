"""Pooling components for 2D spatial feature maps."""

import math
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


class AdaptiveLearnedPool2d(nn.Module):
    """Learned adaptive pooling of 2D feature maps to a fixed spatial ``output_size``.

    Applies mobilenet-style depthwise-separable convolutions, downsamples with the
    strided ``downsampling_core`` until the target size is reached, and mixes the
    result with the global-average-pooled input before a final convolution. The
    output spatial dimensions always equal ``output_size`` regardless of input size.
    """

    max_pad_ratio: float = 3.0

    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        out_features: int,
        output_size: tuple[int, int],
    ) -> None:
        super().__init__()
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError(
                f"output_size dimensions must be positive, got {output_size}."
            )
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.out_features = out_features
        self.output_size = output_size

        # Calculate the minimal divisor for kernel_size and stride
        self.kernel_size, self.stride = self._get_downsample_params(output_size)

        # Mobilenet style convs for efficient execution.
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features,
                kernel_size=3,
                padding="same",
                groups=in_features,
            ),
            nn.Conv2d(
                in_channels=in_features,
                out_channels=intermediate_features,
                kernel_size=1,
                padding="same",
                groups=1,
            ),
            nn.GELU(approximate="tanh"),
        )

        self.downsampling_core = nn.Sequential(
            nn.Conv2d(
                in_channels=intermediate_features,
                out_channels=intermediate_features,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0,
                groups=intermediate_features,
            ),
            nn.Conv2d(
                in_channels=intermediate_features,
                out_channels=intermediate_features,
                kernel_size=1,
                padding="same",
                groups=1,
            ),
            nn.GELU(approximate="tanh"),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=intermediate_features + in_features,
                out_channels=intermediate_features,
                kernel_size=3,
                padding="same",
            ),
            nn.Conv2d(
                in_channels=intermediate_features,
                out_channels=out_features,
                kernel_size=1,
                padding="same",
                groups=1,
            ),
        )

    def _smallest_prime_factor(self, n: int) -> int:
        """Returns the minimal divisor > 1 (smallest prime factor) of n."""
        if n <= 1:
            return 1
        if n % 2 == 0:
            return 2
        # Check odd numbers up to the square root of n
        for i in range(3, math.isqrt(n) + 1, 2):
            if n % i == 0:
                return i
        return n

    def _get_downsample_params(
        self,
        output_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Parse ``output_size`` and return the appropriate kernel_size and stride.

        Uses the smallest prime factor of each target dimension so that repeated
        strided convolutions can reduce the spatial size down to it exactly.

        :param output_size: `(output_h, output_w)` target spatial size.
        :return: `(kernel_size, stride)` tuples `(k_h, k_w)` for both.
        """
        k_h = self._smallest_prime_factor(output_size[0])
        k_w = self._smallest_prime_factor(output_size[1])

        return (k_h, k_w), (k_h, k_w)

    def _pad_to_nearest_multiple(
        self, x: torch.Tensor, multiple: tuple[int, int]
    ) -> torch.Tensor:
        """Pad the input tensor to the next multiple in the height and width dimensions.

        Padding is applied symmetrically on both sides of each dimension (an extra pixel
        goes to the bottom/right when the total padding is odd). If the spatial dimensions
        are already multiples of ``multiple``, the input is returned unchanged.

        :param x: Input feature tensor of shape `(B, C, H, W)`.
        :param multiple: `(multiple_h, multiple_w)` to pad the spatial dimensions to.
        :return: Padded tensor of shape `(B, C, H', W')`.
        """
        height, width = x.shape[-2:]
        pad_h = (multiple[0] - height % multiple[0]) % multiple[0]
        pad_w = (multiple[1] - width % multiple[1]) % multiple[1]

        if pad_h == 0 and pad_w == 0:
            return x

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return F.pad(x, (left, right, top, bottom))

    def _num_downsamples(self, size: int, target: int, stride: int) -> int:
        """Return how many strided downsampling applications are needed to reach ``target``.

        :param size: Current spatial dimension.
        :param target: Target spatial dimension.
        :param stride: Downsampling stride for the dimension.
        :return: Number of ``downsampling_core`` applications required.
        """
        if stride <= 1:
            return 0

        num = 0
        while target * stride**num < size:
            num += 1
        return num

    def _bounded_num_downsamples(self, height: int, width: int) -> int:
        """Return the number of core applications bounded by the allowed padding ratio.

        The stride factor of each target dimension is its smallest prime factor, so the
        exact multiple needed to land on ``output_size`` can exceed the input size by a
        large factor (e.g. 56x56 -> 7x7 requires padding to 343). The count is reduced
        until padding each dimension stays within ``max_pad_ratio`` of its input size;
        the final adaptive pool absorbs the residual reduction.

        :param height: Input height.
        :param width: Input width.
        :return: Number of ``downsampling_core`` applications to run.
        """
        target_h, target_w = self.output_size
        k_h, k_w = self.kernel_size

        num = max(
            self._num_downsamples(height, target_h, k_h),
            self._num_downsamples(width, target_w, k_w),
        )
        while num > 0:
            m_h = target_h * k_h**num
            m_w = target_w * k_w**num
            pad_h = (m_h - height % m_h) % m_h
            pad_w = (m_w - width % m_w) % m_w
            if (
                height + pad_h <= self.max_pad_ratio * height
                and width + pad_w <= self.max_pad_ratio * width
            ):
                break
            num -= 1
        return num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample the input to ``output_size`` and mix it with pooled input features.

        The input is padded up to ``output_size * stride**N`` so that ``N`` applications of
        the strided ``downsampling_core`` bring it to ``output_size``, with ``N`` reduced when
        that would exceed ``max_pad_ratio``; a final adaptive pool guarantees the exact target
        size before the residual input features are fused.

        :param x: Input feature tensor of shape `(B, C, H, W)`.
        :return: Pooled tensor of shape `(B, out_features, output_h, output_w)`.
        """
        target_h, target_w = self.output_size
        k_h, k_w = self.kernel_size

        # Number of strided downsampling applications (shared by both dimensions).
        num_downsamples = self._bounded_num_downsamples(x.shape[-2], x.shape[-1])

        # Pad so repeated strided downsampling lands exactly on the target size.
        x = self._pad_to_nearest_multiple(
            x,
            (
                target_h * k_h**num_downsamples,
                target_w * k_w**num_downsamples,
            ),
        )

        input_features = self.input_conv(x)
        input_avg = F.adaptive_avg_pool2d(x, self.output_size)

        # For loop N times the downsample_core until the target size is reached.
        downsampled = input_features
        for _ in range(num_downsamples):
            downsampled = self.downsampling_core(downsampled)

        # Guarantee exact target spatial size before concatenation.
        downsampled = F.adaptive_avg_pool2d(downsampled, self.output_size)

        # Concatenate the input_avg with the downsampled features along the channel dim.
        pooled = torch.cat([input_avg, downsampled], dim=1)

        return self.output_conv(pooled)
