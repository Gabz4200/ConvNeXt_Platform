r"""2D spatial pooling and unpooling components for deep convolutional networks."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _smallest_prime_factor(n: int) -> int:
    r"""_smallest_prime_factor(n) -> int

    Compute the smallest prime factor of an integer :math:`n \ge 2`.

    Returns ``1`` if :math:`n \le 1`, and :math:`n` itself if :math:`n` is prime.

    Args:
        n (int): Integer to find the smallest prime factor for.

    Returns:
        int: Smallest prime divisor greater than 1, or ``n`` if prime, or ``1`` if :math:`n \le 1`.

    Examples::

        >>> _smallest_prime_factor(12)
        2
        >>> _smallest_prime_factor(35)
        5
        >>> _smallest_prime_factor(17)
        17
    """
    if n <= 1:
        return 1
    if n % 2 == 0:
        return 2
    for i in range(3, math.isqrt(n) + 1, 2):
        if n % i == 0:
            return i
    return n


class LearnedWeightedGAP(nn.Module):
    r"""LearnedWeightedGAP(in_features, kernel_size=1, num_output=1, concat_gap=True)

    Applies learned weighted global average pooling over a 2D spatial feature map.

    Computes :math:`\text{num\_output}` spatial attention weight maps via a 2D convolution,
    normalizes each map with a spatial :func:`~torch.nn.functional.softmax`, and aggregates
    spatial features via weighted summation:

    .. math::
        W_{b, o, h, w} = \frac{\exp(Z_{b, o, h, w})}{\sum_{h', w'} \exp(Z_{b, o, h', w'})}

    .. math::
        \text{Pooled}_{b, o, c} = \sum_{h, w} W_{b, o, h, w} X_{b, c, h, w}

    where :math:`Z = \text{Conv2d}(X)`. Optionally concatenates standard unweighted Global
    Average Pooling (GAP) features:

    .. math::
        \text{GAP}(X)_{b, c} = \frac{1}{H \times W} \sum_{h, w} X_{b, c, h, w}

    Args:
        in_features (int): Number of channels in the input tensor :math:`C_{\text{in}}`.
        kernel_size (int, optional): Kernel size of the 2D spatial weighter convolution.
            Must be an odd integer. Default: 1
        num_output (int, optional): Number of spatial attention maps to generate. Default: 1
        concat_gap (bool, optional): If ``True``, concatenates uniform GAP features to the output.
            Default: ``True``

    Shape:
        - Input: :math:`(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})`
        - Output:
          - If ``num_output == 1`` and ``concat_gap == False``: :math:`(N, C_{\text{in}})`
          - If ``num_output == 1`` and ``concat_gap == True``: :math:`(N, 2C_{\text{in}})`
          - If ``num_output > 1`` and ``concat_gap == False``:
            :math:`(N, \text{num\_output}, C_{\text{in}})`
          - If ``num_output > 1`` and ``concat_gap == True``:
            :math:`(N, \text{num\_output} + 1, C_{\text{in}})`

    Examples::

        >>> m = LearnedWeightedGAP(in_features=64, num_output=1, concat_gap=True)
        >>> input = torch.randn(4, 64, 16, 16)
        >>> output = m(input)
        >>> output.size()
        torch.Size([4, 128])

        >>> # Multi-head spatial attention without uniform GAP concatenation
        >>> m = LearnedWeightedGAP(in_features=64, kernel_size=3, num_output=4, concat_gap=False)
        >>> output = m(input)
        >>> output.size()
        torch.Size([4, 4, 64])
    """

    def __init__(
        self,
        in_features: int,
        kernel_size: int = 1,
        num_output: int = 1,
        concat_gap: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer, got {kernel_size}.")

        self.in_features = in_features
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.concat_gap = concat_gap

        padding = kernel_size // 2

        self.weighter_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=num_output,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Applies learned weighted global average pooling to the input tensor.

        Args:
            x (Tensor): Input feature tensor of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: Pooled output tensor.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got tensor with shape {tuple(x.shape)}."
            )

        logits = self.weighter_conv(x)
        weights = F.softmax(logits.flatten(start_dim=2), dim=-1).view_as(logits)

        pooled = torch.einsum("bohw, bchw -> boc", weights, x)

        if self.num_output == 1:
            pooled = pooled.squeeze(1)

        if self.concat_gap:
            gap = x.mean(dim=[2, 3])
            gap_target = gap if self.num_output == 1 else gap.unsqueeze(1)
            pooled = torch.cat([pooled, gap_target], dim=1)

        return pooled

    def get_config(self) -> dict[str, Any]:
        r"""get_config() -> dict

        Returns module configuration parameters.

        Returns:
            dict[str, Any]: Dictionary containing initialization parameters.
        """
        return {
            "in_features": self.in_features,
            "kernel_size": self.kernel_size,
            "num_output": self.num_output,
            "concat_gap": self.concat_gap,
        }

    def extra_repr(self) -> str:
        r"""extra_repr() -> str

        Set the extra representation of the module.

        Returns:
            str: Module string representation.
        """
        return (
            f"in_features={self.in_features}, "
            f"kernel_size={self.kernel_size}, "
            f"num_output={self.num_output}, "
            f"concat_gap={self.concat_gap}"
        )


class AdaptiveLearnedPool2d(nn.Module):
    r"""AdaptiveLearnedPool2d(in_features, intermediate_features, out_features, output_size)

    Applies learned adaptive spatial downsampling of 2D feature maps to a fixed :attr:`output_size`.

    Uses depthwise-separable convolutions to project input features, repeatedly applies a strided
    ``downsampling_core`` with stride equal to the smallest prime factor of each dimension, and
    fuses the downsampled features with global average pooled input features before projecting
    to :attr:`out_features`.

    Spatial dimensions of the output are guaranteed to be
    :math:`(H_{\text{out}}, W_{\text{out}}) = \text{output\_size}` regardless of input resolution.

    .. note::
        Padding is symmetrically bounded by :attr:`max_pad_ratio` (default: 3.0) to prevent
        excessive intermediate memory blowup when input sizes and target prime factors mismatch.
        A final :func:`~torch.nn.functional.adaptive_avg_pool2d` absorbs any residual scale difference.

    Args:
        in_features (int): Number of channels in the input tensor :math:`C_{\text{in}}`.
        intermediate_features (int): Number of intermediate feature channels used in depthwise cores.
        out_features (int): Number of channels in the output tensor :math:`C_{\text{out}}`.
        output_size (tuple[int, int]): Target spatial output size :math:`(H_{\text{out}}, W_{\text{out}})`.

    Shape:
        - Input: :math:`(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})` where :math:`H_{\text{in}}, W_{\text{in}} \ge 1`.
        - Output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    Examples::

        >>> m = AdaptiveLearnedPool2d(
        ...     in_features=32,
        ...     intermediate_features=64,
        ...     out_features=128,
        ...     output_size=(7, 7),
        ... )
        >>> input = torch.randn(2, 32, 28, 28)
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 128, 7, 7])
    """

    max_pad_ratio: float = 3.0
    _smallest_prime_factor = staticmethod(_smallest_prime_factor)

    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        out_features: int,
        output_size: tuple[int, int],
    ) -> None:
        super().__init__()
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError(f"output_size dimensions must be positive, got {output_size}.")
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.out_features = out_features
        self.output_size = output_size

        self.kernel_size, self.stride = self._get_downsample_params(output_size)

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

    def _get_downsample_params(
        self,
        output_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        r"""_get_downsample_params(output_size) -> tuple[tuple[int, int], tuple[int, int]]

        Computes kernel size and stride derived from the smallest prime factors of ``output_size``.

        Args:
            output_size (tuple[int, int]): Target spatial dimensions :math:`(H_{\text{out}}, W_{\text{out}})`.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: Tuple of ``(kernel_size, stride)`` where each is ``(k_h, k_w)``.
        """
        k_h = self._smallest_prime_factor(output_size[0])
        k_w = self._smallest_prime_factor(output_size[1])

        return (k_h, k_w), (k_h, k_w)

    def _pad_to_nearest_multiple(self, x: torch.Tensor, multiple: tuple[int, int]) -> torch.Tensor:
        r"""_pad_to_nearest_multiple(x, multiple) -> Tensor

        Pads spatial dimensions symmetrically to the nearest integer multiple.

        Args:
            x (Tensor): Input tensor of shape :math:`(N, C, H, W)`.
            multiple (tuple[int, int]): Target spatial dimension multiples :math:`(M_h, M_w)`.

        Returns:
            Tensor: Padded tensor with spatial dimensions divisible by ``multiple``.
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
        r"""_num_downsamples(size, target, stride) -> int

        Computes number of strided downsampling steps required to reach or exceed target.

        Args:
            size (int): Current spatial dimension size.
            target (int): Target spatial dimension size.
            stride (int): Downsampling stride per step.

        Returns:
            int: Number of downsampling applications.
        """
        if stride <= 1:
            return 0

        num = 0
        while target * stride**num < size:
            num += 1
        return num

    def _bounded_num_downsamples(self, height: int, width: int) -> int:
        r"""_bounded_num_downsamples(height, width) -> int

        Calculates downsample applications bounded by :attr:`max_pad_ratio`.

        Args:
            height (int): Input height :math:`H_{\text{in}}`.
            width (int): Input width :math:`W_{\text{in}}`.

        Returns:
            int: Safe number of downsampling core applications.
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
        r"""forward(x) -> Tensor

        Downsamples spatial dimensions to :attr:`output_size` and fuses with adaptive average pooled inputs.

        Args:
            x (Tensor): Input feature tensor of shape :math:`(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})`.

        Returns:
            Tensor: Pooled tensor of shape :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.
        """
        target_h, target_w = self.output_size
        k_h, k_w = self.kernel_size

        input_avg = F.adaptive_avg_pool2d(x, self.output_size)

        num_downsamples = self._bounded_num_downsamples(x.shape[-2], x.shape[-1])

        x = self._pad_to_nearest_multiple(
            x,
            (
                target_h * k_h**num_downsamples,
                target_w * k_w**num_downsamples,
            ),
        )

        input_features = self.input_conv(x)

        downsampled = input_features
        for _ in range(num_downsamples):
            downsampled = self.downsampling_core(downsampled)

        downsampled = F.adaptive_avg_pool2d(downsampled, self.output_size)

        pooled = torch.cat([input_avg, downsampled], dim=1)

        return self.output_conv(pooled)


class AdaptiveLearnedUnpool2d(nn.Module):
    r"""AdaptiveLearnedUnpool2d(in_features, intermediate_features, out_features, output_size)

    Applies learned adaptive spatial upsampling of 2D feature maps to a fixed :attr:`output_size`.

    The transposed-convolution counterpart of :class:`AdaptiveLearnedPool2d`. Uses depthwise-separable
    convolutions to project inputs, repeatedly applies a strided :class:`~torch.nn.ConvTranspose2d`
    ``upsampling_core`` with stride equal to the smallest prime factor of each dimension, and mixes
    upsampled features with bilinearly interpolated input features before projecting to :attr:`out_features`.

    Spatial dimensions of the output are guaranteed to be
    :math:`(H_{\text{out}}, W_{\text{out}}) = \text{output\_size}` regardless of input resolution.

    Args:
        in_features (int): Number of channels in the input tensor :math:`C_{\text{in}}`.
        intermediate_features (int): Number of intermediate feature channels used in depthwise cores.
        out_features (int): Number of channels in the output tensor :math:`C_{\text{out}}`.
        output_size (tuple[int, int]): Target spatial output size :math:`(H_{\text{out}}, W_{\text{out}})`.

    Shape:
        - Input: :math:`(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})` where :math:`H_{\text{in}}, W_{\text{in}} \ge 1`.
        - Output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    Examples::

        >>> m = AdaptiveLearnedUnpool2d(
        ...     in_features=128,
        ...     intermediate_features=64,
        ...     out_features=32,
        ...     output_size=(28, 28),
        ... )
        >>> input = torch.randn(2, 128, 7, 7)
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 32, 28, 28])
    """

    _smallest_prime_factor = staticmethod(_smallest_prime_factor)

    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        out_features: int,
        output_size: tuple[int, int],
    ) -> None:
        super().__init__()
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError(f"output_size dimensions must be positive, got {output_size}.")
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.out_features = out_features
        self.output_size = output_size

        self.kernel_size, self.stride = self._get_upsample_params(output_size)

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

        self.upsampling_core = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=intermediate_features,
                out_channels=intermediate_features,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0,
                output_padding=0,
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

    def _get_upsample_params(
        self,
        output_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        r"""_get_upsample_params(output_size) -> tuple[tuple[int, int], tuple[int, int]]

        Computes kernel size and stride derived from the smallest prime factors of ``output_size``.

        Args:
            output_size (tuple[int, int]): Target spatial dimensions :math:`(H_{\text{out}}, W_{\text{out}})`.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: Tuple of ``(kernel_size, stride)`` where each is ``(k_h, k_w)``.
        """
        k_h = self._smallest_prime_factor(output_size[0])
        k_w = self._smallest_prime_factor(output_size[1])

        return (k_h, k_w), (k_h, k_w)

    def _num_upsamples(self, size: int, target: int, factor: int) -> int:
        r"""_num_upsamples(size, target, factor) -> int

        Computes number of transposed-conv upsampling steps required to reach or exceed target.

        Args:
            size (int): Current spatial dimension size.
            target (int): Target spatial dimension size.
            factor (int): Upsampling factor per step.

        Returns:
            int: Number of upsampling applications.
        """
        if factor <= 1:
            return 0

        num = 0
        while size * factor**num < target:
            num += 1
        return num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Upsamples spatial dimensions to :attr:`output_size` and fuses with interpolated inputs.

        Args:
            x (Tensor): Input feature tensor of shape :math:`(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})`.

        Returns:
            Tensor: Upsampled tensor of shape :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.
        """
        target_h, target_w = self.output_size
        k_h, k_w = self.kernel_size

        num_upsamples = max(
            self._num_upsamples(x.shape[-2], target_h, k_h),
            self._num_upsamples(x.shape[-1], target_w, k_w),
        )

        input_features = self.input_conv(x)
        input_up = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        upsampled = input_features
        for _ in range(num_upsamples):
            upsampled = self.upsampling_core(upsampled)

        upsampled = F.interpolate(
            upsampled, size=self.output_size, mode="bilinear", align_corners=False
        )

        out = torch.cat([input_up, upsampled], dim=1)

        return self.output_conv(out)
