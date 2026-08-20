r"""Spatial and temporal pooling and unpooling components for deep convolutional networks."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _smallest_prime_factor(n: int) -> int:
    r"""_smallest_prime_factor(n) -> int.

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

        if kernel_size == 1:
            self.weighter_conv = nn.Conv2d(
                in_channels=in_features,
                out_channels=num_output,
                kernel_size=1,
            )
        else:
            self.weighter_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=in_features,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=in_features,
                ),
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=num_output,
                    kernel_size=1,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward(x) -> Tensor.

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
        r"""get_config() -> dict.

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
        r"""extra_repr() -> str.

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
                out_channels=intermediate_features + in_features,
                kernel_size=3,
                padding="same",
                groups=intermediate_features + in_features,
            ),
            nn.Conv2d(
                in_channels=intermediate_features + in_features,
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
        r"""_pad_to_nearest_multiple(x, multiple) -> Tensor.

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
        r"""_num_downsamples(size, target, stride) -> int.

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
        r"""_bounded_num_downsamples(height, width) -> int.

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
        r"""Forward(x) -> Tensor.

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

    def get_config(self) -> dict[str, Any]:
        r"""get_config() -> dict.

        Returns module configuration parameters.

        Returns:
            dict[str, Any]: Dictionary containing initialization parameters.
        """
        return {
            "in_features": self.in_features,
            "intermediate_features": self.intermediate_features,
            "out_features": self.out_features,
            "output_size": self.output_size,
        }

    def extra_repr(self) -> str:
        r"""extra_repr() -> str.

        Set the extra representation of the module.

        Returns:
            str: Module string representation.
        """
        return (
            f"in_features={self.in_features}, "
            f"intermediate_features={self.intermediate_features}, "
            f"out_features={self.out_features}, "
            f"output_size={self.output_size}"
        )


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
                out_channels=intermediate_features + in_features,
                kernel_size=3,
                padding="same",
                groups=intermediate_features + in_features,
            ),
            nn.Conv2d(
                in_channels=intermediate_features + in_features,
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
        r"""_num_upsamples(size, target, factor) -> int.

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
        r"""Forward(x) -> Tensor.

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

    def get_config(self) -> dict[str, Any]:
        r"""get_config() -> dict.

        Returns module configuration parameters.

        Returns:
            dict[str, Any]: Dictionary containing initialization parameters.
        """
        return {
            "in_features": self.in_features,
            "intermediate_features": self.intermediate_features,
            "out_features": self.out_features,
            "output_size": self.output_size,
        }

    def extra_repr(self) -> str:
        r"""extra_repr() -> str.

        Set the extra representation of the module.

        Returns:
            str: Module string representation.
        """
        return (
            f"in_features={self.in_features}, "
            f"intermediate_features={self.intermediate_features}, "
            f"out_features={self.out_features}, "
            f"output_size={self.output_size}"
        )


class CausalConv1d(nn.Module):
    r"""CausalConv1d(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True)

    Applies a 1D causal convolution over an input signal composed of several input planes.

    Causality is enforced by padding the temporal dimension exclusively on the left (past)
    with :math:`(K - 1) \times d` zeros when :attr:`stride` is 1, ensuring that the output at
    time :math:`t` depends only on inputs at times :math:`\le t`.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int, optional): Stride of the convolution. Default: 1
        dilation (int, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{\text{in}}, L_{\text{in}})`
        - Output: :math:`(N, C_{\text{out}}, L_{\text{out}})`

    Examples::

        >>> conv = CausalConv1d(in_channels=16, out_channels=32, kernel_size=3)
        >>> x = torch.randn(2, 16, 10)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([2, 32, 10])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = (kernel_size - 1) * dilation if stride == 1 else 0

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward(x) -> Tensor.

        Applies causal 1D convolution with left-only padding.

        Args:
            x (Tensor): Input tensor of shape :math:`(N, C_{\text{in}}, L_{\text{in}})`.

        Returns:
            Tensor: Convolved tensor of shape :math:`(N, C_{\text{out}}, L_{\text{out}})`.
        """
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""step(x_t, state=None) -> tuple[Tensor, Tensor]

        Performs an incremental streaming step over new temporal frame(s) :attr:`x_t`.

        Args:
            x_t (Tensor): Incoming temporal chunk of shape :math:`(N, C_{\text{in}}, L_{\text{chunk}})`.
            state (Tensor, optional): Cached receptive field buffer of shape
                :math:`(N, C_{\text{in}}, \text{padding})`. If ``None``, a zero buffer is initialized.

        Returns:
            tuple[Tensor, Tensor]: Tuple of ``(output_chunk, new_state)``.
        """
        if self.padding == 0:
            return self.conv(x_t), x_t

        if state is None:
            state = torch.zeros(
                x_t.shape[0],
                self.in_channels,
                self.padding,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        x_full = torch.cat([state, x_t], dim=-1)
        out = self.conv(x_full)
        new_state = x_full[..., -self.padding :]
        return out, new_state

    def get_config(self) -> dict[str, Any]:
        r"""get_config() -> dict.

        Returns module configuration parameters.

        Returns:
            dict[str, Any]: Dictionary containing initialization parameters.
        """
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.conv.bias is not None,
        }

    def extra_repr(self) -> str:
        r"""extra_repr() -> str.

        Set the extra representation of the module.

        Returns:
            str: Module string representation.
        """
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"dilation={self.dilation}, "
            f"padding={self.padding}, "
            f"groups={self.groups}"
        )


class CausalAdaptiveLearnedPool(nn.Module):
    r"""CausalAdaptiveLearnedPool(in_features, intermediate_features, out_features, output_size, temporal_dim=2)

    Applies learned adaptive temporal downsampling to a fixed :attr:`output_size` using causal convolutions.

    The temporal counterpart of :class:`AdaptiveLearnedPool2d`. Uses causal depthwise-separable
    1D convolutions (:class:`CausalConv1d`) to project input features, repeatedly applies a causal strided
    ``downsampling_core`` with stride equal to the smallest prime factor of the target temporal length,
    and fuses the downsampled features with causal unpadded adaptive average pooled input features before
    projecting to :attr:`out_features`.

    .. math::
        Y[t] = f(X[\le t]) = \sum_{k=0}^{K-1} W[k] \cdot \tilde{X}[t + (K-1)d - kd]

    where :math:`\tilde{X}` is left-padded with :math:`(K-1)d` zeros at the beginning of the sequence.
    Downsampling iteratively reduces the temporal dimension via prime factor strides:

    .. math::
        T_{m+1} = \lfloor T_m / k \rfloor, \quad k = \text{smallest\_prime\_factor}(T_{\text{out}})

    The final pooled representation combines unpadded causal average pooling with learned convolutions:

    .. math::
        \text{Pooled} = \text{OutputConv}\left(\left[\text{CausalAvgPool}_{T_{\text{out}}}(X), \text{Downsampled}\right]\right)

    All temporal padding is strictly applied to the left (past) with zero right padding, guaranteeing that
    no future information leaks into preceding time steps. The temporal dimension of the output is
    guaranteed to be :math:`T_{\text{out}} = \text{output\_size}` regardless of input length.

    Supports arbitrary input tensors with temporal dimension specified by :attr:`temporal_dim`
    (e.g. 3D :math:`(N, C, T)`, 3D sequence-first :math:`(N, T, C)`, 4D :math:`(N, C, T, W)`,
    5D video :math:`(N, C, T, H, W)`), and provides streaming helper methods
    (:meth:`init_streaming_state` and :meth:`streaming_step`) for online real-time inference.

    See :class:`CausalConv1d` and :class:`AdaptiveLearnedPool2d` for related spatial and 1D building blocks.

    Args:
        in_features (int): Number of channels in the input tensor :math:`C_{\text{in}}`.
        intermediate_features (int): Number of intermediate feature channels used in depthwise cores.
        out_features (int): Number of channels in the output tensor :math:`C_{\text{out}}`.
        output_size (int or tuple[int, ...]): Target temporal output size :math:`T_{\text{out}}`.
        temporal_dim (int, optional): Index of the temporal dimension. Default: 2

    Shape:
        - Input: :math:`(N, C_{\text{in}}, T_{\text{in}})` or :math:`(N, C_{\text{in}}, T_{\text{in}}, H_{\text{in}}, W_{\text{in}})`
          or arbitrary :math:`(N, C_{\text{in}}, T_{\text{in}}, \dots)` where :math:`T_{\text{in}} \ge 1`.
        - Output: :math:`(N, C_{\text{out}}, T_{\text{out}})` or :math:`(N, C_{\text{out}}, T_{\text{out}}, H_{\text{in}}, W_{\text{in}})`.

    Examples::

        >>> m = CausalAdaptiveLearnedPool(
        ...     in_features=32,
        ...     intermediate_features=64,
        ...     out_features=128,
        ...     output_size=4,
        ... )
        >>> # 3D temporal sequence (B, C, T)
        >>> input_3d = torch.randn(2, 32, 16)
        >>> output_3d = m(input_3d)
        >>> output_3d.size()
        torch.Size([2, 128, 4])

        >>> # 5D video tensor (B, C, T, H, W)
        >>> input_video = torch.randn(2, 32, 16, 8, 8)
        >>> output_video = m(input_video)
        >>> output_video.size()
        torch.Size([2, 128, 4, 8, 8])

        >>> # Online streaming frame by frame
        >>> state = m.init_streaming_state(batch_size=1, channels=32, spatial_shape=(8, 8))
        >>> frame = torch.randn(1, 32, 1, 8, 8)
        >>> out_step, state = m.streaming_step(frame, state)
        >>> out_step.size()
        torch.Size([1, 128, 4, 8, 8])
    """

    max_pad_ratio: float = 3.0
    _smallest_prime_factor = staticmethod(_smallest_prime_factor)

    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        out_features: int,
        output_size: int | tuple[int, ...],
        temporal_dim: int = 2,
    ) -> None:
        super().__init__()
        if isinstance(output_size, tuple | list):
            if len(output_size) != 1:
                raise ValueError(
                    f"output_size for temporal pooler must be an integer or 1-tuple, got {output_size}."
                )
            out_t = int(output_size[0])
        else:
            out_t = int(output_size)

        if out_t < 1:
            raise ValueError(f"output_size dimensions must be positive, got {output_size}.")

        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.out_features = out_features
        self.output_size = out_t
        self.temporal_dim = temporal_dim

        self.kernel_size, self.stride = self._get_downsample_params(self.output_size)

        self.input_conv = nn.Sequential(
            CausalConv1d(
                in_channels=in_features,
                out_channels=in_features,
                kernel_size=3,
                groups=in_features,
            ),
            nn.Conv1d(
                in_channels=in_features,
                out_channels=intermediate_features,
                kernel_size=1,
            ),
            nn.GELU(approximate="tanh"),
        )

        self.downsampling_core = nn.Sequential(
            nn.Conv1d(
                in_channels=intermediate_features,
                out_channels=intermediate_features,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0,
                groups=intermediate_features,
            ),
            nn.Conv1d(
                in_channels=intermediate_features,
                out_channels=intermediate_features,
                kernel_size=1,
            ),
            nn.GELU(approximate="tanh"),
        )

        self.output_conv = nn.Sequential(
            CausalConv1d(
                in_channels=intermediate_features + in_features,
                out_channels=intermediate_features + in_features,
                kernel_size=3,
                groups=intermediate_features + in_features,
            ),
            nn.Conv1d(
                in_channels=intermediate_features + in_features,
                out_channels=out_features,
                kernel_size=1,
            ),
        )

    def _get_downsample_params(self, output_size: int) -> tuple[int, int]:
        r"""_get_downsample_params(output_size) -> tuple[int, int]

        Computes kernel size and stride derived from the smallest prime factor of ``output_size``.

        Args:
            output_size (int): Target temporal dimension :math:`T_{\text{out}}`.

        Returns:
            tuple[int, int]: Tuple of ``(kernel_size, stride)``.
        """
        k = self._smallest_prime_factor(output_size)
        return k, k

    def _pad_to_nearest_multiple(self, x: torch.Tensor, multiple: int) -> torch.Tensor:
        r"""_pad_to_nearest_multiple(x, multiple) -> Tensor.

        Pads temporal dimension on the left (past) to the nearest integer multiple.

        Args:
            x (Tensor): Input tensor of shape :math:`(N, C, T)`.
            multiple (int): Target temporal dimension multiple.

        Returns:
            Tensor: Left-padded tensor with temporal dimension divisible by ``multiple``.
        """
        length = x.shape[-1]
        pad_t = (multiple - length % multiple) % multiple
        if pad_t == 0:
            return x
        return F.pad(x, (pad_t, 0))

    def _num_downsamples(self, size: int, target: int, stride: int) -> int:
        r"""_num_downsamples(size, target, stride) -> int.

        Computes number of strided downsampling steps required to reach or exceed target.

        Args:
            size (int): Current temporal dimension size.
            target (int): Target temporal dimension size.
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

    def _bounded_num_downsamples(self, length: int) -> int:
        r"""_bounded_num_downsamples(length) -> int.

        Calculates downsample applications bounded by :attr:`max_pad_ratio`.

        Args:
            length (int): Input temporal length :math:`T_{\text{in}}`.

        Returns:
            int: Safe number of downsampling core applications.
        """
        target = self.output_size
        k = self.kernel_size
        num = self._num_downsamples(length, target, k)
        while num > 0:
            m = target * k**num
            pad = (m - length % m) % m
            if length + pad <= self.max_pad_ratio * length:
                break
            num -= 1
        return num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward(x) -> Tensor.

        Downsamples temporal dimension to :attr:`output_size` and fuses with adaptive average pooled inputs.

        Args:
            x (Tensor): Input feature tensor of shape :math:`(N, C_{\text{in}}, T_{\text{in}}, \dots)`
                with at least 3 dimensions.

        Returns:
            Tensor: Temporally pooled tensor of shape :math:`(N, C_{\text{out}}, T_{\text{out}}, \dots)`.
        """
        if x.ndim < 3:
            raise ValueError(
                f"Expected tensor with at least 3 dimensions (B, C, T, ...), got shape {tuple(x.shape)}."
            )

        ndim = x.ndim
        tdim = self.temporal_dim if self.temporal_dim >= 0 else ndim + self.temporal_dim
        if tdim < 0 or tdim >= ndim:
            raise IndexError(
                f"temporal_dim {self.temporal_dim} out of bounds for tensor with {ndim} dimensions."
            )

        if tdim != 2:
            x_standard = x.transpose(2, tdim)
            out_standard = self._forward_temporal_dim2(x_standard)
            return out_standard.transpose(2, tdim)

        return self._forward_temporal_dim2(x)

    def _forward_temporal_dim2(self, x: torch.Tensor) -> torch.Tensor:
        r"""_forward_temporal_dim2(x) -> Tensor.

        Internal forward path assuming the temporal dimension is located at index 2.

        Args:
            x (Tensor): Input tensor with shape :math:`(N, C, T, \dots)`.

        Returns:
            Tensor: Temporally pooled tensor with shape :math:`(N, C_{\text{out}}, T_{\text{out}}, \dots)`.
        """
        batch_size, channels, time_len = x.shape[:3]
        spatial_dims = x.shape[3:]

        if spatial_dims:
            perm = [0] + list(range(3, len(x.shape))) + [1, 2]
            x_flat = x.permute(*perm).reshape(-1, channels, time_len)
        else:
            x_flat = x

        target_t = self.output_size
        k_t = self.kernel_size

        input_avg = F.adaptive_avg_pool1d(x_flat, self.output_size)
        num_downsamples = self._bounded_num_downsamples(x_flat.shape[-1])

        x_padded = self._pad_to_nearest_multiple(
            x_flat,
            target_t * k_t**num_downsamples,
        )

        input_features = self.input_conv(x_padded)
        downsampled = input_features
        for _ in range(num_downsamples):
            downsampled = self.downsampling_core(downsampled)

        downsampled = F.adaptive_avg_pool1d(downsampled, self.output_size)
        pooled = torch.cat([input_avg, downsampled], dim=1)
        out_flat = self.output_conv(pooled)

        if spatial_dims:
            out_spatial = out_flat.view(batch_size, *spatial_dims, self.out_features, target_t)
            inv_perm = [0, len(spatial_dims) + 1, len(spatial_dims) + 2] + list(
                range(1, len(spatial_dims) + 1)
            )
            return out_spatial.permute(*inv_perm)
        return out_flat

    def get_config(self) -> dict[str, Any]:
        r"""get_config() -> dict.

        Returns module configuration parameters.

        Returns:
            dict[str, Any]: Dictionary containing initialization parameters.
        """
        return {
            "in_features": self.in_features,
            "intermediate_features": self.intermediate_features,
            "out_features": self.out_features,
            "output_size": self.output_size,
            "temporal_dim": self.temporal_dim,
        }

    def extra_repr(self) -> str:
        r"""extra_repr() -> str.

        Set the extra representation of the module.

        Returns:
            str: Module string representation.
        """
        return (
            f"in_features={self.in_features}, "
            f"intermediate_features={self.intermediate_features}, "
            f"out_features={self.out_features}, "
            f"output_size={self.output_size}, "
            f"temporal_dim={self.temporal_dim}"
        )

    def init_streaming_state(
        self,
        batch_size: int,
        channels: int,
        spatial_shape: tuple[int, ...] = (),
        buffer_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, Any]:
        r"""init_streaming_state(batch_size, channels, spatial_shape=(), buffer_size=None,
        device=None, dtype=torch.float32) -> dict.

        Initializes a streaming state buffer for online frame-by-frame processing.

        Args:
            batch_size (int): Batch size :math:`N`.
            channels (int): Channel count :math:`C_{\text{in}}`.
            spatial_shape (tuple[int, ...], optional): Spatial dimensions :math:`(H, W)` for video data.
                Default: ``()``
            buffer_size (int, optional): Temporal buffer depth. If ``None``, defaults to
                ``max(output_size * 4, 16)``.
            device (torch.device, optional): Device for the state buffer. Default: ``None``
            dtype (torch.dtype, optional): Data type for the state buffer. Default: ``torch.float32``

        Returns:
            dict[str, Any]: Initialized state dictionary containing the temporal history buffer.
        """
        buf_len = buffer_size if buffer_size is not None else max(self.output_size * 4, 16)
        shape = (batch_size, channels, buf_len, *spatial_shape)
        buffer = torch.zeros(shape, device=device, dtype=dtype)
        return {
            "buffer": buffer,
            "buffer_size": buf_len,
            "current_len": 0,
        }

    def streaming_step(
        self,
        x_t: torch.Tensor,
        state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        r"""streaming_step(x_t, state=None) -> tuple[Tensor, dict]

        Processes an incoming temporal frame or chunk in streaming mode while updating the state.

        Args:
            x_t (Tensor): Incoming temporal chunk of shape :math:`(N, C_{\text{in}}, T_{\text{chunk}}, \dots)`
                or :math:`(N, C_{\text{in}}, \dots)`.
            state (dict[str, Any], optional): Cached streaming state from prior step or
                :meth:`init_streaming_state`. If ``None``, a new state is automatically initialized.

        Returns:
            tuple[Tensor, dict[str, Any]]: Tuple of ``(pooled_output, updated_state)``.
        """
        if state is None:
            batch_size, channels = x_t.shape[0], x_t.shape[1]
            spatial_shape = tuple(x_t.shape[3:]) if x_t.ndim > 3 else ()
            state = self.init_streaming_state(
                batch_size=batch_size,
                channels=channels,
                spatial_shape=spatial_shape,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        buffer = state["buffer"]
        buf_size = state["buffer_size"]
        curr_len = state["current_len"]

        if x_t.ndim == 2 + len(buffer.shape[3:]):
            x_t = x_t.unsqueeze(2)

        t_chunk = x_t.shape[2]
        new_curr_len = min(curr_len + t_chunk, buf_size)
        new_buffer = torch.cat([buffer[:, :, t_chunk:], x_t], dim=2)

        pooled_out = self.forward(new_buffer)

        updated_state = {
            "buffer": new_buffer,
            "buffer_size": buf_size,
            "current_len": new_curr_len,
        }
        return pooled_out, updated_state


CausalAdaptiveLearnedPool1d = CausalAdaptiveLearnedPool
