"""ConvNeXt-RWKV7 Gamepad model architecture.

Combines AdaptiveLearnedPool2d, DINOv3-compatible ConvNeXt, LearnedWeightedGAP,
CausalConv1d, and RWKV-7 recurrent blocks to predict gamepad control vectors
(17 binary buttons and 2 joysticks with bounded [-1, 1] axes).
"""

from __future__ import annotations

import logging
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn

from src.models.components.convnext import ConvNeXt, convnext_sizes, load_dinov3_weights
from src.models.components.poolers import AdaptiveLearnedPool2d, CausalConv1d, LearnedWeightedGAP
from src.models.components.rwkv7 import RWKV7Block, RWKV7BlockState

logger = logging.getLogger(__name__)


class GamepadStreamingState(NamedTuple):
    """Persistent state for online streaming gamepad inference."""

    conv_state: Tensor
    rwkv_states: list[RWKV7BlockState]


class _InputNormalize(nn.Module):
    """Apply DINOv3-standard ImageNet normalization to raw image inputs.

    Mirrors the preprocessing pipeline in DINOv3's ``DINOv3ViTImageProcessorFast``:
    rescale ``[0, 255]`` inputs to ``[0, 1]`` (when needed), then normalize with
    ImageNet mean and standard deviation. Accepts both uint8 and float32 tensors.

    :param mean: Per-channel normalization mean. Default: (0.485, 0.456, 0.406).
    :param std: Per-channel normalization standard deviation. Default: (0.229, 0.224, 0.225).
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input tensor to ImageNet color space.

        :param x: Image tensor of shape `(N, 3, H, W)` or `(N, T, 3, H, W)` in
            uint8 ``[0, 255]`` or float ``[0, 1]`` range.
        :return: Normalized float32 tensor of same shape.
        """
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return (x - self.mean) / self.std


class GamepadHead(nn.Module):
    """Projection head mapping hidden representations to standard gamepad layout.

    Standard layout:
    - 17 boolean buttons: D-pad (4), Face (4), Bumpers (2), Triggers (2), Sticks (2), Menu (3).
    - 2 joysticks (4 axes total): Left Stick (X, Y) and Right Stick (X, Y) in range [-1.0, 1.0].

    :param in_features: Hidden dimension of the input feature representation.
    :param hidden_dim: Intermediate projection dimension. Default: 256.
    :param num_buttons: Number of boolean buttons. Default: 17.
    :param num_joysticks: Number of 2-axis joysticks. Default: 2.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_buttons: int = 17,
        num_joysticks: int = 2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_buttons = num_buttons
        self.num_joysticks = num_joysticks
        self.num_joystick_axes = num_joysticks * 2

        self.buttons_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_buttons),
        )

        self.joysticks_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_joystick_axes),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute gamepad predictions.

        :param x: Feature tensor of shape `(..., in_features)`.
        :return: Tuple containing:
            - `full_gamepad`: Combined tensor of shape `(..., num_buttons + 4)` where
              `[:17]` are button logits and `[17:]` are joystick coordinates.
            - `buttons`: Button logits of shape `(..., num_buttons)`.
            - `joysticks`: Continuous joystick coordinates of shape `(..., 4)` in `[-1.0, 1.0]`.
        """
        buttons = self.buttons_head(x)
        joysticks = self.joysticks_head(x)
        full_gamepad = torch.cat([buttons, joysticks], dim=-1)
        return full_gamepad, buttons, joysticks


class ConvNeXtRWKV7Gamepad(nn.Module):
    """End-to-end vision-temporal gamepad controller architecture.

    Pipeline:
    1. Input normalization: `_InputNormalize` applies DINOv3-standard ImageNet normalization.
    2. Spatial pooling: `AdaptiveLearnedPool2d` downsamples arbitrary-resolution inputs to a
       fixed resolution (224x224 for standard path, or 56x56 when bypassing the ConvNeXt stem).
    3. Spatial feature extraction: DINOv3-compatible `ConvNeXt` backbone.
    4. Spatial pooling: `LearnedWeightedGAP` aggregates spatial feature maps with attention.
    5. Temporal convolution: `CausalConv1d` with residual shortcut.
    6. Recurrent temporal reasoning: 4x `RWKV7Block` with residual streams.
    7. Output projection: `GamepadHead` mapping to 17 button logits + 4 joystick axes in `[-1, 1]`.

    :param in_chans: Number of input image channels. Default: 3.
    :param pool_intermediate_features: Intermediate channels for AdaptiveLearnedPool2d. Default: 32.
    :param convnext_size: Named ConvNeXt size string (e.g. 'tiny'). Default: 'tiny'.
    :param convnext_dims: Channel widths per ConvNeXt stage. Defaults to [96, 192, 384, 768].
    :param convnext_depths: Blocks per ConvNeXt stage. Defaults to [3, 3, 9, 3].
    :param convnext_drop_path_rate: Stochastic depth rate for ConvNeXt. Default: 0.0.
    :param convnext_layer_scale_init_value: LayerScale init value for ConvNeXt. Default: 1e-6.
    :param pretrained_dinov3: Whether to load pre-trained DINOv3 weights from HF Hub. Default: True.
    :param dinov3_repo_id: HF Hub repository ID for DINOv3 weights.
        Default: 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'.
    :param bypass_stem: If True, pooler outputs stage-0 channels directly and bypasses ConvNeXt stem.
        Default: False.
    :param freeze_convnext: If True, freeze ConvNeXt parameters while allowing gradients to flow
        through to AdaptiveLearnedPool2d. Default: True.
    :param gap_kernel_size: Kernel size for LearnedWeightedGAP. Default: 3.
    :param gap_concat: Whether to concatenate standard GAP features in LearnedWeightedGAP. Default: True.
    :param causal_conv_kernel_size: Kernel size for CausalConv1d. Default: 3.
    :param rwkv_dim: Hidden dimension for RWKV-7 blocks. Default: 256.
    :param rwkv_head_size: Head dimension for RWKV-7 attention. Default: 64.
    :param rwkv_layers: Number of RWKV-7 blocks. Default: 4.
    :param rwkv_dim_ffn: FFN dimension for RWKV-7 blocks. Defaults to 4 * rwkv_dim.
    :param head_hidden_dim: Hidden dimension for GamepadHead projection. Default: 256.
    :param num_buttons: Number of boolean buttons. Default: 17.
    :param num_joysticks: Number of 2-axis joysticks. Default: 2.
    """

    def __init__(
        self,
        in_chans: int = 3,
        pool_intermediate_features: int = 32,
        convnext_size: str = "tiny",
        convnext_dims: list[int] | None = None,
        convnext_depths: list[int] | None = None,
        convnext_drop_path_rate: float = 0.0,
        convnext_layer_scale_init_value: float = 1e-6,
        pretrained_dinov3: bool = True,
        dinov3_repo_id: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        bypass_stem: bool = False,
        freeze_convnext: bool = True,
        gap_kernel_size: int = 3,
        gap_concat: bool = True,
        causal_conv_kernel_size: int = 3,
        rwkv_dim: int = 256,
        rwkv_head_size: int = 64,
        rwkv_layers: int = 4,
        rwkv_dim_ffn: int | None = None,
        head_hidden_dim: int = 256,
        num_buttons: int = 17,
        num_joysticks: int = 2,
    ) -> None:
        super().__init__()
        if convnext_dims is None or convnext_depths is None:
            if convnext_size in convnext_sizes:
                spec = convnext_sizes[convnext_size]
                convnext_dims = convnext_dims or spec["dims"]
                convnext_depths = convnext_depths or spec["depths"]
            else:
                convnext_dims = convnext_dims or [96, 192, 384, 768]
                convnext_depths = convnext_depths or [3, 3, 9, 3]

        self.in_chans = in_chans
        self.convnext_dims = convnext_dims
        self.convnext_depths = convnext_depths
        self.bypass_stem = bypass_stem
        self.freeze_convnext = freeze_convnext
        self.rwkv_dim = rwkv_dim
        self.rwkv_layers = rwkv_layers
        self.num_buttons = num_buttons
        self.num_joysticks = num_joysticks

        # ImageNet normalization matching DINOv3's DINOv3ViTImageProcessorFast.
        # Handles both uint8 [0, 255] and float32 [0, 1] inputs.
        self.normalize = _InputNormalize()

        # 1. AdaptiveLearnedPool2d
        if bypass_stem:
            self.pooler = AdaptiveLearnedPool2d(
                in_features=in_chans,
                intermediate_features=pool_intermediate_features,
                out_features=convnext_dims[0],
                output_size=(56, 56),
            )
        else:
            self.pooler = AdaptiveLearnedPool2d(
                in_features=in_chans,
                intermediate_features=pool_intermediate_features,
                out_features=in_chans,
                output_size=(224, 224),
            )

        # 2. ConvNeXt backbone
        self.convnext = ConvNeXt(
            in_chans=in_chans,
            num_classes=0,
            depths=convnext_depths,
            dims=convnext_dims,
            drop_path_rate=convnext_drop_path_rate,
            layer_scale_init_value=convnext_layer_scale_init_value,
        )

        if pretrained_dinov3:
            load_dinov3_weights(self.convnext, repo_id=dinov3_repo_id)

        if freeze_convnext:
            self.set_convnext_freeze(True)

        # 3. LearnedWeightedGAP
        gap_in_features = convnext_dims[-1]
        self.gap = LearnedWeightedGAP(
            in_features=gap_in_features,
            kernel_size=gap_kernel_size,
            num_output=1,
            concat_gap=gap_concat,
        )
        gap_out_dim = gap_in_features * 2 if gap_concat else gap_in_features

        # 4. Temporal Projection and CausalConv1d with residual shortcut
        self.gap_to_rwkv: nn.Module = (
            nn.Linear(gap_out_dim, rwkv_dim) if gap_out_dim != rwkv_dim else nn.Identity()
        )
        self.causal_conv = CausalConv1d(
            in_channels=rwkv_dim,
            out_channels=rwkv_dim,
            kernel_size=causal_conv_kernel_size,
        )
        self.causal_norm = nn.LayerNorm(rwkv_dim)

        # 5. 4x RWKV-7 Blocks with residual connections
        self.rwkv_blocks = nn.ModuleList([
            RWKV7Block(
                dim=rwkv_dim,
                head_size=rwkv_head_size,
                layer_id=i,
                total_layers=rwkv_layers,
                dim_ffn=rwkv_dim_ffn,
            )
            for i in range(rwkv_layers)
        ])
        self.rwkv_norm = nn.LayerNorm(rwkv_dim)

        # 6. Final Gamepad Head
        self.gamepad_head = GamepadHead(
            in_features=rwkv_dim,
            hidden_dim=head_hidden_dim,
            num_buttons=num_buttons,
            num_joysticks=num_joysticks,
        )

    def set_convnext_freeze(self, freeze: bool) -> None:
        """Freeze or unfreeze ConvNeXt backbone parameters.

        :param freeze: If True, disables gradient calculation for all ConvNeXt weights.
        """
        self.freeze_convnext = freeze
        for param in self.convnext.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.convnext.eval()

    def train(self, mode: bool = True) -> ConvNeXtRWKV7Gamepad:
        """Set module training mode, keeping ConvNeXt in eval mode when frozen.

        :param mode: Whether to set training mode (True) or evaluation mode (False).
        :return: self.
        """
        super().train(mode)
        if self.freeze_convnext:
            self.convnext.eval()
        return self

    def _forward_convnext_stages(self, x: Tensor) -> Tensor:
        """Forward through ConvNeXt stages, respecting stem-bypassing ablation.

        :param x: Image feature tensor of shape `(N, C, H, W)`.
        :return: Final stage spatial feature map of shape `(N, dims[-1], H_out, W_out)`.
        """
        if self.bypass_stem:
            x = self.convnext.stages[0](x)
            for i in range(1, 4):
                x = self.convnext.downsample_layers[i](x)
                x = self.convnext.stages[i](x)
        else:
            for i in range(4):
                x = self.convnext.downsample_layers[i](x)
                x = self.convnext.stages[i](x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        """Extract temporal hidden representations before the gamepad head.

        :param x: Input tensor of shape `(B, C, H, W)` or `(B, T, C, H, W)`.
        :return: Hidden feature tensor of shape `(B, rwkv_dim)` or `(B, T, rwkv_dim)`.
        """
        is_5d = x.ndim == 5
        if is_5d:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
        elif x.ndim == 4:
            B, C, H, W = x.shape
            T = 1
            x_flat = x
        else:
            raise ValueError(
                f"Expected 4D (B, C, H, W) or 5D (B, T, C, H, W) input, got {tuple(x.shape)}"
            )

        # 1. Normalize input images (matches DINOv3ViTImageProcessorFast)
        x_norm = self.normalize(x_flat)

        # 2. Adaptive pooling
        x_pooled = self.pooler(x_norm)

        # 3. ConvNeXt stages
        x_conv = self._forward_convnext_stages(x_pooled)

        # 4. Spatial aggregation with LearnedWeightedGAP
        gap_feat = self.gap(x_conv)
        gap_feat = self.gap_to_rwkv(gap_feat)

        # 4. Temporal convolution with residual shortcut
        gap_seq = gap_feat.view(B, T, self.rwkv_dim).transpose(1, 2)
        conv_out = self.causal_conv(gap_seq)
        conv_res = gap_seq + conv_out
        x_rwkv = self.causal_norm(conv_res.transpose(1, 2))

        # 5. 4x RWKV-7 Blocks with residual connections
        v_first = None
        for block in self.rwkv_blocks:
            residual = x_rwkv
            rwkv_block = cast(RWKV7Block, block)
            dx, _, v_first = rwkv_block._forward_impl(
                x_rwkv,
                state=rwkv_block.initial_state(B, x.device, x.dtype),
                v_first=v_first,
            )
            x_rwkv = residual + dx

        x_rwkv = self.rwkv_norm(x_rwkv)

        if not is_5d:
            x_rwkv = x_rwkv.squeeze(1)

        return x_rwkv

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Perform forward pass through the entire model.

        :param x: Input tensor of shape `(B, C, H, W)` or `(B, T, C, H, W)`.
        :return: Tuple `(full_gamepad, buttons_logits, joysticks_values)`.
        """
        features = self.forward_features(x)
        return self.gamepad_head(features)

    def init_streaming_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> GamepadStreamingState:
        """Initialize streaming state for frame-by-frame online recurrent inference.

        :param batch_size: Batch size `N`.
        :param device: Device to allocate state buffers on.
        :param dtype: Data type for state buffers. Default: torch.float32.
        :return: Initialized `GamepadStreamingState`.
        """
        conv_state = torch.zeros(
            batch_size,
            self.rwkv_dim,
            self.causal_conv.padding,
            device=device,
            dtype=dtype,
        )
        rwkv_states = [
            cast(RWKV7Block, block).initial_state(batch_size, device, dtype)
            for block in self.rwkv_blocks
        ]
        return GamepadStreamingState(conv_state=conv_state, rwkv_states=rwkv_states)

    def step(
        self,
        x_t: Tensor,
        state: GamepadStreamingState | None = None,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], GamepadStreamingState]:
        """Perform an online recurrent streaming step on a single frame or N-frame chunk.

        :param x_t: Incoming frame tensor of shape `(B, C, H, W)` or chunk `(B, T, C, H, W)`.
        :param state: Streaming state from prior step or `init_streaming_state`.
        :return: Tuple `((full_gamepad, buttons, joysticks), updated_state)`.
        """
        is_4d = x_t.ndim == 4
        if is_4d:
            batch_size, c, h, w = x_t.shape
            time_chunk = 1
            x_flat = x_t
        elif x_t.ndim == 5:
            batch_size, time_chunk, c, h, w = x_t.shape
            x_flat = x_t.view(batch_size * time_chunk, c, h, w)
        else:
            raise ValueError(
                f"Expected (B, C, H, W) or (B, T, C, H, W) for streaming step, got {tuple(x_t.shape)}"
            )

        if state is None:
            state = self.init_streaming_state(batch_size, x_t.device, x_t.dtype)

        # 1. Pool + normalize + ConvNeXt + GAP
        x_pooled = self.pooler(x_flat)
        x_norm = self.normalize(x_pooled)
        x_conv = self._forward_convnext_stages(x_norm)
        gap_feat = self.gap(x_conv)
        gap_feat = self.gap_to_rwkv(gap_feat)

        # 2. CausalConv1d incremental step
        gap_seq = gap_feat.view(batch_size, time_chunk, self.rwkv_dim).transpose(1, 2)
        conv_out, new_conv_state = self.causal_conv.step(gap_seq, state.conv_state)
        conv_res = gap_seq + conv_out
        x_rwkv = self.causal_norm(conv_res.transpose(1, 2))

        # 3. RWKV-7 Blocks incremental step
        new_rwkv_states: list[RWKV7BlockState] = []
        v_first = None
        for i, block in enumerate(self.rwkv_blocks):
            residual = x_rwkv
            rwkv_block = cast(RWKV7Block, block)
            dx, block_st, v_first = rwkv_block._forward_impl(
                x_rwkv, state.rwkv_states[i], v_first=v_first
            )
            x_rwkv = residual + dx
            new_rwkv_states.append(block_st)

        x_rwkv = self.rwkv_norm(x_rwkv)
        if is_4d:
            x_rwkv = x_rwkv.squeeze(1)

        output = self.gamepad_head(x_rwkv)
        new_state = GamepadStreamingState(
            conv_state=new_conv_state, rwkv_states=new_rwkv_states
        )
        return output, new_state


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([
        ConvNeXtRWKV7Gamepad,
        GamepadHead,
        GamepadStreamingState,
    ])
