"""ConvNeXt backbone.

Adapted from the DINOv3 implementation:
  https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py

Original paper: "A ConvNet for the 2020s" (Liu et al., 2022)
  https://arxiv.org/pdf/2201.03545

Key changes vs. DINOv3 source:
- Removed numpy dependency (dp_rates built with torch.linspace).
- Removed DINOv3-specific forward_features / forward_features_list interface.
- Exposed a simple `forward(x) -> Tensor` returning the CLS embedding.
- Added `num_classes` head for plug-and-play classification.
"""

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last (default) and channels_first formats.

    channels_last  — (N, H, W, C) channels_first — (N, C, H, W)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first: manual computation
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    """ConvNeXt block.

    Implementation (2): DwConv -> Permute to NHWC -> LayerNorm -> Linear -> GELU -> Linear ->
    Permute back. Slightly faster than the channels_first variant in PyTorch.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return residual + self.drop_path(x)


class ConvNeXt(nn.Module):
    """ConvNeXt image encoder.

    Returns a flat feature vector (the global-average-pooled + normed CLS token)
    suitable for a classification head, or raw patch tokens for dense tasks.

    Args:
        in_chans: Number of input channels. Default: 3.
        num_classes: Number of output classes. 0 disables the head (returns embeddings). Default: 1000.
        depths: Blocks per stage. Default: [3, 3, 9, 3] (tiny).
        dims: Channel widths per stage. Default: [96, 192, 384, 768] (tiny).
        drop_path_rate: Stochastic depth rate. Default: 0.0.
        layer_scale_init_value: LayerScale init value. Default: 1e-6.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        patch_size: int | None = None,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            self.stages.append(
                nn.Sequential(
                    *[
                        Block(
                            dim=dims[i],
                            drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value,
                        )
                        for j in range(depths[i])
                    ]
                )
            )
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.embed_dim = dims[-1]
        self.embed_dims = dims
        self.n_blocks = len(self.downsample_layers)
        self.chunked_blocks = False
        self.n_storage_tokens = 0
        self.patch_size = patch_size
        self.input_pad_size = 4

        # DINOv3 applies identity norm to intermediate stages 0-2, real LayerNorm to stage 3.
        self.norms = nn.ModuleList([nn.Identity() for _ in range(3)] + [self.norm])

        # Classification head (Identity when num_classes == 0)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-init LayerScale gammas to their target value.
        for stage in self.stages:
            if isinstance(stage, (nn.Sequential, nn.ModuleList)):
                for block in stage:
                    if isinstance(block, Block) and block.gamma is not None:
                        nn.init.constant_(block.gamma, block.layer_scale_init_value)

    def forward_features(self, x: Tensor) -> Tensor:
        """Run the 4-stage encoder; return the normed CLS (global-avg-pool) token."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # Global average pool -> (N, C)
        cls = x.mean([-2, -1])
        return self.norm(cls)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits (or raw embeddings when head is Identity)."""
        return self.head(self.forward_features(x))

    def _get_intermediate_layers(
        self, x: Tensor, n: int | Sequence[int] = 1
    ) -> list[tuple[Tensor, Tensor]]:
        h, w = x.shape[-2:]
        output = []
        total_block_len = len(self.downsample_layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i in range(total_block_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                cls = x.mean([-2, -1])  # (N, C)
                patches = x  # (N, C, H, W)
                if self.patch_size is not None:
                    patches = F.interpolate(
                        patches,
                        size=(h // self.patch_size, w // self.patch_size),
                        mode="bilinear",
                        antialias=True,
                    )
                output.append((cls, patches))
        return output

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple[Tensor | tuple[Tensor, Tensor], ...]:
        """Extract feature maps from intermediate stages (DINOv3-compatible API).

        Args:
            x: Input image tensor.
            n: Number of last stages to return, or explicit stage indices.
            reshape: If True, keep spatial NCHW layout; otherwise flatten to NLC.
            return_class_token: If True, also return the CLS token per layer.
            norm: Apply the stage norm before returning.
        """
        outputs = self._get_intermediate_layers(x, n)

        if norm:
            nchw_shapes = [out[1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]
            outputs = [
                (norm_fn(cls), norm_fn(patches.flatten(-2).permute(0, 2, 1)))
                for (cls, patches), norm_fn in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls, tokens.permute(0, 2, 1).reshape(*nchw).contiguous())
                    for (cls, tokens), nchw in zip(outputs, nchw_shapes)
                ]
        elif not reshape:
            outputs = [(cls, patches.flatten(-2).permute(0, 2, 1)) for (cls, patches) in outputs]

        class_tokens = [o[0] for o in outputs]
        patch_tokens = [o[1] for o in outputs]
        if return_class_token:
            return tuple(zip(patch_tokens, class_tokens))
        return tuple(patch_tokens)


if hasattr(torch.serialization, "add_safe_globals"):
    safe_list: list[Any] = [
        ConvNeXt,
        Block,
        LayerNorm,
        DropPath,
        nn.ModuleList,
        nn.Sequential,
        nn.Conv2d,
        nn.Linear,
        nn.Identity,
        nn.GELU,
        nn.CrossEntropyLoss,
    ]
    try:
        import timm.layers

        if hasattr(timm.layers, "DropPath"):
            safe_list.append(timm.layers.DropPath)
    except (ImportError, AttributeError):
        logging.getLogger(__name__).debug(
            "timm.layers.DropPath not available for safe_globals registration"
        )
    torch.serialization.add_safe_globals(safe_list)


convnext_sizes: dict[str, dict[str, Any]] = {
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
    "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
}


def build_convnext(size: str = "tiny", **kwargs: Any) -> ConvNeXt:
    """Instantiate a ConvNeXt from a named size string."""
    if size not in convnext_sizes:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(convnext_sizes)}")
    return ConvNeXt(**convnext_sizes[size], **kwargs)


TIMM_SUB_MAP = {
    "conv_dw.weight": "dwconv.weight",
    "conv_dw.bias": "dwconv.bias",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
    "mlp.fc1.weight": "pwconv1.weight",
    "mlp.fc1.bias": "pwconv1.bias",
    "mlp.fc2.weight": "pwconv2.weight",
    "mlp.fc2.bias": "pwconv2.bias",
    "gamma": "gamma",
}

FB_SUB_MAP = {
    "depthwise_conv.weight": "dwconv.weight",
    "depthwise_conv.bias": "dwconv.bias",
    "layer_norm.weight": "norm.weight",
    "layer_norm.bias": "norm.bias",
    "pointwise_conv1.weight": "pwconv1.weight",
    "pointwise_conv1.bias": "pwconv1.bias",
    "pointwise_conv2.weight": "pwconv2.weight",
    "pointwise_conv2.bias": "pwconv2.bias",
    "gamma": "gamma",
}


def convert_dinov3_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert state_dict keys from timm or HuggingFace/Facebook DINOv3 ConvNeXt format to our
    ConvNeXt format."""
    mapped: dict[str, Tensor] = {}
    if "stem.0.weight" in sd:
        # Timm format
        mapped["downsample_layers.0.0.weight"] = sd["stem.0.weight"]
        mapped["downsample_layers.0.0.bias"] = sd["stem.0.bias"]
        mapped["downsample_layers.0.1.weight"] = sd["stem.1.weight"]
        mapped["downsample_layers.0.1.bias"] = sd["stem.1.bias"]
        for s in range(1, 4):
            mapped[f"downsample_layers.{s}.0.weight"] = sd[f"stages.{s}.downsample.0.weight"]
            mapped[f"downsample_layers.{s}.0.bias"] = sd[f"stages.{s}.downsample.0.bias"]
            mapped[f"downsample_layers.{s}.1.weight"] = sd[f"stages.{s}.downsample.1.weight"]
            mapped[f"downsample_layers.{s}.1.bias"] = sd[f"stages.{s}.downsample.1.bias"]
        for k, v in sd.items():
            if k.startswith("stages."):
                parts = k.split(".")
                if "blocks" in parts:
                    s, b = parts[1], parts[3]
                    sub = ".".join(parts[4:])
                    if sub in TIMM_SUB_MAP:
                        mapped[f"stages.{s}.{b}.{TIMM_SUB_MAP[sub]}"] = v
        if "head.norm.weight" in sd:
            mapped["norm.weight"] = sd["head.norm.weight"]
            mapped["norm.bias"] = sd["head.norm.bias"]
            mapped["norms.3.weight"] = sd["head.norm.weight"]
            mapped["norms.3.bias"] = sd["head.norm.bias"]
    elif "stages.0.downsample_layers.0.weight" in sd:
        # Facebook / HF format
        for k, v in sd.items():
            if k.startswith("stages."):
                parts = k.split(".")
                s = parts[1]
                if parts[2] == "downsample_layers":
                    idx, sub = parts[3], parts[4]
                    mapped[f"downsample_layers.{s}.{idx}.{sub}"] = v
                elif parts[2] == "layers":
                    b, sub = parts[3], parts[4]
                    suffix = parts[5] if len(parts) > 5 else ""
                    key_name = f"{sub}.{suffix}" if suffix else sub
                    if key_name in FB_SUB_MAP:
                        mapped[f"stages.{s}.{b}.{FB_SUB_MAP[key_name]}"] = v
            elif k == "layer_norm.weight":
                mapped["norm.weight"] = v
                mapped["norms.3.weight"] = v
            elif k == "layer_norm.bias":
                mapped["norm.bias"] = v
                mapped["norms.3.bias"] = v
    else:
        mapped = sd

    return mapped


def load_dinov3_weights(
    model: ConvNeXt, repo_id: str = "timm/convnext_tiny.dinov3_lvd1689m"
) -> ConvNeXt:
    """Download DINOv3 ConvNeXt weights from HF Hub and load them into model."""
    import safetensors.torch as st
    from huggingface_hub import hf_hub_download

    weight_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    sd = st.load_file(weight_file)
    mapped_sd = convert_dinov3_state_dict(sd)
    model.load_state_dict(mapped_sd, strict=False)
    return model
