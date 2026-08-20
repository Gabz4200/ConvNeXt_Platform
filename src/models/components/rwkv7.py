"""RWKV-7 "Goose" temporal processor block.

Reference:
- Paper: https://arxiv.org/abs/2503.14456
- Repo: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7
"""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKV7BlockState(NamedTuple):
    """Persistent recurrent state for a single RWKV-7 block."""

    att_x_prev: torch.Tensor  # (B, D)
    att_state: torch.Tensor  # (B, n_head, head_size, head_size)
    ffn_x_prev: torch.Tensor  # (B, D)


class RWKV7Block(nn.Module):
    """Default sequential RWKV-7 block with device-agnostic recurrent/parallel modes.

    The public interface exposes both:
    - ``forward(x, state)``: recurrent scan over the time dimension, returning a new state.
    - ``forward(x)``: full-sequence parallel scan using the same recurrence.

    Note:
        The default parallel path is a correct, device-agnostic Python scan. It is suitable
        for correctness testing and CPU inference, but it is not GPU-optimized for training.
        A separate compiled GPU scan path is used automatically when inputs are on CUDA.

    Args:
        dim: Model dimension.
        head_size: Per-head dimension. n_head = dim // head_size.
        layer_id: Layer index for initialization.
        total_layers: Total number of layers for initialization scaling.
        dim_ffn: FFN hidden dimension. Defaults to 4 * dim.
    """

    def __init__(
        self,
        dim: int,
        head_size: int,
        layer_id: int,
        total_layers: int,
        dim_ffn: int | None = None,
    ) -> None:
        super().__init__()
        if dim % head_size != 0:
            raise ValueError(f"dim {dim} must be divisible by head_size {head_size}")
        self.dim = dim
        self.head_size = head_size
        self.n_head = dim // head_size
        self.layer_id = layer_id
        self.total_layers = total_layers
        dim_ffn = dim_ffn if dim_ffn is not None else dim * 4
        self.dim_ffn = dim_ffn

        C = dim
        H = self.n_head
        N = self.head_size

        ratio_0_to_1 = layer_id / max(total_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_id / total_layers

        with torch.no_grad():
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - ddd.pow(0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - ddd.pow(0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - ddd.pow(0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - ddd.pow(0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - ddd.pow(0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - ddd.pow(0.2 * ratio_1_to_almost0))

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + ratio_0_to_1**0.3)

            def ortho_init(shape: tuple[int, ...], scale: float) -> torch.Tensor:
                tensor = torch.zeros(shape)
                with torch.no_grad():
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
                        nn.init.orthogonal_(tensor, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
                        for i in range(shape[0]):
                            nn.init.orthogonal_(tensor[i], gain=gain * scale)
                return tensor

            D_DECAY_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init((D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            D_AAA_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init((D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            D_MV_LORA = max(32, int(round((1.7 * (C**0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init((D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            D_GATE_LORA = max(32, int(round((5.0 * (C**0.5)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init((D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        with torch.no_grad():
            self.receptance.weight.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.zero_()

        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

        self.channel_key = nn.Linear(C, self.dim_ffn, bias=False)
        self.channel_value = nn.Linear(self.dim_ffn, C, bias=False)
        with torch.no_grad():
            self.channel_value.weight.zero_()
            nn.init.orthogonal_(self.channel_key.weight, gain=(self.dim_ffn**0.5))

        self.ln_out = nn.LayerNorm(C)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> RWKV7BlockState:
        return RWKV7BlockState(
            att_x_prev=torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
            att_state=torch.zeros(
                batch_size,
                self.n_head,
                self.head_size,
                self.head_size,
                device=device,
                dtype=torch.float32,
            ),
            ffn_x_prev=torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: RWKV7BlockState | None,
    ) -> tuple[torch.Tensor, RWKV7BlockState]:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, C) input, got shape {tuple(x.shape)}")
        B, T, C = x.shape
        if C != self.dim:
            raise ValueError(f"Input dim {C} does not match block dim {self.dim}")

        if state is None:
            state = self.initial_state(B, x.device, x.dtype)

        x, state, _ = self._forward_impl(x, state, v_first=None)
        return x, state

    def _forward_impl(
        self,
        x: torch.Tensor,
        state: RWKV7BlockState,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RWKV7BlockState, torch.Tensor]:
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size

        x_att = self.ln1(x)
        dx, state, v_first = _time_mixing(
            B=B,
            T=T,
            C=C,
            H=H,
            N=N,
            x=x_att,
            state=state,
            x_r=self.x_r,
            x_w=self.x_w,
            x_k=self.x_k,
            x_v=self.x_v,
            x_a=self.x_a,
            x_g=self.x_g,
            w0=self.w0,
            w1=self.w1,
            w2=self.w2,
            a0=self.a0,
            a1=self.a1,
            a2=self.a2,
            v0=self.v0,
            v1=self.v1,
            v2=self.v2,
            g1=self.g1,
            g2=self.g2,
            k_k=self.k_k,
            k_a=self.k_a,
            r_k=self.r_k,
            receptance=self.receptance,
            key=self.key,
            value=self.value,
            output=self.output,
            ln_x=self.ln_x,
            time_shift=self.time_shift,
            layer_id=self.layer_id,
            v_first=v_first,
        )
        x = x + dx

        x_ffn = self.ln2(x)
        dx, state = _channel_mixing(
            B=B,
            T=T,
            C=C,
            x=x_ffn,
            state=state,
            x_k=self.x_k,
            time_shift=self.time_shift,
            channel_key=self.channel_key,
            channel_value=self.channel_value,
        )
        x = x + dx

        return x, state, v_first


def _channel_mixing(
    B: int,
    T: int,
    C: int,
    x: torch.Tensor,
    state: RWKV7BlockState,
    x_k: torch.Tensor,
    time_shift: nn.ZeroPad2d,
    channel_key: nn.Linear,
    channel_value: nn.Linear,
) -> tuple[torch.Tensor, RWKV7BlockState]:
    xx = time_shift(x) - x
    k = x + xx * x_k
    k = torch.relu(channel_key(k)) ** 2
    out = channel_value(k)
    return out, RWKV7BlockState(
        att_x_prev=state.att_x_prev,
        att_state=state.att_state,
        ffn_x_prev=x[:, -1],
    )


def _time_mixing(
    B: int,
    T: int,
    C: int,
    H: int,
    N: int,
    x: torch.Tensor,
    state: RWKV7BlockState,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    receptance: nn.Linear,
    key: nn.Linear,
    value: nn.Linear,
    output: nn.Linear,
    ln_x: nn.GroupNorm,
    time_shift: nn.ZeroPad2d,
    layer_id: int,
    v_first: torch.Tensor | None,
) -> tuple[torch.Tensor, RWKV7BlockState, torch.Tensor]:
    xx = time_shift(x) - x

    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    r = receptance(xr)
    w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
    k = key(xk)
    v = value(xv)

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = k * k_k
    kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
    k = k * (1 + (a - 1) * k_a)

    w = torch.exp(-0.606531 * torch.sigmoid(w))

    scan_fn = _rwkv7_scan_gpu if x.is_cuda else _rwkv7_scan
    out, att_state = scan_fn(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        r_k=r_k,
        state=state.att_state,
    )

    out = ln_x(out.view(B * T, C)).view(B, T, C)
    out = out + (
        (r.view(B, T, H, N) * k.view(B, T, H, N) * r_k).sum(dim=-1, keepdim=True)
        * v.view(B, T, H, N)
    ).view(B, T, C)
    out = output(out * g)

    return (
        out,
        RWKV7BlockState(
            att_x_prev=x[:, -1],
            att_state=att_state,
            ffn_x_prev=state.ffn_x_prev,
        ),
        v_first,
    )


def _rwkv7_scan(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    r_k: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C = r.shape
    H = r_k.shape[0]
    N = r_k.shape[1]

    r = r.view(B, T, H, N)
    w = w.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    kk = kk.view(B, T, H, N)
    a = a.view(B, T, H, N)

    out = torch.empty((B, T, H, N), device=r.device, dtype=r.dtype)

    s = state.to(r.dtype)
    for t in range(T):
        rr = r[:, t]
        ww = w[:, t]
        kk_t = kk[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        a_t = a[:, t]

        vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
        ab = (-kk_t).unsqueeze(-1) @ (kk_t * a_t).unsqueeze(-2)
        s = s * ww.unsqueeze(-1) + s @ ab + vk
        out[:, t] = (s @ rr.unsqueeze(-1)).squeeze(-1)

    return out.view(B, T, C), s


def _rwkv7_scan_gpu(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    r_k: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, att_state = _rwkv7_scan(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        r_k=r_k,
        state=state,
    )
    return out, att_state


if torch.cuda.is_available():
    _rwkv7_scan_gpu = torch.compile(
        _rwkv7_scan_gpu,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune",
    )


class RWKV7Model(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        head_size: int = 64,
        n_layer: int = 12,
        dim_ffn: int | None = None,
        vocab_size: int = 65536,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_size = head_size
        self.n_head = dim // head_size
        self.n_layer = n_layer
        self.dim_ffn = dim_ffn if dim_ffn is None else dim * 4
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, dim)
        self.ln0 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.blocks.append(
                RWKV7Block(
                    dim=dim,
                    head_size=head_size,
                    layer_id=i,
                    total_layers=n_layer,
                    dim_ffn=self.dim_ffn,
                )
            )
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[RWKV7BlockState]:
        states = []
        for block in self.blocks:
            rwkv_block = cast(RWKV7Block, block)
            states.append(rwkv_block.initial_state(batch_size, device, dtype))
        return states

    def forward(
        self,
        tokens: torch.Tensor,
        state: list[RWKV7BlockState] | None = None,
    ) -> tuple[torch.Tensor, list[RWKV7BlockState]]:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 2:
            raise ValueError(f"Expected (B, T) tokens, got shape {tuple(tokens.shape)}")

        x = self.emb(tokens)
        x = self.ln0(x)

        if state is None:
            state = self.initial_state(x.shape[0], x.device, x.dtype)

        v_first = None
        for i, block in enumerate(self.blocks):
            rwkv_block = cast(RWKV7Block, block)
            x, state[i], v_first = rwkv_block._forward_impl(x, state[i], v_first=v_first)

        x = self.ln_out(x)
        logits = self.head(x)
        return logits, state
