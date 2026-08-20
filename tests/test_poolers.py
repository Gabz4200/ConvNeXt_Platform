"""Tests for spatial and temporal pooling and unpooling neural network components."""

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.poolers import (
    AdaptiveLearnedPool2d,
    AdaptiveLearnedUnpool2d,
    CausalAdaptiveLearnedPool,
    CausalAdaptiveLearnedPool1d,
    CausalConv1d,
    LearnedWeightedGAP,
)


@pytest.mark.parametrize("num_output", [1, 2, 4])
@pytest.mark.parametrize("concat_gap", [True, False])
def test_learned_weighted_gap_shapes(num_output: int, concat_gap: bool) -> None:
    batch_size = 4
    in_features = 32
    height, width = 16, 16

    pooler = LearnedWeightedGAP(
        in_features=in_features,
        kernel_size=3,
        num_output=num_output,
        concat_gap=concat_gap,
    )

    x = torch.randn(batch_size, in_features, height, width)
    out = pooler(x)

    if num_output == 1:
        expected_channels = in_features * 2 if concat_gap else in_features
        assert out.shape == (batch_size, expected_channels)
    else:
        expected_outputs = num_output + 1 if concat_gap else num_output
        assert out.shape == (batch_size, expected_outputs, in_features)


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_learned_weighted_gap_gradient_flow(kernel_size: int) -> None:
    pooler = LearnedWeightedGAP(
        in_features=16, kernel_size=kernel_size, num_output=1, concat_gap=True
    )
    x = torch.randn(2, 16, 8, 8, requires_grad=True)

    out = pooler(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    for param in pooler.parameters():
        assert param.grad is not None


def test_learned_weighted_gap_config() -> None:
    pooler = LearnedWeightedGAP(in_features=64, kernel_size=3, num_output=2, concat_gap=False)
    config = pooler.get_config()

    assert config == {
        "in_features": 64,
        "kernel_size": 3,
        "num_output": 2,
        "concat_gap": False,
    }
    assert "in_features=64" in pooler.extra_repr()


def test_learned_weighted_gap_invalid_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size must be an odd integer"):
        LearnedWeightedGAP(in_features=32, kernel_size=2)


def test_learned_weighted_gap_invalid_input_dim() -> None:
    pooler = LearnedWeightedGAP(in_features=32, kernel_size=1)
    x_3d = torch.randn(2, 32, 16)
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        pooler(x_3d)


def _make_adaptive_pooler(
    in_features: int = 32,
    intermediate_features: int = 16,
    out_features: int = 8,
    output_size: tuple[int, int] = (4, 4),
) -> AdaptiveLearnedPool2d:
    return AdaptiveLearnedPool2d(
        in_features=in_features,
        intermediate_features=intermediate_features,
        out_features=out_features,
        output_size=output_size,
    )


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [
        ((16, 16), (4, 4)),
        ((17, 17), (4, 4)),
        ((3, 3), (4, 4)),
        ((33, 34), (8, 9)),
        ((64, 96), (8, 12)),
        ((20, 20), (7, 7)),
        ((16, 16), (1, 8)),
        ((16, 16), (1, 1)),
    ],
)
def test_adaptive_learned_pool_output_shape(
    input_size: tuple[int, int], output_size: tuple[int, int]
) -> None:
    batch_size = 2
    in_features = 32
    out_features = 8

    pooler = _make_adaptive_pooler(
        in_features=in_features,
        intermediate_features=16,
        out_features=out_features,
        output_size=output_size,
    )
    x = torch.randn(batch_size, in_features, *input_size)
    out = pooler(x)

    assert out.shape == (batch_size, out_features, *output_size)


@pytest.mark.parametrize(
    ("in_features", "intermediate_features"),
    [(8, 64), (32, 32), (64, 8), (16, 48)],
)
def test_adaptive_learned_pool_arbitrary_channels(
    in_features: int, intermediate_features: int
) -> None:
    pooler = _make_adaptive_pooler(
        in_features=in_features,
        intermediate_features=intermediate_features,
        output_size=(4, 4),
    )
    x = torch.randn(2, in_features, 16, 16)
    out = pooler(x)

    assert out.shape == (2, 8, 4, 4)


@pytest.mark.parametrize("height", [2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 24, 32, 33, 48])
@pytest.mark.parametrize("target", [1, 2, 4, 8])
def test_adaptive_learned_pool_exact_target_size(height: int, target: int) -> None:
    pooler = _make_adaptive_pooler(
        in_features=8,
        intermediate_features=8,
        out_features=4,
        output_size=(target, target),
    )
    x = torch.randn(1, 8, height, height)
    out = pooler(x)

    assert out.shape[-2:] == (target, target)


def test_adaptive_learned_pool_gradient_flow() -> None:
    pooler = _make_adaptive_pooler(in_features=16, intermediate_features=16, output_size=(4, 4))
    x = torch.randn(2, 16, 8, 8, requires_grad=True)

    out = pooler(x)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert all(param.grad is not None for param in pooler.parameters())


@pytest.mark.parametrize(
    ("size", "multiple", "expected"),
    [
        ((2, 3, 16, 16), (8, 9), (2, 3, 16, 18)),
        ((2, 3, 17, 19), (8, 9), (2, 3, 24, 27)),
        ((2, 3, 3, 5), (4, 4), (2, 3, 4, 8)),
        ((2, 3, 20, 30), (10, 10), (2, 3, 20, 30)),
        ((2, 3, 7, 9), (1, 5), (2, 3, 7, 10)),
    ],
)
def test_pad_to_nearest_multiple(
    size: tuple[int, int, int, int],
    multiple: tuple[int, int],
    expected: tuple[int, int, int, int],
) -> None:
    x = torch.randn(*size)
    padded = _make_adaptive_pooler()._pad_to_nearest_multiple(x, multiple)

    assert padded.shape == expected


def test_pad_to_nearest_multiple_returns_same_when_multiple() -> None:
    x = torch.randn(2, 3, 16, 16)
    padded = _make_adaptive_pooler()._pad_to_nearest_multiple(x, (4, 4))

    assert padded is x


def test_pad_to_nearest_multiple_symmetric_alignment() -> None:
    x = torch.randn(2, 3, 17, 19)
    padded = _make_adaptive_pooler()._pad_to_nearest_multiple(x, (8, 9))

    assert padded.shape == (2, 3, 24, 27)
    torch.testing.assert_close(padded[..., 3:20, 4:23], x)


@pytest.mark.parametrize(
    ("size", "target", "stride", "expected"),
    [
        (16, 4, 2, 2),
        (33, 8, 2, 3),
        (4, 8, 2, 0),
        (32, 7, 7, 1),
        (32, 7, 2, 3),
        (48, 12, 2, 2),
        (16, 16, 2, 0),
        (17, 16, 2, 1),
        (16, 1, 1, 0),
    ],
)
def test_num_downsamples(size: int, target: int, stride: int, expected: int) -> None:
    assert _make_adaptive_pooler()._num_downsamples(size, target, stride) == expected


@pytest.mark.parametrize(
    ("input_size", "output_size", "expected"),
    [
        ((64, 64), (8, 8), 3),
        ((16, 16), (4, 4), 2),
        ((56, 56), (7, 7), 1),
        ((20, 20), (7, 7), 1),
        ((33, 34), (8, 9), 2),
        ((7, 56), (7, 7), 0),
        ((3, 3), (4, 4), 0),
        ((16, 16), (1, 8), 1),
    ],
)
def test_bounded_num_downsamples(
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    expected: int,
) -> None:
    pooler = _make_adaptive_pooler(output_size=output_size)
    assert pooler._bounded_num_downsamples(*input_size) == expected


def test_adaptive_learned_pool_does_not_pad_blowup() -> None:
    pooler = _make_adaptive_pooler(
        in_features=32,
        intermediate_features=16,
        out_features=8,
        output_size=(7, 7),
    )
    x = torch.randn(2, 32, 56, 56)
    out = pooler(x)

    assert out.shape == (2, 8, 7, 7)


def test_adaptive_learned_pool_unpadded_input_avg() -> None:
    pooler = _make_adaptive_pooler(
        in_features=8,
        intermediate_features=16,
        out_features=8,
        output_size=(4, 4),
    )
    captured_pooled = []

    def hook(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured_pooled.append(inputs[0])

    hook_handle = pooler.output_conv.register_forward_hook(hook)

    # 7x7 input requires padding to 8x8 for strided convolutions
    x = torch.ones(2, 8, 7, 7)
    _ = pooler(x)
    hook_handle.remove()

    assert len(captured_pooled) == 1
    input_avg = captured_pooled[0][:, :8, :, :]
    expected_avg = F.adaptive_avg_pool2d(x, (4, 4))
    torch.testing.assert_close(input_avg, expected_avg)
    torch.testing.assert_close(input_avg, torch.ones_like(input_avg))


@pytest.mark.parametrize("output_size", [(0, 4), (4, 0), (0, 0)])
def test_adaptive_learned_pool_invalid_output_size(
    output_size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="output_size dimensions must be positive"):
        _make_adaptive_pooler(output_size=output_size)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
        (8, 2),
        (9, 3),
        (12, 2),
        (15, 3),
        (49, 7),
        (97, 97),
    ],
)
def test_smallest_prime_factor(n: int, expected: int) -> None:
    assert _make_adaptive_pooler()._smallest_prime_factor(n) == expected


def test_adaptive_learned_pool_downsample_params() -> None:
    pooler = _make_adaptive_pooler(output_size=(8, 9))
    assert pooler.kernel_size == (2, 3)
    assert pooler.stride == (2, 3)

    assert _make_adaptive_pooler(output_size=(7, 7)).kernel_size == (7, 7)
    assert _make_adaptive_pooler(output_size=(1, 8)).kernel_size == (1, 2)


def _make_adaptive_unpooler(
    in_features: int = 32,
    intermediate_features: int = 16,
    out_features: int = 8,
    output_size: tuple[int, int] = (16, 16),
) -> AdaptiveLearnedUnpool2d:
    return AdaptiveLearnedUnpool2d(
        in_features=in_features,
        intermediate_features=intermediate_features,
        out_features=out_features,
        output_size=output_size,
    )


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [
        ((4, 4), (16, 16)),
        ((3, 3), (8, 8)),
        ((5, 5), (16, 16)),
        ((1, 1), (8, 9)),
        ((8, 8), (32, 48)),
        ((3, 4), (8, 9)),
        ((4, 4), (8, 12)),
        ((16, 16), (4, 4)),
    ],
)
def test_adaptive_learned_unpool_output_shape(
    input_size: tuple[int, int], output_size: tuple[int, int]
) -> None:
    batch_size = 2
    in_features = 32
    out_features = 8

    pooler = _make_adaptive_unpooler(
        in_features=in_features,
        intermediate_features=16,
        out_features=out_features,
        output_size=output_size,
    )
    x = torch.randn(batch_size, in_features, *input_size)
    out = pooler(x)

    assert out.shape == (batch_size, out_features, *output_size)


@pytest.mark.parametrize(
    ("in_features", "intermediate_features"),
    [(8, 64), (32, 32), (64, 8), (16, 48)],
)
def test_adaptive_learned_unpool_arbitrary_channels(
    in_features: int, intermediate_features: int
) -> None:
    pooler = _make_adaptive_unpooler(
        in_features=in_features,
        intermediate_features=intermediate_features,
        output_size=(16, 16),
    )
    x = torch.randn(2, in_features, 4, 4)
    out = pooler(x)

    assert out.shape == (2, 8, 16, 16)


@pytest.mark.parametrize("height", [1, 2, 3, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("target", [1, 4, 8, 16])
def test_adaptive_learned_unpool_exact_target_size(height: int, target: int) -> None:
    pooler = _make_adaptive_unpooler(
        in_features=8,
        intermediate_features=8,
        out_features=4,
        output_size=(target, target),
    )
    x = torch.randn(1, 8, height, height)
    out = pooler(x)

    assert out.shape[-2:] == (target, target)


def test_adaptive_learned_unpool_gradient_flow() -> None:
    pooler = _make_adaptive_unpooler(in_features=16, intermediate_features=16, output_size=(8, 8))
    x = torch.randn(2, 16, 4, 4, requires_grad=True)

    out = pooler(x)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert all(param.grad is not None for param in pooler.parameters())


@pytest.mark.parametrize(
    ("size", "target", "factor", "expected"),
    [
        (1, 8, 2, 3),
        (3, 8, 2, 2),
        (8, 8, 2, 0),
        (9, 8, 2, 0),
        (1, 7, 7, 1),
        (4, 7, 7, 1),
        (7, 7, 7, 0),
        (1, 9, 3, 2),
        (4, 8, 1, 0),
    ],
)
def test_num_upsamples(size: int, target: int, factor: int, expected: int) -> None:
    assert _make_adaptive_unpooler()._num_upsamples(size, target, factor) == expected


def test_adaptive_learned_unpool_upsample_params() -> None:
    pooler = _make_adaptive_unpooler(output_size=(8, 9))
    assert pooler.kernel_size == (2, 3)
    assert pooler.stride == (2, 3)

    assert _make_adaptive_unpooler(output_size=(7, 7)).kernel_size == (7, 7)
    assert _make_adaptive_unpooler(output_size=(1, 8)).kernel_size == (1, 2)


@pytest.mark.parametrize("output_size", [(0, 4), (4, 0), (0, 0)])
def test_adaptive_learned_unpool_invalid_output_size(
    output_size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="output_size dimensions must be positive"):
        _make_adaptive_unpooler(output_size=output_size)


def test_causal_conv1d_shapes() -> None:
    conv = CausalConv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
    x = torch.randn(2, 8, 10)
    out = conv(x)
    assert out.shape == (2, 16, 10)

    # Strided
    conv_strided = CausalConv1d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
    out_strided = conv_strided(x)
    assert out_strided.shape == (2, 16, 5)


def test_causal_conv1d_gradient_causality() -> None:
    conv = CausalConv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1)
    x = torch.randn(1, 4, 8, requires_grad=True)
    out = conv(x)

    for t_out in range(8):
        grad_x = torch.autograd.grad(out[0, 0, t_out], x, retain_graph=True)[0]
        future_grads = grad_x[0, :, t_out + 1 :]
        if future_grads.numel() > 0:
            assert future_grads.abs().max().item() == 0.0


def test_causal_conv1d_streaming_equivalence() -> None:
    conv = CausalConv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1)
    x = torch.randn(2, 4, 10)
    full_out = conv(x)

    state = None
    stream_outs = []
    for t in range(10):
        x_t = x[:, :, t : t + 1]
        out_t, state = conv.step(x_t, state)
        stream_outs.append(out_t)

    stream_out = torch.cat(stream_outs, dim=-1)
    torch.testing.assert_close(full_out, stream_out)


def _make_causal_pooler(
    in_features: int = 32,
    intermediate_features: int = 16,
    out_features: int = 8,
    output_size: int | tuple[int, ...] = 4,
) -> CausalAdaptiveLearnedPool:
    return CausalAdaptiveLearnedPool(
        in_features=in_features,
        intermediate_features=intermediate_features,
        out_features=out_features,
        output_size=output_size,
    )


@pytest.mark.parametrize(
    ("input_t", "output_t"),
    [
        (16, 4),
        (17, 4),
        (3, 4),
        (33, 8),
        (64, 12),
        (20, 7),
        (16, 1),
        (1, 1),
    ],
)
def test_causal_adaptive_pool_3d_output_shape(input_t: int, output_t: int) -> None:
    batch_size = 2
    in_features = 32
    out_features = 8

    pooler = _make_causal_pooler(
        in_features=in_features,
        intermediate_features=16,
        out_features=out_features,
        output_size=output_t,
    )
    x = torch.randn(batch_size, in_features, input_t)
    out = pooler(x)

    assert out.shape == (batch_size, out_features, output_t)


@pytest.mark.parametrize(
    ("input_t", "output_t", "spatial_size"),
    [
        (16, 4, (8, 8)),
        (17, 4, (7, 7)),
        (3, 4, (4, 4)),
        (20, 7, (6, 6)),
        (8, 2, (10, 10)),
    ],
)
def test_causal_adaptive_pool_5d_video_shape(
    input_t: int, output_t: int, spatial_size: tuple[int, int]
) -> None:
    batch_size = 2
    in_features = 16
    out_features = 8

    pooler = _make_causal_pooler(
        in_features=in_features,
        intermediate_features=16,
        out_features=out_features,
        output_size=output_t,
    )
    x = torch.randn(batch_size, in_features, input_t, *spatial_size)
    out = pooler(x)

    assert out.shape == (batch_size, out_features, output_t, *spatial_size)


@pytest.mark.parametrize(
    ("in_features", "intermediate_features"),
    [(8, 64), (32, 32), (64, 8), (16, 48)],
)
def test_causal_adaptive_pool_arbitrary_channels(
    in_features: int, intermediate_features: int
) -> None:
    pooler = _make_causal_pooler(
        in_features=in_features,
        intermediate_features=intermediate_features,
        output_size=4,
    )
    x = torch.randn(2, in_features, 16)
    out = pooler(x)

    assert out.shape == (2, 8, 4)


def test_causal_adaptive_pool_gradient_flow() -> None:
    pooler = _make_causal_pooler(in_features=16, intermediate_features=16, output_size=4)
    x = torch.randn(2, 16, 8, requires_grad=True)

    out = pooler(x)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert all(param.grad is not None for param in pooler.parameters())


@pytest.mark.parametrize(
    ("input_t", "output_t"),
    [
        (16, 4),
        (20, 5),
        (17, 4),
        (8, 2),
    ],
)
def test_causal_adaptive_pool_causality_strict_gradient(input_t: int, output_t: int) -> None:
    pooler = _make_causal_pooler(
        in_features=8,
        intermediate_features=16,
        out_features=4,
        output_size=output_t,
    )
    x = torch.randn(1, 8, input_t, requires_grad=True)
    out = pooler(x)

    for t_out in range(output_t):
        grad_x = torch.autograd.grad(out[0, 0, t_out], x, retain_graph=True)[0]
        if t_out < output_t - 2:
            assert grad_x[0, :, input_t - 1].abs().max().item() == 0.0


def test_causal_adaptive_pool_streaming_step_3d() -> None:
    pooler = _make_causal_pooler(
        in_features=8, intermediate_features=16, out_features=4, output_size=4
    )
    state = pooler.init_streaming_state(batch_size=2, channels=8, buffer_size=16)

    for _ in range(10):
        frame = torch.randn(2, 8, 1)
        out, state = pooler.streaming_step(frame, state)
        assert out.shape == (2, 4, 4)
        assert state["buffer"].shape == (2, 8, 16)


def test_causal_adaptive_pool_streaming_step_5d() -> None:
    pooler = _make_causal_pooler(
        in_features=8, intermediate_features=16, out_features=4, output_size=4
    )
    state = pooler.init_streaming_state(
        batch_size=2, channels=8, spatial_shape=(4, 4), buffer_size=16
    )

    for _ in range(5):
        frame = torch.randn(2, 8, 1, 4, 4)
        out, state = pooler.streaming_step(frame, state)
        assert out.shape == (2, 4, 4, 4, 4)
        assert state["buffer"].shape == (2, 8, 16, 4, 4)


def test_causal_adaptive_pool_streaming_auto_init() -> None:
    pooler = _make_causal_pooler(
        in_features=8, intermediate_features=16, out_features=4, output_size=4
    )
    frame = torch.randn(2, 8, 1)
    out, state = pooler.streaming_step(frame, None)
    assert out.shape == (2, 4, 4)
    assert state is not None
    assert state["buffer"].shape == (2, 8, 16)


def test_causal_adaptive_pool_tuple_output_size() -> None:
    pooler = _make_causal_pooler(output_size=(4,))
    assert pooler.output_size == 4
    x = torch.randn(2, 32, 16)
    out = pooler(x)
    assert out.shape == (2, 8, 4)


@pytest.mark.parametrize("output_size", [0, -1, (0,), (-2,), (4, 4)])
def test_causal_adaptive_pool_invalid_output_size(output_size: Any) -> None:
    with pytest.raises(ValueError, match="output_size"):
        _make_causal_pooler(output_size=output_size)


def test_causal_adaptive_pool_invalid_input_dim() -> None:
    pooler = _make_causal_pooler()
    x_2d = torch.randn(2, 32)
    with pytest.raises(ValueError, match="Expected tensor with at least 3 dimensions"):
        pooler(x_2d)


def test_causal_adaptive_pool_pad_to_nearest_multiple() -> None:
    pooler = _make_causal_pooler()
    x = torch.randn(2, 3, 10)
    padded = pooler._pad_to_nearest_multiple(x, 4)
    assert padded.shape == (2, 3, 12)
    # Check left-only padding: original x should match right slice
    torch.testing.assert_close(padded[..., 2:], x)

    # When already multiple, returns identical tensor
    x_mult = torch.randn(2, 3, 12)
    assert pooler._pad_to_nearest_multiple(x_mult, 4) is x_mult


def test_causal_adaptive_pool_bounded_downsamples() -> None:
    pooler = _make_causal_pooler(output_size=4)
    assert pooler._bounded_num_downsamples(16) == 2
    assert pooler._bounded_num_downsamples(4) == 0
    assert pooler._bounded_num_downsamples(3) == 0


def test_causal_adaptive_pool_unpadded_input_avg() -> None:
    pooler = _make_causal_pooler(
        in_features=8,
        intermediate_features=16,
        out_features=8,
        output_size=4,
    )
    captured_pooled = []

    def hook(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured_pooled.append(inputs[0])

    hook_handle = pooler.output_conv.register_forward_hook(hook)

    # 7 length input requires padding to 8 for strided convolutions
    x = torch.ones(2, 8, 7)
    _ = pooler(x)
    hook_handle.remove()

    assert len(captured_pooled) == 1
    input_avg = captured_pooled[0][:, :8, :]
    expected_avg = F.adaptive_avg_pool1d(x, 4)
    torch.testing.assert_close(input_avg, expected_avg)
    torch.testing.assert_close(input_avg, torch.ones_like(input_avg))


def test_causal_adaptive_pool1d_alias() -> None:
    assert CausalAdaptiveLearnedPool1d is CausalAdaptiveLearnedPool


def test_poolers_get_config_and_extra_repr() -> None:
    # AdaptiveLearnedPool2d
    pool2d = AdaptiveLearnedPool2d(
        in_features=32, intermediate_features=16, out_features=8, output_size=(4, 4)
    )
    assert pool2d.get_config() == {
        "in_features": 32,
        "intermediate_features": 16,
        "out_features": 8,
        "output_size": (4, 4),
    }
    assert "in_features=32" in pool2d.extra_repr()

    # AdaptiveLearnedUnpool2d
    unpool2d = AdaptiveLearnedUnpool2d(
        in_features=8, intermediate_features=16, out_features=32, output_size=(16, 16)
    )
    assert unpool2d.get_config() == {
        "in_features": 8,
        "intermediate_features": 16,
        "out_features": 32,
        "output_size": (16, 16),
    }
    assert "in_features=8" in unpool2d.extra_repr()

    # CausalConv1d
    conv1d = CausalConv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, dilation=2)
    assert conv1d.get_config() == {
        "in_channels": 8,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "dilation": 2,
        "groups": 1,
        "bias": True,
    }
    assert "in_channels=8" in conv1d.extra_repr()
    assert "padding=4" in conv1d.extra_repr()

    # CausalAdaptiveLearnedPool
    cpool = CausalAdaptiveLearnedPool(
        in_features=16, intermediate_features=32, out_features=8, output_size=4, temporal_dim=1
    )
    assert cpool.get_config() == {
        "in_features": 16,
        "intermediate_features": 32,
        "out_features": 8,
        "output_size": 4,
        "temporal_dim": 1,
    }
    assert "in_features=16" in cpool.extra_repr()
    assert "temporal_dim=1" in cpool.extra_repr()


def test_causal_adaptive_pool_custom_temporal_dim_3d() -> None:
    # Input format (B, T, C) with temporal_dim=1
    pooler = CausalAdaptiveLearnedPool(
        in_features=16, intermediate_features=32, out_features=8, output_size=4, temporal_dim=1
    )
    x_btc = torch.randn(2, 10, 16)
    out_btc = pooler(x_btc)
    assert out_btc.shape == (2, 4, 8)

    # Input format (B, C, T) with negative temporal_dim=-1
    pooler_neg = CausalAdaptiveLearnedPool(
        in_features=16, intermediate_features=32, out_features=8, output_size=4, temporal_dim=-1
    )
    x_bct = torch.randn(2, 16, 10)
    out_bct = pooler_neg(x_bct)
    assert out_bct.shape == (2, 8, 4)


def test_causal_adaptive_pool_custom_temporal_dim_5d() -> None:
    # 5D input with temporal_dim=1 (B, T, C, H, W)
    pooler = CausalAdaptiveLearnedPool(
        in_features=16, intermediate_features=32, out_features=8, output_size=4, temporal_dim=1
    )
    x_btchw = torch.randn(2, 10, 16, 6, 6)
    out_btchw = pooler(x_btchw)
    assert out_btchw.shape == (2, 4, 8, 6, 6)


def test_causal_adaptive_pool_invalid_temporal_dim() -> None:
    pooler = CausalAdaptiveLearnedPool(
        in_features=8, intermediate_features=16, out_features=4, output_size=4, temporal_dim=5
    )
    x_3d = torch.randn(2, 8, 10)
    with pytest.raises(IndexError, match="temporal_dim 5 out of bounds"):
        pooler(x_3d)


def test_mobilenet_depthwise_separable_structure() -> None:
    # 1. LearnedWeightedGAP with kernel_size > 1
    gap = LearnedWeightedGAP(in_features=32, kernel_size=3, num_output=4)
    assert isinstance(gap.weighter_conv, nn.Sequential)
    dw_conv, pw_conv = gap.weighter_conv[0], gap.weighter_conv[1]
    assert isinstance(dw_conv, nn.Conv2d)
    assert isinstance(pw_conv, nn.Conv2d)
    assert dw_conv.groups == 32
    assert dw_conv.kernel_size == (3, 3)
    assert pw_conv.kernel_size == (1, 1)

    # 2. AdaptiveLearnedPool2d
    pool2d = AdaptiveLearnedPool2d(
        in_features=16, intermediate_features=32, out_features=64, output_size=(7, 7)
    )
    # input_conv: depthwise + pointwise
    assert pool2d.input_conv[0].groups == 16
    assert pool2d.input_conv[0].kernel_size == (3, 3)
    assert pool2d.input_conv[1].kernel_size == (1, 1)
    # downsampling_core: depthwise + pointwise
    assert pool2d.downsampling_core[0].groups == 32
    assert pool2d.downsampling_core[1].kernel_size == (1, 1)
    # output_conv: depthwise + pointwise
    assert pool2d.output_conv[0].groups == 32 + 16
    assert pool2d.output_conv[0].kernel_size == (3, 3)
    assert pool2d.output_conv[1].kernel_size == (1, 1)

    # 3. AdaptiveLearnedUnpool2d
    unpool2d = AdaptiveLearnedUnpool2d(
        in_features=64, intermediate_features=32, out_features=16, output_size=(28, 28)
    )
    # input_conv: depthwise + pointwise
    assert unpool2d.input_conv[0].groups == 64
    assert unpool2d.input_conv[0].kernel_size == (3, 3)
    assert unpool2d.input_conv[1].kernel_size == (1, 1)
    # upsampling_core: depthwise transposed + pointwise
    assert unpool2d.upsampling_core[0].groups == 32
    assert unpool2d.upsampling_core[1].kernel_size == (1, 1)
    # output_conv: depthwise + pointwise
    assert unpool2d.output_conv[0].groups == 32 + 64
    assert unpool2d.output_conv[0].kernel_size == (3, 3)
    assert unpool2d.output_conv[1].kernel_size == (1, 1)

    # 4. CausalAdaptiveLearnedPool
    cpool = CausalAdaptiveLearnedPool(
        in_features=16, intermediate_features=32, out_features=64, output_size=4
    )
    # input_conv: depthwise causal + pointwise
    assert cpool.input_conv[0].groups == 16
    assert cpool.input_conv[0].kernel_size == 3
    assert cpool.input_conv[1].kernel_size == (1,)
    # downsampling_core: depthwise 1d + pointwise
    assert cpool.downsampling_core[0].groups == 32
    assert cpool.downsampling_core[1].kernel_size == (1,)
    # output_conv: depthwise causal + pointwise
    assert cpool.output_conv[0].groups == 32 + 16
    assert cpool.output_conv[0].kernel_size == 3
    assert cpool.output_conv[1].kernel_size == (1,)


def test_mobilenet_parameter_efficiency() -> None:
    # Verify parameter efficiency of depthwise-separable output conv vs dense conv
    in_feat, inter_feat, out_feat = 64, 128, 256
    pool2d = AdaptiveLearnedPool2d(
        in_features=in_feat,
        intermediate_features=inter_feat,
        out_features=out_feat,
        output_size=(7, 7),
    )

    dw_pw_params = sum(p.numel() for p in pool2d.output_conv.parameters())
    # Standard dense conv: (inter_feat + in_feat) * out_feat * 3 * 3 + biases
    dense_params = (inter_feat + in_feat) * out_feat * 9 + out_feat
    assert dw_pw_params < dense_params
    # Reduction is roughly a factor of 8-9x for the 3x3 stage
    assert dw_pw_params < dense_params * 0.25
