import pytest
import torch

from src.models.components.poolers import (
    AdaptiveLearnedPool2d,
    AdaptiveLearnedUnpool2d,
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


def test_learned_weighted_gap_gradient_flow() -> None:
    pooler = LearnedWeightedGAP(
        in_features=16, kernel_size=1, num_output=1, concat_gap=True
    )
    x = torch.randn(2, 16, 8, 8, requires_grad=True)

    out = pooler(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert pooler.weighter_conv.weight.grad is not None


def test_learned_weighted_gap_config() -> None:
    pooler = LearnedWeightedGAP(
        in_features=64, kernel_size=3, num_output=2, concat_gap=False
    )
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
    pooler = _make_adaptive_pooler(
        in_features=16, intermediate_features=16, output_size=(4, 4)
    )
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
    pooler = _make_adaptive_unpooler(
        in_features=16, intermediate_features=16, output_size=(8, 8)
    )
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
