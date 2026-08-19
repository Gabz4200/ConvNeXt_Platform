import pytest
import torch

from src.models.components.poolers import LearnedWeightedGAP


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
    pooler = LearnedWeightedGAP(in_features=16, kernel_size=1, num_output=1, concat_gap=True)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)

    out = pooler(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert pooler.weighter_conv.weight.grad is not None


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

