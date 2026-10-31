"""The spatial operators: convolution, and the pooling windows over its output.

Both are one MLX kernel here rather than a piece of Metal source, so what the
tests check is the part that is ours: the axis order ATen uses against the one
MLX wants, and the geometry - stride, padding, dilation, groups - surviving that
translation unchanged.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch
import torch.nn.functional as F

from aotolith import compile_callable, compile_module
from aotolith.capture import capture_fx

# One primitive per activation rank, so a parametrized case can pick the
# spatial axis count without rewriting the callable.
_CONV = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}


def _run(call, *values: torch.Tensor):
    expected = call(*values)
    compiled = compile_callable(call, values)
    actual = compiled.plan(*(mx.array(value.numpy()) for value in values))[0]
    mx.eval(actual)
    return compiled, actual, expected


def _assert_close(actual, expected, atol: float = 1e-4) -> None:
    reference = mx.array(expected.detach().float().numpy())
    assert actual.shape == tuple(reference.shape)
    assert bool(mx.allclose(actual, reference, atol=atol, rtol=1e-4))


@pytest.mark.parametrize(
    ("shape", "weight_shape", "stride", "padding", "dilation", "groups"),
    [
        pytest.param((2, 3, 11), (5, 3, 4), 2, 1, 1, 1, id="1d"),
        pytest.param((2, 4, 9, 9), (8, 4, 3, 3), 1, 0, 1, 1, id="2d"),
        pytest.param((2, 4, 9, 9), (8, 4, 3, 3), 2, 1, 1, 1, id="2d-strided"),
        pytest.param((2, 4, 9, 9), (8, 4, 3, 3), 1, 2, 2, 1, id="2d-dilated"),
        pytest.param((2, 6, 8, 8), (9, 2, 3, 3), 1, 1, 1, 3, id="2d-grouped"),
        pytest.param((2, 4, 7, 7), (4, 1, 3, 3), 1, 1, 1, 4, id="2d-depthwise"),
        pytest.param((1, 2, 6, 6, 6), (3, 2, 3, 3, 3), 1, 1, 1, 1, id="3d"),
    ],
)
def test_convolution_matches_pytorch(
    shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> None:
    torch.manual_seed(0)
    value = torch.randn(*shape)
    weight = torch.randn(*weight_shape)
    bias = torch.randn(weight_shape[0])
    operation = _CONV[len(shape)]

    def call(input, weight, bias):
        return operation(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    compiled, actual, expected = _run(call, value, weight, bias)

    _assert_close(actual, expected)
    assert [op.op for op in compiled.input_graph.ops] == ["conv"]


def test_convolution_records_the_geometry_as_typed_attributes() -> None:
    value = torch.randn(1, 2, 8, 8)
    weight = torch.randn(3, 2, 3, 3)

    graph = capture_fx(
        lambda input, weight: F.conv2d(input, weight, stride=2, padding=1),
        (value, weight),
    )
    (conv,) = graph.ops
    assert conv.op == "conv"
    assert conv.attrs == {
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
    }


def test_a_convolution_module_binds_its_weight_and_bias() -> None:
    class Patch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.conv(value))

    torch.manual_seed(1)
    module = Patch().eval()
    value = torch.randn(2, 3, 6, 6)
    with torch.no_grad():
        expected = module(value)

    compiled = compile_module(module, (value,))
    actual = compiled(value)[0]
    mx.eval(actual)

    assert compiled.state_names == ("conv.weight", "conv.bias")
    _assert_close(actual, expected)


def test_a_transposed_convolution_is_refused_by_name() -> None:
    value = torch.randn(1, 2, 5, 5)
    weight = torch.randn(2, 3, 3, 3)

    with pytest.raises(NotImplementedError, match="transposed convolution"):
        capture_fx(
            lambda input, weight: F.conv_transpose2d(input, weight, stride=2),
            (value, weight),
        )
