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
        # MLX groups 1D and 2D convolutions only, so a grouped 3D convolution is
        # split into one plain convolution per group.
        pytest.param((1, 4, 6, 6, 6), (6, 2, 3, 3, 3), 1, 1, 1, 2, id="3d-grouped"),
        pytest.param((1, 4, 6, 6, 6), (8, 1, 3, 3, 3), 1, 1, 1, 4, id="3d-depthwise"),
        pytest.param(
            (2, 6, 7, 5, 7), (9, 2, 2, 3, 2), 2, 1, 2, 3, id="3d-grouped-strided"
        ),
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
        "transposed": False,
        "output_padding": (0, 0),
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


_CONV_TRANSPOSE = {3: F.conv_transpose1d, 4: F.conv_transpose2d, 5: F.conv_transpose3d}


@pytest.mark.parametrize(
    ("shape", "weight_shape", "stride", "padding", "dilation", "output_padding"),
    [
        pytest.param((2, 3, 7), (3, 5, 4), 2, 1, 1, 0, id="1d"),
        pytest.param((2, 3, 6, 6), (3, 5, 3, 3), 2, 1, 1, 0, id="2d"),
        pytest.param((2, 3, 6, 6), (3, 5, 3, 3), 2, 1, 1, 1, id="2d-output-padded"),
        pytest.param((2, 3, 6, 6), (3, 5, 3, 3), 2, 1, 2, 1, id="2d-dilated"),
        # Padding wider than the kernel crops the output, which is a forward
        # convolution with a negative pad and so the case the pre-image crop
        # exists for.
        pytest.param((2, 3, 6, 6), (3, 5, 3, 3), 2, 4, 1, 0, id="2d-cropped"),
        pytest.param((2, 3, 5, 5, 5), (3, 4, 2, 2, 2), 2, 1, 1, 0, id="3d"),
        pytest.param((2, 3, 5, 5, 5), (3, 4, 2, 2, 2), 2, 1, 2, 1, id="3d-dilated"),
    ],
)
def test_a_transposed_convolution_matches_pytorch(
    shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: int,
    padding: int,
    dilation: int,
    output_padding: int,
) -> None:
    # Torch's transposed convolution is the MLX backward pass, which is only
    # accurate to about 1e-2 in float32; the backend rebuilds it as a forward
    # convolution over a zero-set input so this stays at the same 1e-4 as the
    # forward direction.
    torch.manual_seed(0)
    value = torch.randn(*shape)
    weight = torch.randn(*weight_shape)
    bias = torch.randn(weight_shape[1])
    operation = _CONV_TRANSPOSE[len(shape)]

    def call(input, weight, bias):
        return operation(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
        )

    compiled, actual, expected = _run(call, value, weight, bias)

    _assert_close(actual, expected)
    assert [op.op for op in compiled.input_graph.ops] == ["conv"]
    assert compiled.input_graph.ops[0].attrs["transposed"] is True


@pytest.mark.parametrize(
    ("shape", "weight_shape", "groups"),
    [
        pytest.param((2, 4, 6), (4, 2, 3), 2, id="1d"),
        pytest.param((2, 4, 6, 6), (4, 2, 3, 3), 2, id="2d"),
        pytest.param((2, 4, 6, 6, 6), (4, 1, 3, 3, 3), 4, id="3d-depthwise"),
    ],
)
def test_a_grouped_transposed_convolution_matches_pytorch(
    shape: tuple[int, ...], weight_shape: tuple[int, ...], groups: int
) -> None:
    # Torch orders the output channels group-major, which is the order the
    # per-group pieces come out in before the concatenate.
    torch.manual_seed(2)
    value = torch.randn(*shape)
    weight = torch.randn(*weight_shape)
    operation = _CONV_TRANSPOSE[len(shape)]

    def call(input, weight):
        return operation(
            input, weight, stride=2, padding=1, groups=groups, output_padding=1
        )

    compiled, actual, expected = _run(call, value, weight)

    _assert_close(actual, expected)
    assert compiled.input_graph.ops[0].attrs["groups"] == groups


def test_a_transposed_convolution_module_binds_its_weight_and_bias() -> None:
    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = torch.nn.ConvTranspose2d(3, 4, 3, stride=2, padding=1)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.up(value))

    torch.manual_seed(3)
    module = Decoder().eval()
    value = torch.randn(2, 3, 5, 5)
    with torch.no_grad():
        expected = module(value)

    compiled = compile_module(module, (value,))
    actual = compiled(mx.array(value.numpy()))[0]
    mx.eval(actual)

    assert compiled.state_names == ("up.weight", "up.bias")
    _assert_close(actual, expected)


def test_transposed_convolution_records_the_flag_and_output_padding() -> None:
    value = torch.randn(1, 2, 5, 5)
    weight = torch.randn(2, 3, 3, 3)

    graph = capture_fx(
        lambda input, weight: F.conv_transpose2d(
            input, weight, stride=2, padding=1, output_padding=1
        ),
        (value, weight),
    )
    (conv,) = graph.ops
    assert conv.attrs == {
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "transposed": True,
        "output_padding": (1, 1),
    }


_MAX_POOL = {3: F.max_pool1d, 4: F.max_pool2d, 5: F.max_pool3d}
_AVG_POOL = {3: F.avg_pool1d, 4: F.avg_pool2d, 5: F.avg_pool3d}
_ADAPTIVE_MAX_POOL = {
    3: F.adaptive_max_pool1d,
    4: F.adaptive_max_pool2d,
    5: F.adaptive_max_pool3d,
}
_ADAPTIVE_AVG_POOL = {
    3: F.adaptive_avg_pool1d,
    4: F.adaptive_avg_pool2d,
    5: F.adaptive_avg_pool3d,
}

# One activation per spatial rank, so a parametrized case picks the rank of the
# pooling it exercises without rewriting the callable.
_POOL_SHAPES = {3: (2, 3, 12), 4: (2, 3, 9, 9), 5: (1, 2, 6, 6, 6)}


def _indices(call, value: torch.Tensor) -> tuple[mx.array, torch.Tensor, torch.Tensor]:
    expected_values, expected_indices = call(value)
    compiled = compile_callable(call, (value,))
    values, indices = compiled.plan(mx.array(value.numpy()))
    mx.eval(values, indices)
    assert indices.dtype == mx.int64
    return values, indices, (expected_values, expected_indices)


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation", "ceil_mode"),
    [
        pytest.param(2, None, 0, 1, False, id="windows"),
        pytest.param(3, 2, 1, 1, False, id="strided-padded"),
        pytest.param(2, 2, 0, 1, True, id="ceil"),
        pytest.param(2, 1, 1, 3, False, id="dilated"),
    ],
)
def test_max_pool_matches_pytorch(
    rank: int,
    kernel: int,
    stride: int | None,
    padding: int,
    dilation: int,
    ceil_mode: bool,
) -> None:
    torch.manual_seed(0)
    value = torch.randn(*_POOL_SHAPES[rank])
    operation = _MAX_POOL[rank]

    def call(input):
        return operation(input, kernel, stride, padding, dilation, ceil_mode)

    _, actual, expected = _run(call, value)

    _assert_close(actual, expected)


@pytest.mark.parametrize("rank", [3, 4, 5])
def test_max_pool_reports_the_indices_pytorch_reports(rank: int) -> None:
    torch.manual_seed(0)
    value = torch.randn(*_POOL_SHAPES[rank])
    operation = _MAX_POOL[rank]

    values, indices, (expected_values, expected_indices) = _indices(
        lambda input: operation(input, 2, 2, return_indices=True), value
    )

    _assert_close(values, expected_values)
    assert bool(mx.array_equal(indices, mx.array(expected_indices.numpy())))


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "ceil_mode", "count_include_pad"),
    [
        pytest.param(2, None, 0, False, True, id="windows"),
        pytest.param(3, 2, 1, False, True, id="strided-padded"),
        pytest.param(3, 2, 1, False, False, id="padding-excluded"),
        pytest.param(2, 2, 0, True, True, id="ceil"),
        pytest.param(3, 2, 1, True, True, id="ceil-padded"),
    ],
)
def test_average_pool_matches_pytorch(
    rank: int,
    kernel: int,
    stride: int | None,
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
) -> None:
    torch.manual_seed(0)
    value = torch.randn(*_POOL_SHAPES[rank])
    operation = _AVG_POOL[rank]

    def call(input):
        return operation(input, kernel, stride, padding, ceil_mode, count_include_pad)

    _, actual, expected = _run(call, value)

    _assert_close(actual, expected)


def test_average_pool_honours_a_divisor_override() -> None:
    torch.manual_seed(0)
    value = torch.randn(1, 2, 7, 7)

    _, actual, expected = _run(
        lambda input: F.avg_pool2d(input, 3, 2, 1, False, True, 5), value
    )

    _assert_close(actual, expected)


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("output_size", [1, 3])
def test_adaptive_max_pool_matches_pytorch(rank: int, output_size: int) -> None:
    torch.manual_seed(0)
    value = torch.randn(*_POOL_SHAPES[rank])
    operation = _ADAPTIVE_MAX_POOL[rank]

    _, actual, expected = _run(lambda input: operation(input, output_size), value)

    _assert_close(actual, expected)


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("output_size", [1, 2, 4])
def test_adaptive_average_pool_matches_pytorch(rank: int, output_size: int) -> None:
    torch.manual_seed(0)
    value = torch.randn(*_POOL_SHAPES[rank])
    operation = _ADAPTIVE_AVG_POOL[rank]

    _, actual, expected = _run(lambda input: operation(input, output_size), value)

    _assert_close(actual, expected)


def test_adaptive_max_pool_reports_the_indices_pytorch_reports() -> None:
    torch.manual_seed(0)
    value = torch.randn(1, 2, 7, 7)

    values, indices, (expected_values, expected_indices) = _indices(
        lambda input: F.adaptive_max_pool2d(input, (2, 3), return_indices=True), value
    )

    _assert_close(values, expected_values)
    assert bool(mx.array_equal(indices, mx.array(expected_indices.numpy())))


def test_adaptive_pool_over_a_recipe_with_ragged_windows() -> None:
    # Three outputs for seven inputs split into windows of three, two and two -
    # the case a fixed window cannot express.
    torch.manual_seed(0)
    value = torch.randn(1, 2, 7, 7)

    _, actual, expected = _run(
        lambda input: F.adaptive_avg_pool2d(input, (2, 3)), value
    )

    _assert_close(actual, expected)


def test_pooling_records_the_geometry_as_typed_attributes() -> None:
    value = torch.randn(1, 2, 8, 8)

    graph = capture_fx(
        lambda input: F.avg_pool2d(
            input, 3, 2, 1, ceil_mode=True, count_include_pad=False
        ),
        (value,),
    )
    (pool,) = graph.ops
    assert pool.op == "pool"
    assert pool.attrs == {
        "mode": "avg",
        "kernel": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": None,
    }


def test_a_pooling_module_runs_inside_a_network() -> None:
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.pool = torch.nn.MaxPool2d(2)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.pool(torch.relu(self.conv(value)))

    torch.manual_seed(0)
    module = Net().eval()
    value = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        expected = module(value)

    compiled = compile_module(module, (value,))
    actual = compiled(value)[0]
    mx.eval(actual)

    assert [op.op for op in compiled.input_graph.ops] == ["conv", "relu", "pool"]
    _assert_close(actual, expected)


def test_a_half_precision_pooling_reduces_in_float32() -> None:
    value = torch.randn(2, 4, 10, 10).half()

    compiled, actual, expected = _run(lambda input: F.avg_pool2d(input, 3, 1, 1), value)

    assert actual.dtype == mx.float16
    _assert_close(actual, expected)


def test_a_maximum_pooling_round_trips_through_its_unpooling() -> None:
    # Unpooling scatters the values back to the indices the pooling reported,
    # which is the one consumer that reads them.
    torch.manual_seed(0)
    value = torch.randn(1, 2, 6, 6)
    expected_values, expected_indices = F.max_pool2d_with_indices(value, 2, 2)

    def call(values, indices):
        return F.max_unpool2d(values, indices, 2)

    compiled = compile_callable(call, (expected_values, expected_indices))
    actual = compiled.plan(
        mx.array(expected_values.numpy()), mx.array(expected_indices.numpy())
    )[0]
    mx.eval(actual)

    _assert_close(actual, call(expected_values, expected_indices))


def _interpolate(**kwargs):
    return lambda input: F.interpolate(input, **kwargs)


@pytest.mark.parametrize(
    ("shape", "options"),
    [
        pytest.param((1, 2, 4, 4), {"size": (7, 7), "mode": "bilinear"}, id="bilinear"),
        pytest.param(
            (1, 2, 4, 4),
            {"size": (7, 5), "mode": "bilinear", "align_corners": True},
            id="bilinear-aligned",
        ),
        pytest.param(
            (1, 2, 5, 5), {"size": (3, 3), "mode": "bilinear"}, id="bilinear-down"
        ),
        pytest.param(
            (1, 2, 4, 4),
            {"scale_factor": 1.7, "mode": "bilinear"},
            id="bilinear-fraction",
        ),
        pytest.param((1, 2, 4, 4), {"size": (7, 7), "mode": "bicubic"}, id="bicubic"),
        pytest.param(
            (1, 2, 9, 9), {"scale_factor": 0.4, "mode": "bicubic"}, id="bicubic-down"
        ),
        pytest.param(
            (1, 2, 3, 3, 3), {"size": (5, 4, 5), "mode": "trilinear"}, id="trilinear"
        ),
        pytest.param(
            (1, 2, 3, 3, 3), {"size": (6, 6, 6), "mode": "nearest"}, id="nearest"
        ),
        pytest.param(
            (1, 2, 4, 4),
            {"scale_factor": 2.5, "mode": "nearest"},
            id="nearest-fraction",
        ),
        pytest.param((1, 2, 5, 5), {"size": (1, 1), "mode": "bilinear"}, id="collapse"),
        pytest.param((1, 2, 1, 1), {"size": (4, 4), "mode": "bilinear"}, id="from-one"),
    ],
)
def test_resize_matches_pytorch(shape: tuple[int, ...], options: dict) -> None:
    torch.manual_seed(0)
    value = torch.randn(*shape)

    compiled, actual, expected = _run(_interpolate(**options), value)

    assert [op.op for op in compiled.input_graph.ops] == ["resize"]
    _assert_close(actual, expected)


def test_resize_in_half_precision_round_trips_the_dtype() -> None:
    value = torch.randn(1, 2, 4, 4).half()

    _, actual, expected = _run(_interpolate(size=(9, 9), mode="bilinear"), value)

    assert actual.dtype == mx.float16
    _assert_close(actual, expected, atol=2e-3)


def test_resize_records_the_mode_and_the_source_scale() -> None:
    graph = capture_fx(
        lambda input: F.interpolate(
            input, scale_factor=2, mode="bilinear", align_corners=True
        ),
        (torch.randn(1, 2, 4, 4),),
    )
    (resize,) = graph.ops
    assert resize.op == "resize"
    assert resize.attrs == {
        "mode": "linear",
        "output_size": (8, 8),
        "align_corners": True,
        "source_scale": (0.5, 0.5),
    }


def test_a_fractional_scale_factor_keeps_the_scale_it_was_called_with() -> None:
    # The output extent is rounded, but the source coordinate uses 1/1.7, so
    # the recorded scale is the request's rather than 4/6.
    graph = capture_fx(
        lambda input: F.interpolate(input, scale_factor=1.7, mode="bilinear"),
        (torch.randn(1, 2, 4, 4),),
    )
    (resize,) = graph.ops
    assert resize.attrs["output_size"] == (6, 6)
    assert resize.attrs["source_scale"] == pytest.approx((1 / 1.7, 1 / 1.7))


def test_an_antialiased_resize_is_refused_by_name() -> None:
    value = torch.randn(1, 2, 8, 8)

    with pytest.raises(NotImplementedError, match="antialias"):
        capture_fx(
            lambda input: F.interpolate(
                input, size=(4, 4), mode="bilinear", antialias=True
            ),
            (value,),
        )
