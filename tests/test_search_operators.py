"""The search and bucketing operators: placement and its counting variant.

``searchsorted`` answers where a value belongs in a sorted list, ``bucketize``
asks the same question with the arguments reversed, and ``histc`` counts the
answers into uniform bins. They are checked end to end because the interesting
parts are the ATen bin edges - the exact maximum belongs to the last bin, a
value outside the range is not counted at all - and those live in the lowering.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith.backends import AppleMetalBackend
from aotolith.capture import capture_fx
from aotolith.passes import rewrite


def _run(call, *values: torch.Tensor):
    lowered = rewrite(capture_fx(call, values))
    plan = AppleMetalBackend().compile(lowered)
    arrays = tuple(mx.array(value.numpy()) for value in values)
    actual = plan(*arrays)[0]
    mx.eval(actual)
    return lowered, actual


@pytest.mark.parametrize("right", [False, True])
def test_searchsorted_matches_pytorch(right: bool) -> None:
    torch.manual_seed(11)
    boundaries = torch.randn(6).sort().values
    values = torch.randn(5) * 2.0

    lowered, actual = _run(
        lambda a, b: torch.searchsorted(a, b, right=right), boundaries, values
    )

    expected = torch.searchsorted(boundaries, values, right=right)
    assert [op.op for op in lowered.ops] == ["searchsorted"]
    assert lowered.ops[0].attrs == {"right": right}
    assert actual.dtype == mx.int64
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))


def test_bucketize_is_searchsorted_with_the_arguments_reversed() -> None:
    torch.manual_seed(13)
    boundaries = torch.randn(7).sort().values
    values = torch.randn(4) * 2.0

    _, actual = _run(lambda a, b: torch.bucketize(b, a), boundaries, values)

    expected = torch.bucketize(values, boundaries)
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))


def test_searchsorted_places_a_scalar() -> None:
    boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0])

    lowered, actual = _run(lambda a: torch.searchsorted(a, 1.5), boundaries)

    # The scalar becomes a constant so the search keeps one shape rule.
    assert [op.op for op in lowered.ops] == ["scalar_tensor", "searchsorted"]
    assert actual.shape == ()
    assert bool(
        mx.array_equal(actual, mx.array(torch.searchsorted(boundaries, 1.5).numpy()))
    )


@pytest.mark.parametrize(
    "values, bins, low, high",
    [
        pytest.param([0.2, 0.4, 0.6, 0.8], 4, 0.0, 1.0, id="quarters"),
        pytest.param([-1.0, 0.0, 0.5, 2.0], 4, 0.0, 1.0, id="outside_range"),
        pytest.param([0.0, 1.25, 2.5, 3.75, 5.0], 4, 0.0, 5.0, id="bin_edges"),
        pytest.param([0.3, 0.3, -4.0, 4.0], 2, 0.0, 10.0, id="one_bin"),
    ],
)
def test_histc_matches_pytorch(values, bins, low, high) -> None:
    tensor = torch.tensor(values)

    lowered, actual = _run(
        lambda t: torch.histc(t, bins=bins, min=low, max=high), tensor
    )

    expected = torch.histc(tensor, bins=bins, min=low, max=high)
    assert [op.op for op in lowered.ops] == ["histc"]
    assert actual.shape == expected.shape
    assert actual.dtype == mx.float32
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))


def test_histc_derives_the_range_from_the_data() -> None:
    torch.manual_seed(17)
    tensor = torch.randn(12) * 3.0

    _, actual = _run(lambda t: torch.histc(t, bins=5), tensor)

    expected = torch.histc(tensor, bins=5)
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))


def test_histc_widens_a_constant_range() -> None:
    # ATen takes the range from the data when ``min == max``, and if the data is
    # a single value it widens the range by one on each side instead of
    # dividing by zero.
    tensor = torch.full((4,), 3.0)

    _, actual = _run(lambda t: torch.histc(t, bins=4), tensor)

    expected = torch.histc(tensor, bins=4)
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))
    assert bool(mx.array_equal(actual, mx.array([0.0, 0.0, 4.0, 0.0])))


def test_histc_drops_nan_and_out_of_range_values() -> None:
    nan = float("nan")
    tensor = torch.tensor([nan, 1.0, 2.0, nan, -100.0, 100.0])

    _, actual = _run(lambda t: torch.histc(t, bins=4, min=0.0, max=3.0), tensor)

    expected = torch.histc(tensor, bins=4, min=0.0, max=3.0)
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))
    assert bool(mx.array_equal(actual[1:3], mx.array([1.0, 1.0])))


def test_histc_flattens_a_ranked_input() -> None:
    torch.manual_seed(19)
    tensor = torch.randn(4, 5)

    _, actual = _run(lambda t: torch.histc(t, bins=3, min=-1.0, max=1.0), tensor)

    expected = torch.histc(tensor, bins=3, min=-1.0, max=1.0)
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))
