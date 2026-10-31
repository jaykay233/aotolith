"""The scan and selection reductions that stay on MLX.

``cumsum``/``cumprod`` were already here; ``logcumsumexp`` joins them because
neither Metal nor MLX has a log-domain scan, and the rest of the selection
reductions (``sort``, ``topk``, ``argmax``) already run on MLX. These are
checked end to end - capture, lower, select, execute - because the interesting
failures are in the lowering, not in the arithmetic.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith.backends import AppleMetalBackend
from aotolith.capture import capture_fx
from aotolith.passes import rewrite


def _run(call, *values: torch.Tensor):
    lowered, actual = _run_all(call, *values)
    return lowered, actual[0]


def _run_all(call, *values: torch.Tensor):
    lowered = rewrite(capture_fx(call, values))
    plan = AppleMetalBackend().compile(lowered)
    arrays = tuple(mx.array(value.numpy()) for value in values)
    actual = plan(*arrays)
    actual = actual if isinstance(actual, tuple) else (actual,)
    for array in actual:
        mx.eval(array)
    return lowered, actual


@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((4, 6), 1, id="rank2"),
        pytest.param((3, 5, 7), 2, id="rank3"),
        pytest.param((6, 7), 0, id="leading_axis"),
        pytest.param((5,), 0, id="vector"),
    ],
)
def test_logcumsumexp_matches_pytorch(shape, dim) -> None:
    torch.manual_seed(5)
    value = torch.randn(*shape) * 4.0

    lowered, actual = _run(lambda v: torch.logcumsumexp(v, dim), value)

    expected = mx.array(torch.logcumsumexp(value, dim).numpy())
    assert actual.dtype == expected.dtype
    assert bool(mx.allclose(actual, expected, atol=2e-6, rtol=2e-6))
    # Like cumsum, this one is a scan MLX runs rather than a Metal kernel.
    assert [op.op for op in lowered.ops] == ["logcumsumexp"]


def test_logcumsumexp_keeps_the_infinite_and_nan_cases() -> None:
    # The shift that makes the scan stable is exactly where an infinite operand
    # would produce ``-inf - -inf``: ATen returns -inf for a run of -inf and inf
    # for a run of inf, and a NaN poisons everything after it.
    inf = float("inf")
    nan = float("nan")
    value = torch.tensor(
        [
            [-inf, -inf, 0.0],
            [1.0, -inf, 3.0],
            [-100.0, 100.0, 0.0],
            [inf, 0.0, 1.0],
            [nan, 1.0, 2.0],
        ]
    )

    _, actual = _run(lambda v: torch.logcumsumexp(v, 1), value)

    expected = mx.array(torch.logcumsumexp(value, 1).numpy())
    assert bool(mx.array_equal(actual, expected, equal_nan=True))


def test_logcumsumexp_runs_in_half_precision() -> None:
    value = torch.randn(3, 8).to(torch.bfloat16) * 3.0
    array = mx.array(value.float().numpy()).astype(mx.bfloat16)

    lowered = rewrite(capture_fx(lambda v: torch.logcumsumexp(v, 1), (value,)))
    plan = AppleMetalBackend().compile(lowered)
    actual = plan(array)[0]
    mx.eval(actual)
    expected = mx.array(torch.logcumsumexp(value, 1).float().numpy()).astype(
        mx.bfloat16
    )

    assert actual.dtype == mx.bfloat16
    assert bool(mx.allclose(actual, expected, atol=2e-2, rtol=2e-2))


@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((4, 7), 1, id="odd_length"),
        pytest.param((4, 6), 1, id="even_length"),
        pytest.param((3, 4, 5), 1, id="rank3"),
        pytest.param((6, 7), 0, id="leading_axis"),
        pytest.param((5,), 0, id="vector"),
    ],
)
def test_median_matches_pytorch(shape, dim) -> None:
    torch.manual_seed(5)
    value = torch.randn(*shape)

    lowered, actual = _run(lambda v: torch.median(v, dim).values, value)

    expected = mx.array(torch.median(value, dim).values.numpy())
    assert actual.dtype == expected.dtype
    assert bool(mx.array_equal(actual, expected))
    assert [op.op for op in lowered.ops] == ["median"]


def test_median_reduces_every_axis() -> None:
    torch.manual_seed(11)
    value = torch.randn(3, 4, 5)

    lowered, actual = _run(torch.median, value)

    assert actual.shape == ()
    assert bool(mx.array_equal(actual, mx.array(torch.median(value).numpy())))
    # A full reduction names every axis and picks the lower median of them.
    (op,) = lowered.ops
    assert op.op == "median"
    assert op.attrs["dims"] == (0, 1, 2)


def test_median_reports_the_selected_position() -> None:
    # ATen keeps duplicate values, so the index of a repeated median is the
    # first position the sort produces, and the gather has to follow it.
    value = torch.tensor([[3.0, 1.0, 3.0, 1.0], [5.0, 4.0, 4.0, 5.0]])

    _, actual = _run_all(lambda v: torch.median(v, 1), value)

    values, indices = torch.median(value, 1)
    assert bool(mx.array_equal(actual[0], mx.array(values.numpy())))
    assert bool(mx.array_equal(actual[1], mx.array(indices.numpy())))
    assert actual[1].dtype == mx.int64


def test_median_keeps_keepdim() -> None:
    torch.manual_seed(7)
    value = torch.randn(3, 4, 5)

    _, actual = _run(lambda v: torch.median(v, 2, keepdim=True).values, value)

    expected = torch.median(value, 2, keepdim=True).values
    assert actual.shape == expected.shape
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))


def test_median_propagates_nan_where_nanmedian_ignores_it() -> None:
    nan = float("nan")
    value = torch.tensor(
        [
            [1.0, 2.0, nan, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [nan, nan, nan, 1.0],
        ]
    )

    _, plain = _run_all(
        lambda v: (
            torch.median(v, 1).values,
            torch.median(v, 1).indices,
            torch.median(v),
        ),
        value,
    )
    _, ignoring = _run_all(
        lambda v: (
            torch.nanmedian(v, 1).values,
            torch.nanmedian(v, 1).indices,
            torch.nanmedian(v),
        ),
        value,
    )

    assert bool(
        mx.allclose(
            plain[0],
            mx.array(torch.median(value, 1).values.numpy()),
            equal_nan=True,
        )
    )
    # ATen reports the first NaN of a row as the position of the median it
    # propagates, so the index tracks the value rather than the ordering.
    assert bool(
        mx.array_equal(plain[1], mx.array(torch.median(value, 1).indices.numpy()))
    )
    assert bool(mx.allclose(plain[2], mx.array(nan), equal_nan=True))
    assert bool(
        mx.array_equal(ignoring[0], mx.array(torch.nanmedian(value, 1).values.numpy()))
    )
    assert bool(
        mx.array_equal(ignoring[1], mx.array(torch.nanmedian(value, 1).indices.numpy()))
    )
    assert bool(
        mx.allclose(
            ignoring[2], mx.array(torch.nanmedian(value).numpy()), equal_nan=True
        )
    )


def test_nanmedian_answers_nan_for_an_all_nan_row() -> None:
    nan = float("nan")
    value = torch.tensor([[1.0, 3.0, 2.0], [nan, nan, nan]])

    _, actual = _run_all(lambda v: torch.nanmedian(v, 1), value)

    expected = torch.nanmedian(value, 1)
    assert bool(
        mx.allclose(actual[0], mx.array(expected.values.numpy()), equal_nan=True)
    )
    assert bool(mx.array_equal(actual[1], mx.array(expected.indices.numpy())))


def test_median_runs_in_half_precision() -> None:
    value = torch.randn(3, 8).to(torch.bfloat16)
    array = mx.array(value.float().numpy()).astype(mx.bfloat16)

    lowered = rewrite(capture_fx(lambda v: torch.median(v, 1).values, (value,)))
    plan = AppleMetalBackend().compile(lowered)
    actual = plan(array)[0]
    mx.eval(actual)
    expected = mx.array(torch.median(value, 1).values.float().numpy()).astype(
        mx.bfloat16
    )

    assert actual.dtype == mx.bfloat16
    assert bool(mx.array_equal(actual, expected))
