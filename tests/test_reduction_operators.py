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
    lowered = rewrite(capture_fx(call, values))
    plan = AppleMetalBackend().compile(lowered)
    arrays = tuple(mx.array(value.numpy()) for value in values)
    actual = plan(*arrays)[0]
    mx.eval(actual)
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
