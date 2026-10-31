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


@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((4, 6), 1, id="rank2"),
        pytest.param((3, 4, 5), 2, id="rank3"),
        pytest.param((3, 4), 0, id="leading_axis"),
        pytest.param((7,), 0, id="vector"),
    ],
)
def test_cummax_and_cummin_match_pytorch(shape, dim) -> None:
    torch.manual_seed(13)
    value = torch.randn(*shape)

    for scan in (torch.cummax, torch.cummin):
        _, actual = _run_all(lambda v: scan(v, dim), value)
        expected = scan(value, dim)
        assert bool(mx.array_equal(actual[0], mx.array(expected.values.numpy())))
        assert bool(mx.array_equal(actual[1], mx.array(expected.indices.numpy())))
        assert actual[1].dtype == mx.int64


def test_cummax_keeps_only_the_selection_that_is_read() -> None:
    value = torch.randn(3, 6)

    lowered, actual = _run(lambda v: torch.cummax(v, 1).values, value)

    assert [op.op for op in lowered.ops] == ["cummax"]
    assert bool(mx.array_equal(actual, mx.array(torch.cummax(value, 1).values.numpy())))


def test_cummax_lets_a_nan_own_the_rest_of_the_scan() -> None:
    # ATen ranks NaN above every value for ``cummax`` and below every value for
    # ``cummin``, so it takes the running extremum and the position with it.
    nan = float("nan")
    value = torch.tensor(
        [
            [1.0, -1.0, nan, 2.0, 3.0],
            [nan, 1.0, 2.0, 3.0, 4.0],
            [nan, nan, nan, nan, nan],
        ]
    )

    for scan in (torch.cummax, torch.cummin):
        _, actual = _run_all(lambda v: scan(v, 1), value)
        expected = scan(value, 1)
        assert bool(
            mx.allclose(actual[0], mx.array(expected.values.numpy()), equal_nan=True)
        )
        assert bool(mx.array_equal(actual[1], mx.array(expected.indices.numpy())))


def test_cummax_reports_the_latest_position_of_a_repeated_extremum() -> None:
    # Ties advance the position, which is why the running comparison is
    # inclusive rather than strict.
    value = torch.tensor([[3.0, 3.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])

    _, actual = _run_all(
        lambda v: (
            torch.cummax(v, 1).values,
            torch.cummax(v, 1).indices,
            torch.cummin(v, 1).values,
            torch.cummin(v, 1).indices,
        ),
        value,
    )

    maxima, max_positions, minima, min_positions = actual
    expected_max = torch.cummax(value, 1)
    expected_min = torch.cummin(value, 1)
    assert bool(mx.array_equal(maxima, mx.array(expected_max.values.numpy())))
    assert bool(mx.array_equal(max_positions, mx.array(expected_max.indices.numpy())))
    assert bool(mx.array_equal(minima, mx.array(expected_min.values.numpy())))
    assert bool(mx.array_equal(min_positions, mx.array(expected_min.indices.numpy())))


def test_cummax_runs_on_integers_and_half_precision() -> None:
    integers = torch.randint(-3, 4, (3, 5))
    lowered, actual = _run_all(lambda v: torch.cummax(v, 1), integers)
    expected = torch.cummax(integers, 1)
    assert [op.op for op in lowered.ops] == ["cummax", "cummax_indices"]
    assert actual[0].dtype == mx.int64
    assert bool(mx.array_equal(actual[0], mx.array(expected.values.numpy())))

    halves = torch.randn(3, 6).to(torch.bfloat16)
    array = mx.array(halves.float().numpy()).astype(mx.bfloat16)
    lowered = rewrite(capture_fx(lambda v: torch.cummin(v, 1).values, (halves,)))
    plan = AppleMetalBackend().compile(lowered)
    result = plan(array)[0]
    mx.eval(result)
    expected = torch.cummin(halves, 1).values
    assert result.dtype == mx.bfloat16
    assert bool(
        mx.array_equal(result, mx.array(expected.float().numpy()).astype(mx.bfloat16))
    )


# ``var_mean``/``std_mean`` are two reductions ATen fuses into one node; the
# frontend splits them back into the pair MLX already runs, sharing the
# normalized axes and the correction so the pair stays consistent.
VAR_MEAN_CASES = [
    pytest.param(lambda v: torch.var_mean(v, dim=1), id="dim"),
    pytest.param(lambda v: torch.var_mean(v), id="every_axis"),
    pytest.param(lambda v: torch.var_mean(v, dim=0, correction=0), id="population"),
    pytest.param(lambda v: torch.var_mean(v, dim=1, correction=2), id="correction"),
    pytest.param(lambda v: torch.var_mean(v, dim=-1, keepdim=True), id="keepdim"),
    pytest.param(lambda v: torch.var_mean(v, dim=(0, 2)), id="two_axes"),
]


@pytest.mark.parametrize("call", VAR_MEAN_CASES)
def test_var_mean_matches_pytorch(call) -> None:
    torch.manual_seed(23)
    value = torch.randn(4, 6, 3) * 3.0

    lowered, actual = _run_all(call, value)

    expected = call(value)
    assert [op.op for op in lowered.ops] == ["var", "mean"]
    assert actual[0].dtype == mx.float32
    assert bool(mx.allclose(actual[0], mx.array(expected[0].numpy()), atol=2e-6))
    assert bool(mx.allclose(actual[1], mx.array(expected[1].numpy()), atol=2e-6))


def test_var_mean_emits_only_the_member_the_graph_reads() -> None:
    # The split is per-use: a graph that throws the mean away must not keep
    # paying for the second reduction.
    torch.manual_seed(24)
    value = torch.randn(5, 7)

    lowered, actual = _run_all(lambda v: torch.var_mean(v, dim=1)[0], value)

    assert [op.op for op in lowered.ops] == ["var"]
    expected = torch.var_mean(value, dim=1)[0]
    assert bool(mx.allclose(actual[0], mx.array(expected.numpy()), atol=2e-6))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda v: torch.std_mean(v, dim=1), id="dim"),
        pytest.param(lambda v: torch.std_mean(v), id="every_axis"),
        pytest.param(lambda v: torch.std_mean(v, dim=0, correction=0), id="population"),
        pytest.param(lambda v: torch.std_mean(v, dim=1, keepdim=True), id="keepdim"),
    ],
)
def test_std_mean_matches_pytorch(call) -> None:
    torch.manual_seed(25)
    value = torch.randn(4, 6) * 2.0

    lowered, actual = _run_all(call, value)

    expected = call(value)
    assert [op.op for op in lowered.ops] == ["std", "mean"]
    assert bool(mx.allclose(actual[0], mx.array(expected[0].numpy()), atol=2e-6))
    assert bool(mx.allclose(actual[1], mx.array(expected[1].numpy()), atol=2e-6))


def test_var_mean_runs_in_half_precision() -> None:
    torch.manual_seed(26)
    halves = torch.randn(4, 6).to(torch.bfloat16)

    lowered = rewrite(capture_fx(lambda v: torch.var_mean(v, dim=1), (halves,)))
    plan = AppleMetalBackend().compile(lowered)
    array = mx.array(halves.float().numpy()).astype(mx.bfloat16)
    variance, mean = plan(array)
    mx.eval(variance, mean)

    expected = torch.var_mean(halves, dim=1)
    assert [op.op for op in lowered.ops] == ["var", "mean"]
    assert variance.dtype == mx.bfloat16
    assert mean.dtype == mx.bfloat16
    assert bool(
        mx.allclose(
            variance.astype(mx.float32),
            mx.array(expected[0].float().numpy()),
            atol=2e-2,
        )
    )
    assert bool(
        mx.allclose(
            mean.astype(mx.float32),
            mx.array(expected[1].float().numpy()),
            atol=2e-2,
        )
    )


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.0, float("inf")])
def test_dist_matches_pytorch(p) -> None:
    # ``dist`` is ``(left - right).norm(p)`` with the difference gone before the
    # norm, so the pair of operators is the whole implementation.
    torch.manual_seed(27)
    left = torch.randn(3, 8)
    right = torch.randn(3, 8)

    lowered, actual = _run_all(lambda a, b: torch.dist(a, b, p), left, right)

    expected = torch.dist(left, right, p)
    assert [op.op for op in lowered.ops] == ["sub", "vector_norm"]
    assert bool(mx.allclose(actual[0], mx.array(expected.numpy()), atol=1e-5))


def test_dist_reduces_the_whole_tensor_and_broadcasts_its_operands() -> None:
    # ``dist`` has no ``dim``: it always answers one scalar, so a batched pair
    # collapses to a rank-0 result instead of one value per row.
    torch.manual_seed(28)
    left = torch.randn(4, 5)
    row = torch.randn(1, 5)
    batched = torch.randn(3, 4, 5)

    _, broadcast = _run_all(lambda a, b: torch.dist(a, b), left, row)
    _, batched_actual = _run_all(lambda a, b: torch.dist(a, b), batched, left)

    assert broadcast[0].shape == ()
    assert bool(
        mx.allclose(broadcast[0], mx.array(torch.dist(left, row).numpy()), atol=1e-5)
    )
    assert batched_actual[0].shape == ()
    assert bool(
        mx.allclose(
            batched_actual[0], mx.array(torch.dist(batched, left).numpy()), atol=1e-5
        )
    )
