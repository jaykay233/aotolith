"""The arithmetic, bitwise and rounding operators, end to end.

These are the operators the decomposition table leaves alone and Metal has no
one-liner for: ``fmax``/``fmin`` ignore a NaN operand, ``nextafter`` steps one
representable value, ``trunc_divide`` rounds toward zero rather than down, and
``gcd``/``lcm``/the shifts are integer-only. Each is checked twice - the numbers
match PyTorch, and the plan really did emit a Metal kernel - because a silent
fall back to MLX would keep a parity assertion green while the codegen it is
meant to cover went unexercised.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith import compile_callable
from aotolith.backends import AppleMetalBackend
from aotolith.capture import capture_fx
from aotolith.passes import rewrite


def _run(call, *values: torch.Tensor):
    compiled = compile_callable(call, values)
    actual = compiled.plan(*(mx.array(value.numpy()) for value in values))[0]
    mx.eval(actual)
    return compiled, actual


def _assert_native(compiled) -> None:
    assert any(selection.is_native for selection in compiled.plan.selections), (
        "the operator should have compiled to a Metal kernel"
    )


_BIT_VIEWS = {mx.float32: mx.int32, mx.float16: mx.int16, mx.bfloat16: mx.int16}


_MX_DTYPES = {torch.float16: mx.float16, torch.bfloat16: mx.bfloat16}


def _assert_exact(actual, expected: torch.Tensor) -> None:
    reference = mx.array(expected.numpy())
    assert actual.dtype == reference.dtype
    assert bool(mx.array_equal(actual, reference, equal_nan=True)), (
        f"{actual} != {reference}"
    )
    bits = _BIT_VIEWS.get(actual.dtype)
    if bits is None:
        return
    # ``array_equal`` treats the two zeros as equal; the operators here promise
    # the sign as well, so the zero lanes are compared as bit patterns.
    zero_lanes = reference == 0
    assert bool(
        mx.array_equal(
            mx.where(zero_lanes, mx.view(actual, bits), 0),
            mx.where(zero_lanes, mx.view(reference, bits), 0),
        )
    ), "the sign of a zero differs"


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda v: torch.fmax(v, v * 2.0), id="fmax"),
        pytest.param(lambda v: torch.fmin(v, v * 2.0), id="fmin"),
        pytest.param(lambda v: torch.nextafter(v, v * 0.5), id="nextafter"),
        pytest.param(lambda v: torch.trunc(v / 3.0), id="trunc"),
        pytest.param(
            lambda v: torch.div(v, 3.0, rounding_mode="trunc"), id="trunc_divide"
        ),
    ],
)
def test_the_float_family_matches_pytorch(call) -> None:
    torch.manual_seed(11)
    value = torch.randn(6, 9) * 2.0

    compiled, actual = _run(call, value)

    _assert_exact(actual, call(value))
    _assert_native(compiled)


def test_nextafter_walks_the_edges_of_the_format() -> None:
    # The inputs a bit-level step has to special-case: signed zeros, both
    # infinities and the largest finite magnitude.
    inf = float("inf")
    value = torch.tensor([[0.0, -0.0, inf, -inf], [1.0, -1.0, 3.5, -3.5]])
    target = torch.tensor([[-0.0, 0.0, 3.0, -3.0], [inf, -inf, -0.0, 0.0]])

    compiled, actual = _run(torch.nextafter, value, target)

    _assert_exact(actual, torch.nextafter(value, target))
    _assert_native(compiled)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(torch.fmax, id="fmax"),
        pytest.param(torch.fmin, id="fmin"),
    ],
)
def test_a_nan_operand_is_ignored(operation) -> None:
    # The difference from maximum/minimum, which is why they are separate ops:
    # a NaN on either side is skipped, and NaN only survives when both are NaN.
    nan = float("nan")
    value = torch.tensor([[nan, 1.0, nan], [2.5, -2.5, nan]])
    other = torch.tensor([[1.0, nan, nan], [nan, -2.5, nan]])

    compiled, actual = _run(operation, value, other)

    _assert_exact(actual, operation(value, other))
    _assert_native(compiled)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(torch.fmax, id="fmax"),
        pytest.param(torch.fmin, id="fmin"),
    ],
)
def test_a_zero_pair_keeps_the_sign_pytorch_reports(operation) -> None:
    # fmax(-0.0, +0.0) is +0.0 and fmin(-0.0, +0.0) is -0.0, so a plain
    # comparison is not enough: it would hand back whichever operand came first.
    value = torch.tensor([0.0, -0.0, 0.0, -0.0])
    other = -value

    compiled, actual = _run(operation, value, other)

    _assert_exact(actual, operation(value, other))
    _assert_native(compiled)


@pytest.mark.parametrize(
    "call, value, expected_ops",
    [
        pytest.param(
            lambda v: torch.div(v, 3, rounding_mode="trunc"),
            torch.tensor([[-7, -1, 0, 1, 7], [8, -8, 9, -9, 10]], dtype=torch.int64),
            ["trunc_divide"],
            id="trunc",
        ),
        pytest.param(
            lambda v: torch.div(v, 3, rounding_mode="floor"),
            torch.tensor([[-7, -1, 0, 1, 7], [8, -8, 9, -9, 10]], dtype=torch.int64),
            ["floor_divide"],
            id="floor",
        ),
        pytest.param(
            lambda v: torch.div(v, 3.0, rounding_mode="trunc"),
            torch.tensor([[-7.5, -1.5, 0.0, 1.5, 7.5]]),
            ["trunc_divide"],
            id="float_trunc",
        ),
        pytest.param(
            lambda v: torch.div(v, 3.0),
            torch.tensor([[-7.5, -1.5, 0.0, 1.5, 7.5]]),
            ["div"],
            id="true",
        ),
    ],
)
def test_division_rounding_mode_picks_the_operator(call, value, expected_ops) -> None:
    # ``rounding_mode`` selects a different ATen target, so each spelling is
    # checked against the operator it has to lower to. Truncation is the one
    # that cannot be expressed as a floor, which is why it has its own op.
    compiled, actual = _run(call, value)

    assert [op.op for op in compiled.lowered_graph.ops] == expected_ops
    _assert_exact(actual, call(value))


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(torch.gcd, id="gcd"),
        pytest.param(torch.lcm, id="lcm"),
    ],
)
def test_the_integer_family_matches_pytorch(operation) -> None:
    # Both are defined on the magnitudes, so a negative operand still has to
    # give a positive result, and a zero pair has to give zero rather than a
    # division by zero.
    value = torch.tensor([-18, 7, 0, 2147483647], dtype=torch.int32)
    other = torch.tensor([12, 12, 0, 12], dtype=torch.int32)

    compiled, actual = _run(operation, value, other)

    _assert_exact(actual, operation(value, other))
    _assert_native(compiled)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(torch.bitwise_left_shift, id="left"),
        pytest.param(torch.bitwise_right_shift, id="right"),
    ],
)
def test_the_shifts_match_pytorch(operation) -> None:
    value = torch.tensor([[-8, -1, 0, 1, 5], [2147483647, -2147483648, 3, -3, 7]])
    amounts = torch.tensor([[2, 5, 0, 30, 3], [1, 7, 2, 4, 9]])

    compiled, actual = _run(operation, value, amounts)

    _assert_exact(actual, operation(value, amounts))
    _assert_native(compiled)


@pytest.mark.parametrize(
    "amounts",
    [
        pytest.param(torch.tensor([33, 40, -1, 0]), id="out_of_range"),
        pytest.param(torch.tensor([31, 32, 63, 1]), id="at_the_edge"),
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(torch.bitwise_left_shift, id="left"),
        pytest.param(torch.bitwise_right_shift, id="right"),
    ],
)
def test_a_shift_past_the_width_shifts_every_bit_out(operation, amounts) -> None:
    # ATen defines the out-of-range count rather than leaving it undefined: a
    # left shift yields zero, an arithmetic right shift yields the sign fill,
    # and a negative count behaves like an out-of-range one.
    value = torch.tensor([-8, -1, 5, 1], dtype=torch.int32)

    compiled, actual = _run(operation, value, amounts)

    _assert_exact(actual, operation(value, amounts))
    _assert_native(compiled)


def test_the_family_fuses_like_any_other_pointwise_operator() -> None:
    # The arithmetic around the operator joins the same region, so the whole
    # expression is one kernel rather than three.
    torch.manual_seed(3)
    value = torch.randn(5, 7)

    compiled, actual = _run(lambda v: torch.fmax(v, v * 2.0) + 1.0, value)

    assert [op.op for op in compiled.lowered_graph.ops] == ["fused_elementwise"]
    _assert_exact(actual, torch.fmax(value, value * 2.0) + 1.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nextafter_steps_at_the_operand_width(dtype) -> None:
    # The kernel computes in float, so a float-sized step would round straight
    # back onto the operand when the result is stored as half. ``numpy`` has no
    # bfloat16, so both the input and the expectation travel as float32.
    value = torch.tensor([1.0, -2.5, 0.0, -0.0]).to(dtype)
    target = (value * 1.5).to(dtype)

    compiled = compile_callable(torch.nextafter, (value, target))
    actual = compiled.plan(
        mx.array(value.float().numpy()).astype(_MX_DTYPES[dtype]),
        mx.array(target.float().numpy()).astype(_MX_DTYPES[dtype]),
    )[0]
    mx.eval(actual)
    expected = mx.array(torch.nextafter(value, target).float().numpy()).astype(
        _MX_DTYPES[dtype]
    )

    assert actual.dtype == _MX_DTYPES[dtype]
    assert bool(mx.array_equal(actual, expected))
    _assert_native(compiled)


def test_the_family_also_runs_in_half_precision() -> None:
    # A bf16 kernel rounds differently from the f32 one, so the tolerance is
    # the dtype's rather than the operator's. ``torch.Tensor.numpy`` refuses
    # bf16, so the array is built from the float32 copy and cast back.
    value = torch.randn(4, 7).to(torch.bfloat16) * 0.5
    array = mx.array(value.float().numpy()).astype(mx.bfloat16)

    compiled = compile_callable(lambda v: torch.fmax(v, v * 2.0), (value,))
    actual = compiled.plan(array)[0]
    mx.eval(actual)
    expected = mx.array(torch.fmax(value, value * 2.0).float().numpy()).astype(
        mx.bfloat16
    )

    assert actual.dtype == mx.bfloat16
    assert bool(mx.allclose(actual, expected, atol=2e-2, rtol=2e-2))
    _assert_native(compiled)


def test_an_integer_pair_uses_a_plain_comparison() -> None:
    # No NaN to ignore, so fmax/fmin reduce to max/min and the integer emitter
    # takes over instead of the float one.
    value = torch.tensor([3, -5, 0, 9], dtype=torch.int32)
    other = torch.tensor([4, -6, 0, 8], dtype=torch.int32)

    for operation in (torch.fmax, torch.fmin):
        compiled, actual = _run(operation, value, other)
        _assert_exact(actual, operation(value, other))
        _assert_native(compiled)


def test_the_integer_fallbacks_agree_with_the_kernel() -> None:
    # The same comparison for the integer helpers, which autotune weighs against
    # the Metal kernel: a disagreement would make the two answers a coin flip.
    value = torch.tensor([12, -18, 0, 7], dtype=torch.int32)
    other = torch.tensor([18, 12, 0, 5], dtype=torch.int32)
    call = lambda v, w: (  # noqa: E731
        torch.lcm(v, w)
        + torch.gcd(v, w)
        + torch.bitwise_left_shift(v, 2)
        + torch.bitwise_right_shift(w, 1)
        + torch.fmax(v, w)
        + torch.fmin(v, w)
    )

    lowered = rewrite(capture_fx(call, (value, other)))
    native = AppleMetalBackend().compile(lowered)
    fallback = AppleMetalBackend(
        selection_overrides={
            selection.output_id: "mlx" for selection in native.selections
        }
    ).compile(lowered)

    assert all(selection.is_native for selection in native.selections)
    assert not any(selection.is_native for selection in fallback.selections)

    inputs = (mx.array(value.numpy()), mx.array(other.numpy()))
    reference = fallback(*inputs)[0]
    actual = native(*inputs)[0]
    mx.eval(reference, actual)
    assert bool(mx.array_equal(actual, reference))


def test_the_mlx_fallback_agrees_with_the_kernel() -> None:
    # Autotune compares the two implementations, so they have to be the same
    # function to the last bit; otherwise it would be choosing between
    # different answers.
    torch.manual_seed(17)
    value = torch.randn(3, 4) * 2.0
    call = lambda v: torch.fmax(v, torch.nextafter(v, v * 0.5))  # noqa: E731

    lowered = rewrite(capture_fx(call, (value,)))
    native = AppleMetalBackend().compile(lowered)
    fallback = AppleMetalBackend(
        selection_overrides={
            selection.output_id: "mlx" for selection in native.selections
        }
    ).compile(lowered)

    assert all(selection.is_native for selection in native.selections)
    assert not any(selection.is_native for selection in fallback.selections)

    inputs = (mx.array(value.numpy()),)
    reference = fallback(*inputs)[0]
    actual = native(*inputs)[0]
    mx.eval(reference, actual)
    assert bool(mx.array_equal(actual, reference))
