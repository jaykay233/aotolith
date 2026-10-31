"""The trigonometric and hyperbolic family, end to end.

Torch treats these nine as core ATen primitives, so no decomposition rewrites
them and the frontend has to lower each one itself. Metal's standard library has
all nine, so they become ordinary pointwise expressions - which also means they
fuse into a neighbouring region like any other elementwise operator.

Each case asserts both halves: the numbers match PyTorch, and the plan actually
emitted a Metal kernel. Without the second check a silent fall back to MLX would
keep the test green while the codegen it is meant to cover went unexercised.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith import compile_callable
from aotolith.backends import AppleMetalBackend
from aotolith.capture import capture_fx
from aotolith.passes import rewrite

#: Every operator, with a domain that keeps it inside its own branch: ``asin``
#: and ``acos`` are only real on [-1, 1] and ``atanh`` only on (-1, 1), so
#: feeding them the same wide random tensor would test NaN propagation instead.
CASES = [
    pytest.param(lambda v: torch.tan(v), id="tan"),
    pytest.param(lambda v: torch.asin(torch.tanh(v) * 0.9), id="asin"),
    pytest.param(lambda v: torch.acos(torch.tanh(v) * 0.9), id="acos"),
    pytest.param(lambda v: torch.atan(v), id="atan"),
    pytest.param(lambda v: torch.sinh(v), id="sinh"),
    pytest.param(lambda v: torch.cosh(v), id="cosh"),
    pytest.param(lambda v: torch.asinh(v), id="asinh"),
    pytest.param(lambda v: torch.acosh(v.abs() + 1.0), id="acosh"),
    pytest.param(lambda v: torch.atanh(torch.tanh(v) * 0.9), id="atanh"),
]

#: The subset defined on every real number, which is what an integer operand can
#: be fed without the case turning into a NaN test.
DOMAIN_FREE = [
    pytest.param(lambda v: torch.tan(v), id="tan"),
    pytest.param(lambda v: torch.atan(v), id="atan"),
    pytest.param(lambda v: torch.sinh(v), id="sinh"),
    pytest.param(lambda v: torch.cosh(v), id="cosh"),
    pytest.param(lambda v: torch.asinh(v), id="asinh"),
]


def _run(call, value: torch.Tensor):
    compiled = compile_callable(call, (value,))
    actual = compiled.plan(mx.array(value.numpy()))[0]
    mx.eval(actual)
    return compiled, actual


@pytest.mark.parametrize("call", CASES)
def test_the_family_matches_pytorch(call) -> None:
    torch.manual_seed(11)
    value = torch.randn(6, 9) * 2.0

    compiled, actual = _run(call, value)
    expected = mx.array(call(value).numpy())

    assert actual.dtype == expected.dtype
    assert bool(mx.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    assert any(selection.is_native for selection in compiled.plan.selections), (
        "the operator should have compiled to a Metal kernel"
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda v: torch.tan(v + 1.0), id="tan"),
        pytest.param(lambda v: torch.atan(v * 2.0), id="atan"),
    ],
)
def test_the_family_fuses_like_any_other_pointwise_operator(call) -> None:
    # The arithmetic around the operator joins the same region, so the whole
    # expression is one kernel rather than three.
    torch.manual_seed(3)
    value = torch.randn(5, 7)

    compiled, actual = _run(call, value)

    assert [op.op for op in compiled.lowered_graph.ops] == ["fused_elementwise"]
    assert bool(
        mx.allclose(actual, mx.array(call(value).numpy()), atol=2e-6, rtol=2e-6)
    )


@pytest.mark.parametrize("call", DOMAIN_FREE)
def test_an_integer_operand_is_promoted_to_float(call) -> None:
    # ATen's rule for these operators is INT_TO_FLOAT: the result is float32
    # even though the operand is integral, and the kernel only has the float
    # overload, so the frontend has to insert the cast.
    value = torch.tensor([[-2, -1, 0, 1, 2], [3, 4, 5, 6, 7]], dtype=torch.int32)

    compiled, actual = _run(call, value)
    expected = call(value)

    assert expected.dtype == torch.float32
    assert actual.dtype == mx.float32
    assert bool(mx.allclose(actual, mx.array(expected.numpy()), atol=2e-6, rtol=2e-6))
    assert any(selection.op == "cast" for selection in compiled.plan.selections)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda v: v * 0.5, id="mul"),
        pytest.param(lambda v: v + 0.5, id="add"),
        pytest.param(lambda v: v * 2, id="mul_integer_scalar"),
    ],
)
def test_an_integer_tensor_applies_a_float_scalar_in_float(call) -> None:
    # The same promotion rule seen from the other side: a Python float is a
    # higher promotion category than an integer tensor, so the multiply happens
    # in float32. An integer scalar does not promote, so that case stays int.
    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    _, actual = _run(call, value)
    expected = call(value)

    assert actual.dtype == mx.array(expected.numpy()).dtype
    assert bool(mx.allclose(actual, mx.array(expected.numpy()), atol=0.0, rtol=0.0))


def test_the_family_also_runs_in_half_precision() -> None:
    # A bf16 kernel rounds differently from the f32 one, so the tolerance is
    # the dtype's rather than the operator's. ``torch.Tensor.numpy`` refuses
    # bf16, so the array is built from the float32 copy and cast back.
    value = torch.randn(4, 7).to(torch.bfloat16) * 0.5
    array = mx.array(value.float().numpy()).astype(mx.bfloat16)

    compiled = compile_callable(torch.atan, (value,))
    actual = compiled.plan(array)[0]
    mx.eval(actual)
    expected = mx.array(torch.atan(value).float().numpy()).astype(mx.bfloat16)

    assert actual.dtype == mx.bfloat16
    assert bool(mx.allclose(actual, expected, atol=2e-2, rtol=2e-2))
    assert any(selection.is_native for selection in compiled.plan.selections)


def test_the_mlx_fallback_agrees_with_the_kernel() -> None:
    # Autotune compares the two implementations, so they have to be the same
    # function to within rounding; otherwise it would be choosing between
    # different answers.
    torch.manual_seed(17)
    value = torch.randn(3, 4) * 2.0
    call = lambda v: torch.sinh(v) + torch.tanh(v)  # noqa: E731

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
    assert bool(mx.allclose(actual, reference, atol=2e-6, rtol=2e-6))


@pytest.mark.parametrize(
    "shape, weight_shape",
    [
        pytest.param((2, 5), (5,), id="rank2_channel"),
        pytest.param((2, 3, 4), (3,), id="rank3_channel"),
        pytest.param((2, 3, 4, 4), (3,), id="rank4_channel"),
        pytest.param((6,), (1,), id="vector"),
    ],
)
def test_prelu_matches_pytorch(shape, weight_shape) -> None:
    torch.manual_seed(3)
    value = torch.randn(*shape)
    weight = torch.randn(*weight_shape)

    def call(v, w):
        return torch.nn.functional.prelu(v, w)

    compiled = compile_callable(call, (value, weight))
    actual = compiled.plan(mx.array(value.numpy()), mx.array(weight.numpy()))[0]
    mx.eval(actual)

    expected = call(value, weight)
    assert bool(mx.allclose(actual, mx.array(expected.numpy()), atol=1e-6, rtol=1e-6))
    # The three operators of the slope are folded into the surrounding region
    # instead of running one pass each.
    assert [op.op for op in compiled.lowered_graph.ops][-1] == "fused_elementwise"
    assert compiled.plan.selections[-1].is_native


def test_prelu_zero_stays_zero_for_both_branches() -> None:
    # ``where(x > 0, x, x * w)`` keeps a zero at the boundary from turning into
    # a signed zero of the other branch.
    value = torch.tensor([-1.0, 0.0, 1.0])
    weight = torch.tensor([0.25])
    compiled = compile_callable(
        lambda v, w: torch.nn.functional.prelu(v, w), (value, weight)
    )
    actual = compiled.plan(mx.array(value.numpy()), mx.array(weight.numpy()))[0]
    mx.eval(actual)

    expected = torch.nn.functional.prelu(value, weight)
    assert bool(mx.array_equal(actual, mx.array(expected.numpy())))


def test_prelu_runs_in_half_precision() -> None:
    torch.manual_seed(5)
    value = torch.randn(2, 4).to(torch.bfloat16)
    weight = torch.randn(4).to(torch.bfloat16)
    compiled = compile_callable(
        lambda v, w: torch.nn.functional.prelu(v, w), (value, weight)
    )
    actual = compiled.plan(
        mx.array(value.float().numpy()).astype(mx.bfloat16),
        mx.array(weight.float().numpy()).astype(mx.bfloat16),
    )[0]
    mx.eval(actual)

    expected = torch.nn.functional.prelu(value, weight)
    assert actual.dtype == mx.bfloat16
    assert bool(
        mx.allclose(
            actual.astype(mx.float32),
            mx.array(expected.float().numpy()),
            atol=2e-2,
            rtol=2e-2,
        )
    )
