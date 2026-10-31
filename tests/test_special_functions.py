"""Special functions that have no MSL equivalent and need a helper."""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith import compile_callable


def _compare(call, value: torch.Tensor, atol: float, rtol: float) -> None:
    actual = compile_callable(call, (value,)).plan(mx.array(value.numpy()))[0]
    expected = mx.array(call(value).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert actual.dtype == expected.dtype
    left = actual.astype(mx.float32) if actual.dtype != mx.bool_ else actual
    right = expected.astype(mx.float32) if expected.dtype != mx.bool_ else expected
    assert bool(mx.allclose(left, right, atol=atol, rtol=rtol))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda v: torch.lgamma(v), id="lgamma"),
        pytest.param(lambda v: torch.lgamma(v.abs() + 1.0), id="lgamma_shifted"),
        pytest.param(lambda v: torch.special.i0(v), id="i0"),
        pytest.param(lambda v: torch.erfinv(v.clamp(-0.9, 0.9)), id="erfinv"),
        pytest.param(lambda v: torch.digamma(v), id="digamma"),
    ],
)
def test_special_functions_match_pytorch(call) -> None:
    torch.manual_seed(7)
    # Span both sides of zero so lgamma's reflection formula and i0's two
    # polynomial branches are both exercised.
    value = torch.rand(6, 8) * 6.0 - 2.5
    _compare(call, value, atol=2e-5, rtol=2e-5)


def test_lgamma_reports_the_poles_as_infinite() -> None:
    # ATen returns +inf at zero and at every negative integer, and for both
    # infinities; only a NaN argument stays NaN. lgamma(3) is log(2), which is
    # the non-pole in this row.
    value = torch.tensor([0.0, -1.0, -2.0, 3.0, float("inf"), float("-inf")])
    actual = compile_callable(torch.lgamma, (value,)).plan(mx.array(value.numpy()))[0]
    mx.eval(actual)
    assert actual.tolist()[:3] == [float("inf")] * 3
    assert actual[3].item() == pytest.approx(0.6931472, abs=1e-6)
    assert actual.tolist()[4:] == [float("inf")] * 2


def test_digamma_reports_the_poles_as_nan_and_the_zeros_as_infinite() -> None:
    # ATen returns NaN at every negative integer (the reflection terms
    # disagree there) and an infinity at zero whose sign is the opposite of the
    # zero's. Half-integers are the finite neighbours either side of a pole.
    value = torch.tensor([0.0, -0.0, -1.0, -2.0, -0.5, 0.5, 10.0])
    actual = compile_callable(torch.digamma, (value,)).plan(mx.array(value.numpy()))[0]
    expected = torch.digamma(value)
    mx.eval(actual)

    assert actual[0].item() == float("-inf")
    assert actual[1].item() == float("inf")
    assert bool(mx.isnan(actual[2])) and bool(mx.isnan(actual[3]))
    assert bool(
        mx.allclose(actual[4:], mx.array(expected[4:].numpy()), atol=2e-6, rtol=2e-6)
    )


def test_digamma_stays_accurate_next_to_a_pole() -> None:
    # The reflection term is pi * cot(pi * x), and evaluating it at a
    # fractional part that is close to +-1 loses the distance to the pole. The
    # helper reduces the angle to [-0.5, 0.5] first, so these values - where
    # digamma is in the thousands - still agree to a few ulps.
    value = torch.tensor([-0.9999294, -1.9998026, -3.9998398, -0.0001, 0.0001])
    actual = compile_callable(torch.digamma, (value,)).plan(mx.array(value.numpy()))[0]
    expected = torch.digamma(value)
    mx.eval(actual)

    assert bool(mx.allclose(actual, mx.array(expected.numpy()), atol=1e-3, rtol=2e-6))


def test_the_fallback_implementation_matches_pytorch_too() -> None:
    # Autotune compares the kernel against the MLX path, so both have to agree
    # with ATen; this pins the MLX half of lgamma and digamma, whose
    # approximations are duplicated by hand in ``runtime/mlx_ops``.
    from aotolith.backends import AppleMetalBackend
    from aotolith.capture import capture_fx
    from aotolith.passes import rewrite

    torch.manual_seed(13)
    value = torch.rand(5, 6) * 6.0 - 2.5
    for call in (torch.lgamma, torch.digamma):
        lowered = rewrite(capture_fx(call, (value,)))
        native = AppleMetalBackend().compile(lowered)
        fallback = AppleMetalBackend(
            selection_overrides={
                selection.output_id: "mlx" for selection in native.selections
            }
        ).compile(lowered)

        assert all(selection.is_native for selection in native.selections)
        assert not any(selection.is_native for selection in fallback.selections)

        actual = fallback(mx.array(value.numpy()))[0]
        expected = mx.array(call(value).numpy())
        mx.eval(actual, expected)
        assert bool(mx.allclose(actual, expected, atol=2e-5, rtol=2e-5))


def test_signbit_is_a_bit_test() -> None:
    # -0.0 is the case a comparison would get wrong.
    value = torch.tensor([[0.0, -0.0, -1.0, 2.0]])
    _compare(torch.signbit, value, atol=0.0, rtol=0.0)


def test_signbit_accepts_integers() -> None:
    value = torch.tensor([[-3, 0, 7]], dtype=torch.int32)
    _compare(torch.signbit, value, atol=0.0, rtol=0.0)


def test_a_decomposition_that_needs_lgamma_compiles() -> None:
    # ``torch.special.multigammaln`` is expressed in terms of lgamma, so it only
    # became compilable when the leaf operator did. The input is the symmetric
    # positive-definite matrix the function expects: (d + 1) / 2 diagonal.
    diagonal = torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0]])
    _compare(
        lambda v: torch.special.multigammaln(v, 5),
        diagonal,
        atol=2e-5,
        rtol=2e-5,
    )
