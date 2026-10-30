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
