"""The ATen decomposition table that legalization runs on the way in."""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith import compile_callable
from aotolith.decompose import aten_decompositions, is_kept_native, native_targets
from aotolith.frontend import NATIVE_TARGETS, capture_fx


def test_native_targets_is_the_frontends_own_list() -> None:
    assert native_targets() == NATIVE_TARGETS


def test_attention_and_layer_norm_are_never_decomposed() -> None:
    # These own a Metal kernel; a decomposition would replace it with a longer
    # graph and lose the fused implementation.
    for target in (
        "aten.scaled_dot_product_attention.default",
        "aten.native_layer_norm.default",
        "aten._softmax.default",
        "aten.gelu.default",
    ):
        assert is_kept_native(target)


def test_the_table_does_not_rewrite_an_operator_the_frontend_lowers() -> None:
    table = aten_decompositions()
    overlap = {str(target) for target in table} & NATIVE_TARGETS
    assert overlap == set()


def test_attention_still_reaches_the_backend_as_one_operation() -> None:
    query = torch.randn(1, 2, 4, 8)
    graph = capture_fx(
        lambda value: torch.nn.functional.scaled_dot_product_attention(
            value, value, value
        ),
        (query,),
    )
    assert "attention" in {op.op for op in graph.ops}


def test_the_table_covers_the_long_tail() -> None:
    table = {str(target) for target in aten_decompositions()}
    # A sample of operators no backend would implement by hand.
    for target in (
        "aten.huber_loss.default",
        "aten.glu.default",
        "aten.celu.default",
        "aten.mse_loss.default",
        "aten.smooth_l1_loss.default",
        "aten.binary_cross_entropy_with_logits.default",
    ):
        assert target in table


def test_the_table_is_built_once() -> None:
    assert aten_decompositions() is aten_decompositions()


def test_a_composite_the_frontend_never_named_compiles() -> None:
    # `celu` is in neither the frontend's operator table nor its special cases:
    # it arrives only through a decomposition, and the operators that
    # decomposition emits are all supported.
    x = torch.rand(3, 4)
    graph = capture_fx(lambda value: torch.nn.functional.celu(value), (x,))
    assert "celu" not in {op.op for op in graph.ops}


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda value: torch.special.entr(value), id="entr"),
        pytest.param(lambda value: torch.nn.functional.celu(value), id="celu"),
        pytest.param(
            lambda value: torch.nn.functional.huber_loss(
                value, torch.zeros_like(value)
            ),
            id="huber",
        ),
        pytest.param(
            lambda value: torch.nn.functional.smooth_l1_loss(
                value, torch.zeros_like(value)
            ),
            id="smooth_l1",
        ),
        pytest.param(
            lambda value: torch.nn.functional.mse_loss(value, torch.zeros_like(value)),
            id="mse",
        ),
        pytest.param(lambda value: torch.nn.functional.glu(value, dim=-1), id="glu"),
    ],
)
def test_decomposed_composites_run_on_metal(call) -> None:
    torch.manual_seed(11)
    x = torch.rand(4, 6) - 0.25
    actual = compile_callable(call, (x,)).plan(mx.array(x.numpy()))[0]
    expected = mx.array(call(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert bool(mx.allclose(actual, expected, atol=1e-6, rtol=1e-6))
