"""Batch, instance and group normalization, which arrive without a Torch rule.

Torch ships decompositions for the composite operators but not for the two
backend kernels underneath them, so ``native_batch_norm`` and
``native_group_norm`` reach the frontend intact and are rewritten here into the
moments and the affine fix-up. The tests check the two states a normalization
can be in - normalizing with statistics the caller owns, or with moments it has
to compute - and the one it cannot be in here, a training run that writes the
running moments back into state the graph only reads.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith import compile_callable, compile_module
from aotolith.backends import AppleMetalBackend
from aotolith.capture import capture_fx
from aotolith.passes import rewrite


def _compile(module: torch.nn.Module, *values: torch.Tensor):
    compiled = compile_module(module, values)
    actual = compiled(*values)[0]
    mx.eval(actual)
    return compiled, actual


def _close(actual, expected, atol: float = 1e-5) -> bool:
    reference = mx.array(expected.detach().float().numpy())
    return bool(mx.allclose(actual, reference, atol=atol, rtol=1e-5))


def test_batch_norm_normalizes_with_the_running_moments() -> None:
    torch.manual_seed(0)
    module = torch.nn.BatchNorm2d(4).eval()
    value = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        expected = module(value)

    compiled, actual = _compile(module, value)

    assert _close(actual, expected)
    # The whole normalization is a broadcast affine fix-up: nothing reduces,
    # because the moments it uses are bound state.
    ops = {op.op for op in compiled.input_graph.ops}
    assert not ops & {"mean", "var"}
    assert ops <= {"reshape", "add", "sub", "mul", "rsqrt"}


@pytest.mark.parametrize(
    ("module", "shape"),
    [
        pytest.param(torch.nn.BatchNorm1d(4), (3, 4), id="1d"),
        pytest.param(torch.nn.BatchNorm1d(4), (3, 4, 5), id="1d-sequential"),
        pytest.param(torch.nn.BatchNorm2d(4), (2, 4, 3, 3), id="2d"),
        pytest.param(torch.nn.BatchNorm3d(4), (2, 4, 3, 3, 3), id="3d"),
    ],
)
def test_batch_norm_covers_every_rank(
    module: torch.nn.Module, shape: tuple[int, ...]
) -> None:
    torch.manual_seed(1)
    module = module.eval()
    value = torch.randn(*shape)
    with torch.no_grad():
        expected = module(value)

    _, actual = _compile(module, value)

    assert _close(actual, expected)


def test_batch_norm_computes_the_batch_moments_when_it_owns_none() -> None:
    torch.manual_seed(2)
    module = torch.nn.BatchNorm2d(4, track_running_stats=False).eval()
    value = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        expected = module(value)

    compiled, actual = _compile(module, value)

    assert _close(actual, expected)
    ops = {op.op for op in compiled.input_graph.ops}
    assert {"mean", "var"} <= ops


def test_batch_norm_in_training_refuses_to_write_the_running_moments() -> None:
    # A module cannot even reach the tracer in training mode; the functional
    # spelling can, and it is the one that has to say why it cannot be lowered.
    module = torch.nn.BatchNorm2d(4).train()
    value = torch.randn(2, 4, 3, 3)

    with pytest.raises(ValueError, match="requires module.eval"):
        compile_module(module, (value,))

    weight, bias = torch.randn(4), torch.randn(4)
    running_mean, running_var = torch.randn(4), torch.rand(4) + 1.0

    def call(input, weight, bias, mean, var):
        return torch.nn.functional.batch_norm(
            input, mean, var, weight, bias, training=True, eps=1e-5
        )

    with pytest.raises(NotImplementedError, match="updates the running moments"):
        compile_callable(call, (value, weight, bias, running_mean, running_var))


def test_instance_norm_gets_its_moments_from_each_sample() -> None:
    # ``InstanceNorm2d`` reshapes the batch into channels and normalizes each
    # sample on its own, which is the batch normalization with training moments
    # and no state to keep.
    torch.manual_seed(3)
    module = torch.nn.InstanceNorm2d(4, affine=True).eval()
    value = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        expected = module(value)

    compiled, actual = _compile(module, value)

    assert _close(actual, expected)
    assert {"mean", "var"} <= {op.op for op in compiled.input_graph.ops}


def test_instance_norm_with_tracked_statistics_uses_them_in_eval() -> None:
    torch.manual_seed(4)
    module = torch.nn.InstanceNorm2d(4, affine=True, track_running_stats=True).eval()
    value = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        expected = module(value)

    compiled, actual = _compile(module, value)

    assert _close(actual, expected)
    assert not {op.op for op in compiled.input_graph.ops} & {"mean", "var"}


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_group_norm_matches_pytorch(groups: int) -> None:
    torch.manual_seed(5)
    module = torch.nn.GroupNorm(groups, 8, affine=True).eval()
    value = torch.randn(3, 8, 2, 2)
    with torch.no_grad():
        expected = module(value)

    compiled, actual = _compile(module, value)

    assert _close(actual, expected)
    ops = {op.op for op in compiled.input_graph.ops}
    # One row reduction per group, expressed in operators the backend has.
    assert "group_norm" not in ops
    assert "mean" in ops


def test_group_norm_writes_the_moments_it_reduces() -> None:
    torch.manual_seed(6)
    value = torch.randn(2, 8, 3, 3)
    weight = torch.randn(8)
    bias = torch.randn(8)

    def call(input, weight, bias):
        return torch.ops.aten.native_group_norm.default(
            input, weight, bias, 2, 8, 9, 4, 1e-5
        )

    expected = call(value, weight, bias)
    compiled = compile_callable(call, (value, weight, bias))
    arrays = [mx.array(tensor.numpy()) for tensor in (value, weight, bias)]
    actual = compiled.plan(*arrays)
    mx.eval(*actual)

    assert len(actual) == 3
    for produced, reference in zip(actual, expected):
        assert produced.shape == tuple(reference.shape)
        assert _close(produced, reference)


def test_group_norm_rejects_groups_that_do_not_divide_the_channels() -> None:
    # ATen rejects the contraction itself, so the frontend never sees the call;
    # the guard below is what the rule would say if it did.
    value = torch.randn(2, 6, 3, 3)

    with pytest.raises(RuntimeError, match="divisible by num_groups"):
        capture_fx(lambda input: torch.nn.functional.group_norm(input, 4), (value,))


def test_the_inference_spelling_without_training_uses_the_running_moments() -> None:
    # ``_native_batch_norm_legit_no_training`` is the entry point a backend that
    # has already ruled out training dispatches to; it takes the moments
    # positionally and cannot update them.
    torch.manual_seed(7)
    value = torch.randn(2, 4, 3, 3)
    weight, bias = torch.randn(4), torch.randn(4)
    running_mean, running_var = torch.randn(4), torch.rand(4) + 1.0

    def call(input, weight, bias, mean, var):
        return torch.ops.aten._native_batch_norm_legit_no_training.default(
            input, weight, bias, mean, var, 0.1, 1e-5
        )[0]

    expected = call(value, weight, bias, running_mean.clone(), running_var.clone())
    actual = compile_callable(
        call, (value, weight, bias, running_mean, running_var)
    ).plan(
        *(
            mx.array(tensor.numpy())
            for tensor in (value, weight, bias, running_mean, running_var)
        )
    )[0]
    mx.eval(actual)

    assert _close(actual, expected)


def test_half_precision_batch_norm_accumulates_in_float32() -> None:
    torch.manual_seed(8)
    module = torch.nn.BatchNorm2d(4, track_running_stats=False).eval().half()
    value = torch.randn(2, 4, 3, 3).half()
    with torch.no_grad():
        expected = module(value)

    _, actual = _compile(module, value)

    assert actual.dtype == mx.float16
    # A half-precision reduction of the same data would be visibly worse than
    # the float32 one ATen accumulates in.
    assert _close(actual, expected, atol=2e-3)


def test_the_rewritten_normalization_still_plans_on_the_backend() -> None:
    torch.manual_seed(9)
    value = torch.randn(2, 4, 3, 3)
    weight, bias = torch.randn(4), torch.randn(4)

    def call(input, weight, bias):
        return torch.nn.functional.group_norm(input, 2, weight, bias)

    lowered = rewrite(capture_fx(call, (value, weight, bias)))
    plan = AppleMetalBackend().compile(lowered)
    actual = plan(*(mx.array(tensor.numpy()) for tensor in (value, weight, bias)))[0]
    mx.eval(actual)

    expected = call(value, weight, bias)
    assert _close(actual, expected)
