from __future__ import annotations

import functools
import math

import mlx.core as mx
import pytest
import torch

from aotolith import (
    compile_callable,
    compile_graph,
    eliminate_common_subexpressions,
    fuse_pointwise,
    load_compiled_module,
    save_compiled,
)
import aotolith.codegen.msl as metal_msl
import aotolith.runtime.mlx_ops.pointwise as metal_pointwise
import aotolith.target.apple as metal_target
from aotolith.ir import DType, Graph, Operation, TensorType, Value


_MLX_DTYPES = {
    torch.float16: mx.float16,
    torch.bfloat16: mx.bfloat16,
    torch.float32: mx.float32,
    torch.bool: mx.bool_,
    torch.int32: mx.int32,
    torch.int64: mx.int64,
}


@functools.lru_cache(maxsize=None)
def _device_builds_tensor_core_kernels() -> bool:
    """Ask whether compiling here actually yields a tensor-core plan.

    This is the backend's own verdict, not a second opinion: `compile` tries to
    load one emitted tensor-core kernel and re-lowers without them if the
    device refuses, so the kernel name already reflects what runs here. A
    paravirtualised GPU (the hosted CI runners) reports `__HAVE_TENSOR__` and
    rejects the metallib, and its plans come back as simdgroup kernels.
    """

    torch.manual_seed(1)
    inputs = (
        torch.randn(64, 64, dtype=torch.float16),
        torch.randn(64, 64, dtype=torch.float16),
    )
    compiled = compile_callable(
        lambda x, weight: x @ weight, inputs, codegen_matmul=True
    )
    return "tensorcore" in compiled.plan.kernels[0].name


def _require_tensor_cores() -> None:
    """Skip tests that assert tensor-core source where the device has none."""

    if not _device_builds_tensor_core_kernels():
        pytest.skip("this GPU cannot load MetalPerformancePrimitives kernels")


def _torch_to_mlx(value: torch.Tensor) -> mx.array:
    source = value.float().numpy() if value.is_floating_point() else value.numpy()
    return mx.array(source).astype(_MLX_DTYPES[value.dtype])


def _assert_values_match(
    actual: mx.array,
    expected: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Compare elementwise, keeping NaN and infinite expectations exact."""

    mx.eval(actual)
    got = actual.reshape(-1).tolist()
    want = expected.reshape(-1).tolist()
    for got_value, want_value in zip(got, want, strict=True):
        if math.isnan(want_value):
            assert math.isnan(got_value)
        elif math.isinf(want_value):
            assert got_value == want_value
        else:
            assert got_value == pytest.approx(want_value, abs=atol, rel=rtol)


def _manual_gate_graph(dtype: DType = DType.F32) -> Graph:
    tensor_type = TensorType((3,), dtype)
    values = {
        "x": Value("x", tensor_type),
        "gate": Value("gate", tensor_type),
        "silu": Value("silu", tensor_type),
        "out": Value("out", tensor_type),
    }
    ops = (
        Operation("silu", ("gate",), ("silu",)),
        Operation("mul", ("x", "silu"), ("out",)),
    )
    return Graph(("x", "gate"), values, ops, ("out",))


def test_pointwise_fusion_emits_one_metal_region() -> None:
    graph = fuse_pointwise(_manual_gate_graph())
    assert [op.op for op in graph.ops] == ["fused_elementwise"]

    result = compile_graph(graph)
    assert len(result.plan.kernels) == 1
    assert "metal::exp" in result.plan.kernels[0].source


def test_repeated_compilation_reuses_identical_metal_kernel() -> None:
    # The first compilation probes for tensor cores, and that probe is itself a
    # cached metal kernel, so memoize it before counting misses.
    metal_target._compiler_reports_tensor_ops()
    metal_msl._metal_kernel.cache_clear()
    x = torch.randn(2, 8)
    first = compile_callable(lambda value: torch.relu(value), (x,))
    second = compile_callable(lambda value: torch.relu(value), (x,))

    cache_info = metal_msl._metal_kernel.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1

    input_array = mx.array(x.numpy())
    expected = mx.maximum(input_array, 0.0)
    for result in (first, second):
        actual = result.plan(input_array)[0]
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected)


def test_scalar_tensor_input_is_loaded_without_indexing() -> None:
    value = torch.randn(2, 3)
    scale = torch.tensor(2.5)
    compiled = compile_callable(lambda left, right: left * right + 1.0, (value, scale))

    actual = compiled.plan(mx.array(value.numpy()), mx.array(scale.numpy()))[0]
    expected = mx.array((value * scale + 1.0).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected)


def test_single_token_mlp_pairs_gate_and_up_into_one_projection() -> None:
    def mlp(x, gate_w, up_w):
        gate = torch.nn.functional.linear(x, gate_w)
        up = torch.nn.functional.linear(x, up_w)
        return torch.nn.functional.silu(gate) * up

    torch.manual_seed(11)
    torch_inputs = (
        torch.randn(1, 16),
        torch.randn(32, 16),
        torch.randn(32, 16),
    )
    result = compile_callable(mlp, torch_inputs, codegen_matmul=True)
    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]
    assert result.lowered_graph.ops[0].attrs.get("epilogue") == "silu_mul_paired"

    inputs = tuple(mx.array(value.numpy()) for value in torch_inputs)
    actual = result.plan(*inputs)[0]
    gate = mx.matmul(inputs[0], inputs[1].T)
    up = mx.matmul(inputs[0], inputs[2].T)
    expected = gate / (1.0 + mx.exp(-gate)) * up
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_multi_row_mlp_keeps_gate_and_up_separate() -> None:
    def mlp(x, gate_w, up_w):
        gate = torch.nn.functional.linear(x, gate_w)
        up = torch.nn.functional.linear(x, up_w)
        return torch.nn.functional.silu(gate) * up

    torch.manual_seed(12)
    torch_inputs = (
        torch.randn(3, 16),
        torch.randn(32, 16),
        torch.randn(32, 16),
    )
    result = compile_callable(mlp, torch_inputs)
    # A taller operand goes to the tiled multiply, which has no paired form, so
    # the gate keeps its own projection and the multiply stays in the epilogue.
    assert [op.op for op in result.lowered_graph.ops] == ["matmul", "matmul"]
    epilogues = [op.attrs.get("epilogue") for op in result.lowered_graph.ops]
    assert epilogues == [None, "silu_mul"]

    inputs = tuple(mx.array(value.numpy()) for value in torch_inputs)
    actual = result.plan(*inputs)[0]
    gate = mx.matmul(inputs[0], inputs[1].T)
    up = mx.matmul(inputs[0], inputs[2].T)
    expected = gate / (1.0 + mx.exp(-gate)) * up
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mlp_end_to_end_on_metal() -> None:
    def mlp(x, gate_w, up_w, down_w):
        gate = x @ gate_w
        up = x @ up_w
        return (torch.nn.functional.silu(gate) * up) @ down_w

    torch.manual_seed(7)
    torch_inputs = (
        torch.randn(2, 16),
        torch.randn(16, 32),
        torch.randn(16, 32),
        torch.randn(32, 16),
    )
    result = compile_callable(mlp, torch_inputs)
    assert [op.op for op in result.lowered_graph.ops] == [
        "matmul",
        "matmul",
        "matmul",
    ]
    # silu(gate) * up is folded into the up-projection's epilogue.
    assert result.lowered_graph.ops[1].attrs.get("epilogue") == "silu_mul"

    inputs = tuple(mx.array(value.numpy()) for value in torch_inputs)
    actual = result.plan(*inputs)[0]
    gate = mx.matmul(inputs[0], inputs[1])
    up = mx.matmul(inputs[0], inputs[2])
    expected = mx.matmul(gate / (1.0 + mx.exp(-gate)) * up, inputs[3])
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_pointwise_region_scheduled_after_interleaved_producer() -> None:
    def fn(x, gate_weight, up_weight):
        gate = torch.nn.functional.silu(x @ gate_weight)
        up = x @ up_weight
        return gate * up

    torch.manual_seed(41)
    inputs = (torch.randn(3, 8), torch.randn(8, 16), torch.randn(8, 16))
    result = compile_callable(fn, inputs)
    lowered_ops = [op.op for op in result.lowered_graph.ops]
    assert lowered_ops == ["matmul", "matmul"]
    # The activation and the multiply land in the second projection's epilogue
    # instead of a separate pointwise kernel.
    assert "fused_elementwise" not in lowered_ops
    assert result.lowered_graph.ops[-1].attrs.get("epilogue") == "silu_mul"

    mlx_inputs = tuple(mx.array(value.numpy()) for value in inputs)
    actual = result.plan(*mlx_inputs)[0]
    gate = mlx_inputs[0] @ mlx_inputs[1]
    gate = gate / (1.0 + mx.exp(-gate))
    expected = gate * (mlx_inputs[0] @ mlx_inputs[2])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_pointwise_fusion_preserves_observed_intermediate() -> None:
    def fn(x):
        activated = torch.relu(x)
        return activated, torch.sigmoid(activated)

    torch.manual_seed(42)
    x = torch.randn(2, 8)
    result = compile_callable(fn, (x,))

    assert [op.op for op in result.lowered_graph.ops] == ["relu", "sigmoid"]
    actual = result.plan(mx.array(x.numpy()))
    expected = tuple(mx.array(value.numpy()) for value in fn(x))
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert mx.allclose(actual_value, expected_value)


def test_pointwise_fusion_reuses_shared_expression() -> None:
    def fn(x):
        activated = torch.sigmoid(x)
        return (activated + activated) * activated

    torch.manual_seed(43)
    x = torch.randn(2, 8)
    result = compile_callable(fn, (x,))

    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    source = result.plan.kernels[0].source
    assert source.count("metal::exp") == 1

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected)


@pytest.mark.parametrize("op_name", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_comparison_ops_produce_metal_boolean_outputs(op_name) -> None:
    compare = getattr(torch, op_name)

    def fn(left, right):
        return compare(left, right)

    torch.manual_seed(11)
    inputs = (torch.randn(13), torch.randn(13))
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    assert result.plan.kernels[0].tensor_type.dtype == DType.BOOL

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.bool_
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_boolean_predicates_and_where_fuse_into_one_kernel(dtype) -> None:
    def fn(left, right, enabled):
        condition = (left > right) & (left < 0.5)
        condition = condition | (right < -0.5)
        condition = condition ^ (left == 0.5)
        condition = ~condition
        return torch.where(condition & enabled, left + right, left - right)

    torch.manual_seed(13)
    inputs = (
        torch.randn(4, 7).to(dtype),
        torch.randn(4, 7).to(dtype),
        torch.rand(4, 7) > 0.5,
    )
    result = compile_callable(fn, inputs)
    input_ops = [op.op for op in result.input_graph.ops]
    expected_ops = {
        "gt",
        "lt",
        "eq",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
        "where",
    }
    assert expected_ops <= set(input_ops)
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert len(result.plan.kernels) == 1

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_where_fuses_scalar_tensor_constant() -> None:
    def fn(value):
        return torch.where(value > 0, value, 0.0)

    value = torch.randn(19)
    result = compile_callable(fn, (value,))
    assert "scalar_tensor" in [op.op for op in result.input_graph.ops]
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "scalar"),
    [
        (torch.bool, True),
        (torch.bool, False),
        (torch.float32, True),
        (torch.float32, False),
    ],
)
def test_comparison_with_boolean_scalar(dtype, scalar) -> None:
    def fn(value):
        return value == scalar

    torch.manual_seed(17)
    if dtype == torch.bool:
        value = torch.rand(23) > 0.5
    else:
        value = torch.randn(23)
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["eq"]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.bool_
    assert mx.array_equal(actual, expected)


def test_where_fuses_two_boolean_scalar_constants() -> None:
    def fn(condition):
        return torch.where(condition, True, False)

    condition = torch.rand(17) > 0.5
    result = compile_callable(fn, (condition,))
    input_ops = [op.op for op in result.input_graph.ops]
    assert input_ops.count("scalar_tensor") == 2
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(_torch_to_mlx(condition))[0]
    expected = mx.array(fn(condition).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.bool_
    assert mx.array_equal(actual, expected)


def test_bitwise_boolean_scalars_fuse_with_numeric_selection() -> None:
    def fn(value):
        condition = (value > 0) & True
        condition = condition | False
        condition = condition ^ True
        return torch.where(condition, value, -1.0)

    torch.manual_seed(19)
    value = torch.randn(29)
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.input_graph.ops if op.op.startswith("bitwise_")] == [
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    ]
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert len(result.plan.kernels) == 1

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_isnan_and_isinf_fuse_into_boolean_output() -> None:
    def fn(value):
        return torch.isnan(value) | torch.isinf(value)

    value = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, 2.0])
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert result.plan.kernels[0].tensor_type.dtype == DType.BOOL

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_masked_fill_lowers_broadcast_boolean_mask(dtype) -> None:
    def fn(value, mask):
        return value.masked_fill(mask, float("-inf"))

    torch.manual_seed(23)
    value = torch.randn(2, 2, 3, 4).to(dtype)
    mask = torch.rand(1, 1, 3, 4) > 0.5
    result = compile_callable(fn, (value, mask))
    assert [op.op for op in result.lowered_graph.ops] == ["masked_fill"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["masked_fill"]

    actual = result.plan(_torch_to_mlx(value), _torch_to_mlx(mask))[0]
    expected = _torch_to_mlx(fn(value, mask))
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_masked_fill_preserves_boolean_tensor_type() -> None:
    def fn(value, mask):
        return value.masked_fill(mask, True)

    value = torch.zeros(19, dtype=torch.bool)
    mask = torch.rand(19) > 0.5
    result = compile_callable(fn, (value, mask))
    actual = result.plan(_torch_to_mlx(value), _torch_to_mlx(mask))[0]
    expected = mx.array(fn(value, mask).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.bool_
    assert mx.array_equal(actual, expected)


def test_masked_fill_fuses_with_inline_predicate() -> None:
    def fn(value):
        return value.masked_fill((value < 0) | torch.isnan(value), 0.0)

    torch.manual_seed(27)
    value = torch.randn(31)
    value[3] = float("nan")
    value[7] = -float("inf")
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_common_subexpression_elimination_reuses_matmul() -> None:
    matrix = TensorType((2, 3), DType.F32)
    weight = TensorType((3, 4), DType.F32)
    output = TensorType((2, 4), DType.F32)
    values = {
        "x": Value("x", matrix),
        "w": Value("w", weight),
        "left": Value("left", output),
        "right": Value("right", output),
        "out": Value("out", output),
    }
    graph = Graph(
        ("x", "w"),
        values,
        (
            Operation("matmul", ("x", "w"), ("left",)),
            Operation("matmul", ("x", "w"), ("right",)),
            Operation("add", ("left", "right"), ("out",)),
        ),
        ("out",),
    )

    optimized = eliminate_common_subexpressions(graph)
    assert [op.op for op in optimized.ops] == ["matmul", "add"]
    assert optimized.ops[1].inputs == (optimized.ops[0].outputs[0],) * 2

    result = compile_graph(optimized)
    x = torch.randn(2, 3)
    w = torch.randn(3, 4)
    actual = result.plan(mx.array(x.numpy()), mx.array(w.numpy()))[0]
    projected = mx.matmul(mx.array(x.numpy()), mx.array(w.numpy()))
    expected = projected + projected
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected)


def test_common_subexpression_elimination_folds_a_multi_output_operation() -> None:
    source = TensorType((2, 6), DType.F32)
    normalized = TensorType((2, 6), DType.F32)
    statistic = TensorType((2, 1), DType.F32)
    values = {
        "x": Value("x", source),
        "first_out": Value("first_out", normalized),
        "first_mean": Value("first_mean", statistic),
        "second_out": Value("second_out", normalized),
        "second_mean": Value("second_mean", statistic),
        "summed": Value("summed", normalized),
        "out": Value("out", normalized),
    }
    attrs = {
        "normalized_shape": (6,),
        "eps": 1e-5,
        "has_weight": False,
        "has_bias": False,
    }
    graph = Graph(
        ("x",),
        values,
        (
            Operation("layer_norm", ("x",), ("first_out", "first_mean"), attrs),
            Operation("layer_norm", ("x",), ("second_out", "second_mean"), attrs),
            Operation("add", ("first_out", "second_out"), ("summed",)),
            Operation("mul", ("summed", "second_mean"), ("out",)),
        ),
        ("out",),
    )

    optimized = eliminate_common_subexpressions(graph)
    # Both calls reduce the same input the same way, so the second one is
    # dropped and *every* output of it is aliased onto the first one's - folding
    # only the first output would leave the statistics computed twice.
    assert [op.op for op in optimized.ops] == ["layer_norm", "add", "mul"]
    assert optimized.ops[0].outputs == ("first_out", "first_mean")
    assert optimized.ops[1].inputs == ("first_out", "first_out")
    assert optimized.ops[2].inputs == ("summed", "first_mean")


def test_common_subexpression_key_preserves_cast_dtype() -> None:
    source = TensorType((2, 3), DType.F16)
    first_type = TensorType((2, 3), DType.F32)
    second_type = TensorType((2, 3), DType.BF16)
    values = {
        "x": Value("x", source),
        "first": Value("first", first_type),
        "second": Value("second", second_type),
    }
    graph = Graph(
        ("x",),
        values,
        (
            Operation("cast", ("x",), ("first",)),
            Operation("cast", ("x",), ("second",)),
        ),
        ("first", "second"),
    )

    optimized = eliminate_common_subexpressions(graph)
    assert [op.op for op in optimized.ops] == ["cast", "cast"]
    assert optimized.outputs == ("first", "second")


def test_common_subexpression_key_preserves_attributes() -> None:
    tensor = TensorType((2, 3), DType.F32)
    values = {
        "x": Value("x", tensor),
        "y": Value("y", tensor),
        "first": Value("first", tensor),
        "second": Value("second", tensor),
    }
    graph = Graph(
        ("x", "y"),
        values,
        (
            Operation("add", ("x", "y"), ("first",), {"alpha": 1.0}),
            Operation("add", ("x", "y"), ("second",), {"alpha": 2.0}),
        ),
        ("first", "second"),
    )

    optimized = eliminate_common_subexpressions(graph)
    assert [op.op for op in optimized.ops] == ["add", "add"]
    assert optimized.outputs == ("first", "second")


def test_common_subexpression_elimination_normalizes_fused_region_values() -> None:
    tensor = TensorType((3,), DType.F32)
    values = {
        value_id: Value(value_id, tensor)
        for value_id in ("a", "b", "x", "y", "out0", "out1")
    }
    graph = Graph(
        ("a", "b"),
        values,
        (
            Operation("silu", ("a",), ("x",)),
            Operation("mul", ("x", "b"), ("out0",)),
            Operation("silu", ("a",), ("y",)),
            Operation("mul", ("y", "b"), ("out1",)),
        ),
        ("out0", "out1"),
    )

    optimized = eliminate_common_subexpressions(fuse_pointwise(graph))
    assert [op.op for op in optimized.ops] == ["fused_elementwise"]
    assert optimized.outputs == ("out0", "out0")

    result = compile_graph(optimized)
    torch.manual_seed(45)
    a = torch.randn(3)
    b = torch.randn(3)
    actual = result.plan(mx.array(a.numpy()), mx.array(b.numpy()))
    expected = mx.array((torch.nn.functional.silu(a) * b).numpy())
    mx.eval(*actual, expected)
    assert actual[0] is actual[1]
    assert mx.allclose(actual[0], expected, atol=1e-5, rtol=1e-5)


def test_common_subexpression_elimination_rewrites_fused_region_boundaries() -> None:
    matrix = TensorType((2, 3), DType.F32)
    weight = TensorType((3, 4), DType.F32)
    output = TensorType((2, 4), DType.F32)
    bias = TensorType((4,), DType.F32)
    values = {
        "x": Value("x", matrix),
        "w": Value("w", weight),
        "bias": Value("bias", bias),
        "left": Value("left", output),
        "right": Value("right", output),
        "left_activation": Value("left_activation", output),
        "right_activation": Value("right_activation", output),
        "out0": Value("out0", output),
        "out1": Value("out1", output),
    }
    graph = Graph(
        ("x", "w", "bias"),
        values,
        (
            Operation("matmul", ("x", "w"), ("left",)),
            Operation("silu", ("left",), ("left_activation",)),
            Operation("mul", ("left_activation", "bias"), ("out0",)),
            Operation("matmul", ("x", "w"), ("right",)),
            Operation("silu", ("right",), ("right_activation",)),
            Operation("mul", ("right_activation", "bias"), ("out1",)),
        ),
        ("out0", "out1"),
    )

    optimized = eliminate_common_subexpressions(fuse_pointwise(graph))
    assert [op.op for op in optimized.ops] == ["matmul", "fused_elementwise"]
    assert optimized.outputs == ("out0", "out0")
    body = optimized.ops[1].attrs["body"]
    assert body[0].inputs == ("left",)
    assert all("right" not in body_op.inputs for body_op in body)

    result = compile_graph(optimized)
    torch.manual_seed(46)
    inputs = (torch.randn(2, 3), torch.randn(3, 4), torch.randn(4))
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))
    projected = mx.matmul(mx.array(inputs[0].numpy()), mx.array(inputs[1].numpy()))
    expected = projected / (1.0 + mx.exp(-projected)) * mx.array(inputs[2].numpy())
    mx.eval(*actual, expected)
    assert actual[0] is actual[1]
    assert mx.allclose(actual[0], expected, atol=1e-5, rtol=1e-5)


def test_compiler_eliminates_duplicate_graph_outputs() -> None:
    def fn(x, weight):
        left = x @ weight
        right = x @ weight
        return left, right

    torch.manual_seed(44)
    x = torch.randn(2, 3)
    weight = torch.randn(3, 4)
    result = compile_callable(fn, (x, weight))

    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]
    assert result.lowered_graph.outputs == (result.lowered_graph.ops[0].outputs[0],) * 2

    actual = result.plan(mx.array(x.numpy()), mx.array(weight.numpy()))
    expected = tuple(mx.array(value.numpy()) for value in fn(x, weight))
    mx.eval(*actual, *expected)
    assert len(actual) == 2
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert mx.allclose(actual_value, expected_value, atol=5e-3, rtol=5e-3)


def test_layout_ops_match_pytorch() -> None:
    def fn(x):
        grouped = x.reshape(2, 3, 4, 5).transpose(1, 2).permute(0, 2, 1, 3)
        return (
            grouped.reshape(2, -1, 5),
            torch.ops.aten.alias.default(grouped).clone(),
            grouped[:, :, :, :1].expand(2, 3, 4, 5),
            grouped.unsqueeze(0).expand(2, 2, 3, 4, 5),
        )

    x = torch.randn(24, 5)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.input_graph.ops] == [
        "reshape",
        "transpose",
        "permute",
        "reshape",
        "alias",
        "contiguous",
        "slice",
        "expand",
        "unsqueeze",
        "expand",
    ]

    actual = result.plan(mx.array(x.numpy()))
    expected = fn(x)
    expected_arrays = tuple(mx.array(value.numpy()) for value in expected)
    mx.eval(*actual, *expected_arrays)
    for actual_value, expected_value in zip(actual, expected_arrays, strict=True):
        assert mx.allclose(actual_value, expected_value)


def test_identity_and_inverse_layouts_are_eliminated() -> None:
    def fn(x):
        return x.reshape(2, 3).transpose(0, 1).transpose(0, 1).reshape(3, 2).clone()

    x = torch.randn(2, 3)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["reshape"]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_squeeze_identity_and_inverse_are_eliminated() -> None:
    def fn(x):
        unchanged = x.squeeze(1)
        round_trip = x.unsqueeze(1).squeeze(1)
        return unchanged, round_trip

    x = torch.randn(2, 3)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.input_graph.ops] == [
        "squeeze",
        "unsqueeze",
        "squeeze",
    ]
    assert result.lowered_graph.ops == ()

    actual = result.plan(mx.array(x.numpy()))
    mx.eval(*actual)
    assert mx.array_equal(actual[0], mx.array(x.numpy()))
    assert mx.array_equal(actual[1], mx.array(x.numpy()))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_stack_lowers_to_unsqueeze_and_cat(dim: int) -> None:
    def fn(left, right):
        return torch.stack((left, right), dim=dim)

    torch.manual_seed(45)
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    result = compile_callable(fn, inputs)

    axis = dim if dim >= 0 else dim + 3
    assert [op.op for op in result.lowered_graph.ops] == [
        "unsqueeze",
        "unsqueeze",
        "cat",
    ]
    assert [op.attrs["dim"] for op in result.lowered_graph.ops] == [axis] * 3
    assert result.plan.kernels == ()

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    input_shape = inputs[0].shape
    assert actual.shape == input_shape[:axis] + (2,) + input_shape[axis:]
    assert mx.array_equal(actual, expected)


def test_cat_drops_a_zero_length_constant_operand() -> None:
    """A zero-length constant is a KV-cache placeholder, not a graph input.

    ``use_cache=True`` models start their cache from ``torch.tensor([])``, which
    make_fx lifts into a constant. The IR has no type for an empty tensor, but a
    zero-length concatenation operand contributes nothing, so dropping it must
    leave both the result and the rest of the graph untouched.
    """

    def fn(value):
        return torch.cat((torch.tensor([]), value), dim=0)

    torch.manual_seed(51)
    value = torch.randn(3, 4)
    result = compile_callable(fn, (value,))

    # Nothing survives: the concatenation is the identity on its only operand.
    assert result.lowered_graph.ops == ()

    actual = result.plan(mx.array(value.numpy()))[0]
    mx.eval(actual)
    assert mx.array_equal(actual, mx.array(value.numpy()))


def test_cat_keeps_the_concatenation_when_two_operands_survive() -> None:
    def fn(left, right):
        return torch.cat((torch.tensor([]), left, right), dim=0)

    torch.manual_seed(52)
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    result = compile_callable(fn, inputs)

    assert [op.op for op in result.lowered_graph.ops] == ["cat"]
    assert [op.attrs["dim"] for op in result.lowered_graph.ops] == [0]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_non_empty_constant_tensor_names_the_use_cache_hint() -> None:
    """A constant that cannot be folded must explain how it got there."""

    def fn(value):
        return value * torch.tensor([2.0, 3.0, 4.0])

    with pytest.raises(NotImplementedError) as excinfo:
        compile_callable(fn, (torch.randn(3),))

    message = str(excinfo.value)
    assert "_tensor_constant0" in message
    assert "(3,)" in message
    assert "use_cache" in message
    assert "compile_module" in message


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_embedding_uses_mlx_gather_without_a_kernel(index_dtype) -> None:
    def fn(indices, weight):
        return torch.nn.functional.embedding(indices, weight, padding_idx=0)

    torch.manual_seed(44)
    inputs = (
        torch.tensor([[0, 2], [3, 1]], dtype=index_dtype),
        torch.randn(4, 8, dtype=torch.float16),
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["embedding"]
    assert result.plan.kernels == ()

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert actual.shape == (2, 2, 8)
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(("dim", "index_dtype"), [(1, torch.int64), (-1, torch.int32)])
def test_index_select_uses_mlx_gather_without_a_kernel(dim, index_dtype) -> None:
    def fn(source, indices):
        return torch.index_select(source, dim=dim, index=indices)

    torch.manual_seed(45)
    inputs = (
        torch.randn(2, 3, 4, dtype=torch.float16),
        torch.tensor([2, 0], dtype=index_dtype),
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["index_select"]
    assert result.lowered_graph.ops[0].attrs["dim"] == dim % 3
    assert result.plan.kernels == ()

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert actual.shape == tuple(fn(*inputs).shape)
    assert mx.array_equal(actual, expected)


def test_index_select_rejects_out_of_range_indices() -> None:
    def fn(source, indices):
        return torch.index_select(source, dim=1, index=indices)

    source = torch.randn(2, 3, 4)
    indices = torch.tensor([2, 0], dtype=torch.int64)
    result = compile_callable(fn, (source, indices))
    source_mlx = _torch_to_mlx(source)

    for invalid in (torch.tensor([3, 0]), torch.tensor([-1, 0])):
        with pytest.raises(IndexError, match="index out of range"):
            result.plan(source_mlx, _torch_to_mlx(invalid))


def test_deferred_bounds_checks_cover_every_lookup() -> None:
    """Queued bounds checks must all run, not just the first one."""

    def fn(left, left_indices, right, right_indices):
        return torch.nn.functional.embedding(
            left_indices, left
        ) + torch.nn.functional.embedding(right_indices, right)

    left = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    right = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    left_indices = torch.tensor([0, 3], dtype=torch.int64)
    right_indices = torch.tensor([0, 1], dtype=torch.int64)
    result = compile_callable(fn, (left, left_indices, right, right_indices))

    valid_inputs = (left, left_indices, right, right_indices)
    actual = result.plan(*(_torch_to_mlx(value) for value in valid_inputs))[0]
    expected = _torch_to_mlx(fn(*valid_inputs))
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)

    overflowing = (left, left_indices, right, torch.tensor([0, 6]))
    with pytest.raises(IndexError, match="index out of range"):
        result.plan(*(_torch_to_mlx(value) for value in overflowing))


def test_deferred_bounds_checks_reduce_before_combining() -> None:
    """Queued lookups can have different index shapes; their masks cannot mix.

    Combining the raw masks makes the flush itself raise a broadcast error, so
    the plan fails on valid indices instead of returning the answer.
    """

    def fn(left, left_indices, right, right_indices):
        return (
            torch.nn.functional.embedding(left_indices, left),
            torch.nn.functional.embedding(right_indices, right),
        )

    left = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    right = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    left_indices = torch.tensor([0, 2, 1], dtype=torch.int64)
    right_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    result = compile_callable(fn, (left, left_indices, right, right_indices))

    valid = (left, left_indices, right, right_indices)
    outputs = result.plan(*(_torch_to_mlx(value) for value in valid))
    expected = tuple(_torch_to_mlx(value) for value in fn(*valid))
    mx.eval(*outputs, *expected)
    assert len(outputs) == len(expected) == 2
    for actual, want in zip(outputs, expected, strict=True):
        assert mx.array_equal(actual, want)

    overflowing = (
        left,
        torch.tensor([0, 3, 1]),
        right,
        torch.tensor([[0, 1], [2, 3]]),
    )
    with pytest.raises(IndexError, match="index out of range"):
        result.plan(*(_torch_to_mlx(value) for value in overflowing))


@pytest.mark.parametrize(
    ("dim", "index_dtype", "values"),
    [
        (0, torch.int32, [[2, 0], [1, 2]]),
        (-1, torch.int64, [[3, 2], [0, 1]]),
    ],
)
def test_gather_uses_mlx_without_a_kernel(dim, index_dtype, values) -> None:
    def fn(source, indices):
        return torch.gather(source, dim, indices)

    torch.manual_seed(46)
    source = torch.randn(3, 4, dtype=torch.float16)
    indices = torch.tensor(values, dtype=index_dtype)
    result = compile_callable(fn, (source, indices))
    assert [op.op for op in result.lowered_graph.ops] == ["gather"]
    assert result.lowered_graph.ops[0].attrs["dim"] == dim % 2
    assert result.plan.kernels == ()

    actual = result.plan(_torch_to_mlx(source), _torch_to_mlx(indices))[0]
    expected = _torch_to_mlx(fn(source, indices))
    mx.eval(actual, expected)
    assert actual.shape == tuple(fn(source, indices).shape)
    assert mx.array_equal(actual, expected)


def test_gather_rejects_out_of_range_indices() -> None:
    def fn(source, indices):
        return torch.gather(source, dim=-1, index=indices)

    source = torch.randn(3, 4)
    indices = torch.tensor([[3, 2], [0, 1]], dtype=torch.int64)
    result = compile_callable(fn, (source, indices))
    source_mlx = _torch_to_mlx(source)

    for invalid in (
        torch.tensor([[4, 0], [0, 0]]),
        torch.tensor([[-1, 0], [0, 0]]),
    ):
        with pytest.raises(IndexError, match="index out of range"):
            result.plan(source_mlx, _torch_to_mlx(invalid))


@pytest.mark.parametrize(
    ("dim", "index_dtype", "index_values"),
    [
        (1, torch.int32, [[[2, 0], [1, 2]]]),
        (-1, torch.int64, [[[3, 2], [1, 0], [2, 3]]]),
    ],
)
def test_scatter_src_uses_mlx_without_a_kernel(dim, index_dtype, index_values) -> None:
    def fn(source, indices, update):
        return source.scatter(dim, indices, update)

    torch.manual_seed(47)
    source = torch.randn(2, 3, 4, dtype=torch.float16)
    indices = torch.tensor(index_values, dtype=index_dtype)
    update = torch.randn(indices.shape, dtype=torch.float16)
    result = compile_callable(fn, (source, indices, update))

    assert [op.op for op in result.lowered_graph.ops] == ["scatter"]
    assert result.lowered_graph.ops[0].attrs["dim"] == dim % source.ndim
    assert result.plan.kernels == ()

    actual = result.plan(
        _torch_to_mlx(source),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    assert actual.shape == source.shape
    assert actual.dtype == _MLX_DTYPES[source.dtype]
    assert mx.array_equal(actual, expected)


def test_scatter_src_uses_index_shaped_prefix_of_larger_source() -> None:
    def fn(source, indices, update):
        return source.scatter(1, indices, update)

    source = torch.zeros(2, 4, dtype=torch.float32)
    indices = torch.tensor([[1, 3], [0, 2]], dtype=torch.int64)
    update = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    result = compile_callable(fn, (source, indices, update))

    actual = result.plan(
        _torch_to_mlx(source),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_scatter_value_uses_mlx_without_a_kernel() -> None:
    def fn(source, indices):
        return source.scatter(-1, indices, -1.5)

    source = torch.zeros(2, 4, dtype=torch.float16)
    indices = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    result = compile_callable(fn, (source, indices))

    assert [op.op for op in result.lowered_graph.ops] == ["scatter"]
    assert result.lowered_graph.ops[0].attrs == {"dim": 1, "value": -1.5}
    assert result.plan.kernels == ()

    actual = result.plan(_torch_to_mlx(source), _torch_to_mlx(indices))[0]
    expected = _torch_to_mlx(fn(source, indices))
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_scatter_does_not_mutate_the_input_tensor() -> None:
    def fn(source, indices, update):
        return source.scatter(0, indices, update)

    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    indices = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
    update = torch.tensor([[20, 21], [22, 23]], dtype=torch.float32)
    result = compile_callable(fn, (source, indices, update))
    source_mlx = _torch_to_mlx(source)
    original = _torch_to_mlx(source)

    actual = result.plan(source_mlx, _torch_to_mlx(indices), _torch_to_mlx(update))[0]
    mx.eval(actual, source_mlx, original)
    assert mx.array_equal(source_mlx, original)
    assert mx.array_equal(actual, _torch_to_mlx(fn(source, indices, update)))


@pytest.mark.parametrize("invalid", [[-1, 0], [4, 0]])
def test_scatter_rejects_out_of_range_indices(invalid) -> None:
    def fn(source, indices):
        return source.scatter(-1, indices, 1.0)

    source = torch.zeros(3, 4)
    indices = torch.tensor([[3, 2], [0, 1]], dtype=torch.int64)
    result = compile_callable(fn, (source, indices))
    source_mlx = _torch_to_mlx(source)

    with pytest.raises(IndexError, match="index out of range"):
        result.plan(
            source_mlx,
            _torch_to_mlx(torch.tensor([invalid, [0, 1]], dtype=torch.int64)),
        )


def test_scatter_rejects_source_smaller_than_indices() -> None:
    source_type = TensorType((2, 4), DType.F32)
    index_type = TensorType((2, 3), DType.I64)
    update_type = TensorType((2, 2), DType.F32)
    values = {
        "source": Value("source", source_type),
        "indices": Value("indices", index_type),
        "update": Value("update", update_type),
        "out": Value("out", source_type),
    }
    graph = Graph(
        ("source", "indices", "update"),
        values,
        (
            Operation(
                "scatter",
                ("source", "indices", "update"),
                ("out",),
                {"dim": 1},
            ),
        ),
        ("out",),
    )

    with pytest.raises(ValueError, match="source shape .* smaller than indices shape"):
        compile_graph(graph)


def test_scatter_add_accumulates_duplicate_indices_instead_of_overwriting() -> None:
    """`scatter_add` is the reduction form: two writes to one slot both count.

    A duplicated index is exactly where an implementation that quietly reuses
    the overwrite path still returns a plausible-looking tensor.
    """

    def fn(source, indices, update):
        return source.scatter_add(0, indices, update)

    source = torch.arange(12, dtype=torch.float16).reshape(4, 3)
    # Row 1 is targeted twice, row 3 not at all.
    indices = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 2, 1]], dtype=torch.int64)
    update = torch.arange(9, dtype=torch.float16).reshape(3, 3)
    result = compile_callable(fn, (source, indices, update))

    assert [op.op for op in result.lowered_graph.ops] == ["scatter"]
    assert result.lowered_graph.ops[0].attrs == {
        "dim": 0,
        "reduce": "add",
        "include_self": True,
    }
    assert result.plan.kernels == ()

    actual = result.plan(
        _torch_to_mlx(source),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    assert actual.shape == source.shape
    assert actual.dtype == _MLX_DTYPES[source.dtype]
    # fp16 accumulation order is the device's, so compare with a tolerance.
    assert mx.allclose(actual, expected, atol=1e-2, rtol=1e-2)
    # The untouched row keeps its original value.
    assert mx.array_equal(actual[3], _torch_to_mlx(source)[3])


def test_moe_combine_routes_tokens_with_index_add() -> None:
    """The MoE combine path: weighted tokens accumulated per expert slot.

    Two tokens route to the same expert, which is the case a per-expert
    overwrite would get wrong.
    """

    def fn(tokens, expert_ids, weights):
        contributions = tokens * weights[:, None]
        combined = torch.zeros(4, tokens.shape[-1], dtype=tokens.dtype)
        return combined.index_add(0, expert_ids, contributions)

    torch.manual_seed(50)
    tokens = torch.randn(5, 3)
    expert_ids = torch.tensor([0, 1, 0, 3, 0], dtype=torch.int64)
    weights = torch.tensor([0.5, 1.5, -0.25, 2.0, 0.75])
    result = compile_callable(fn, (tokens, expert_ids, weights))

    assert "index_add" in [op.op for op in result.lowered_graph.ops]

    actual = result.plan(
        _torch_to_mlx(tokens),
        _torch_to_mlx(expert_ids),
        _torch_to_mlx(weights),
    )[0]
    expected = _torch_to_mlx(fn(tokens, expert_ids, weights))
    mx.eval(actual, expected)
    assert actual.shape == (4, 3)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    # Expert 2 receives no token and stays exactly zero.
    assert mx.array_equal(actual[2], mx.zeros((3,), dtype=mx.float32))


def test_moe_combine_written_with_inplace_index_add_matches_pytorch() -> None:
    """The same combine written with ``index_add_`` is legalized, not rejected."""

    def fn(tokens, expert_ids, weights):
        contributions = tokens * weights[:, None]
        combined = torch.zeros(4, tokens.shape[-1], dtype=tokens.dtype)
        combined.index_add_(0, expert_ids, contributions)
        return combined

    torch.manual_seed(51)
    tokens = torch.randn(5, 3)
    expert_ids = torch.tensor([0, 1, 0, 3, 0], dtype=torch.int64)
    weights = torch.tensor([0.5, 1.5, -0.25, 2.0, 0.75])
    result = compile_callable(fn, (tokens, expert_ids, weights))

    assert "index_add" in [op.op for op in result.lowered_graph.ops]

    actual = result.plan(
        _torch_to_mlx(tokens),
        _torch_to_mlx(expert_ids),
        _torch_to_mlx(weights),
    )[0]
    expected = _torch_to_mlx(fn(tokens, expert_ids, weights))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert mx.array_equal(actual[2], mx.zeros((3,), dtype=mx.float32))


@pytest.mark.parametrize(
    ("reduce", "expected_reduce"),
    [("sum", "add"), ("prod", "prod"), ("amax", "amax"), ("amin", "amin")],
)
def test_scatter_reduce_covers_every_supported_reduction(
    reduce: str, expected_reduce: str
) -> None:
    def fn(source, indices, update):
        return source.scatter_reduce(1, indices, update, reduce=reduce)

    torch.manual_seed(48)
    source = torch.randn(2, 4)
    indices = torch.tensor([[0, 2, 2, 3], [1, 1, 0, 3]], dtype=torch.int64)
    update = torch.randn(2, 4)
    result = compile_callable(fn, (source, indices, update))

    assert result.lowered_graph.ops[0].attrs["reduce"] == expected_reduce

    actual = result.plan(
        _torch_to_mlx(source),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_scatter_reduce_without_self_ignores_the_base_tensor() -> None:
    def fn(source, indices, update):
        return source.scatter_reduce(
            0, indices, update, reduce="amax", include_self=False
        )

    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    update = torch.tensor([[5.0, -2.0], [3.0, 7.0]])
    base = torch.full((3, 2), 100.0)
    other_base = torch.full((3, 2), -100.0)
    result = compile_callable(fn, (base, indices, update))
    assert result.lowered_graph.ops[0].attrs["include_self"] is False

    first = result.plan(
        _torch_to_mlx(base),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    second = result.plan(
        _torch_to_mlx(other_base),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    expected = _torch_to_mlx(fn(base, indices, update))
    mx.eval(first, second, expected)
    assert mx.array_equal(first, expected)
    # Every indexed row is the reduction of the updates alone, so the base only
    # survives outside the scattered region.
    assert mx.array_equal(first[:2], second[:2])
    assert not mx.array_equal(first[2], second[2])
    assert mx.array_equal(first[2], _torch_to_mlx(base)[2])


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_index_add_accumulates_the_update_along_one_axis(dim: int) -> None:
    def fn(source, indices, update):
        return torch.index_add(source, dim, indices, update)

    torch.manual_seed(49)
    source = torch.randn(4, 5)
    axis = dim % source.ndim
    indices = torch.tensor([0, 3, 0], dtype=torch.int64)
    update_shape = list(source.shape)
    update_shape[axis] = 3
    update = torch.randn(update_shape)
    result = compile_callable(fn, (source, indices, update))

    assert [op.op for op in result.lowered_graph.ops] == ["index_add"]
    assert result.lowered_graph.ops[0].attrs == {"dim": axis, "alpha": 1.0}

    actual = result.plan(
        _torch_to_mlx(source),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_index_add_alpha_scales_each_update() -> None:
    def fn(source, indices, update):
        return torch.index_add(source, 0, indices, update, alpha=2.5)

    source = torch.zeros(4, 3)
    indices = torch.tensor([1, 1, 3], dtype=torch.int32)
    update = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    result = compile_callable(fn, (source, indices, update))
    assert result.lowered_graph.ops[0].attrs["alpha"] == 2.5

    actual = result.plan(
        _torch_to_mlx(source),
        _torch_to_mlx(indices),
        _torch_to_mlx(update),
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_index_add_does_not_mutate_the_base_tensor() -> None:
    def fn(source, indices, update):
        return torch.index_add(source, 0, indices, update)

    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    update = torch.ones(2, 3)
    result = compile_callable(fn, (source, indices, update))
    source_mlx = _torch_to_mlx(source)
    original = _torch_to_mlx(source)

    actual = result.plan(source_mlx, _torch_to_mlx(indices), _torch_to_mlx(update))[0]
    mx.eval(actual, source_mlx, original)
    assert mx.array_equal(source_mlx, original)
    assert mx.array_equal(actual, _torch_to_mlx(fn(source, indices, update)))


@pytest.mark.parametrize("invalid", [[-1, 0], [4, 0]])
def test_index_add_rejects_out_of_range_indices(invalid) -> None:
    def fn(source, indices, update):
        return torch.index_add(source, 0, indices, update)

    source = torch.zeros(3, 2)
    indices = torch.tensor([0, 2], dtype=torch.int64)
    update = torch.ones(2, 2)
    result = compile_callable(fn, (source, indices, update))

    with pytest.raises(IndexError, match="index out of range"):
        result.plan(
            _torch_to_mlx(source),
            _torch_to_mlx(torch.tensor(invalid, dtype=torch.int64)),
            _torch_to_mlx(update),
        )


def test_index_add_rejects_a_multi_dimensional_index() -> None:
    source_type = TensorType((4, 3), DType.F32)
    index_type = TensorType((2, 2), DType.I64)
    value_type = TensorType((2, 2, 3), DType.F32)
    values = {
        "source": Value("source", source_type),
        "indices": Value("indices", index_type),
        "update": Value("update", value_type),
        "out": Value("out", source_type),
    }
    graph = Graph(
        ("source", "indices", "update"),
        values,
        (
            Operation(
                "index_add",
                ("source", "indices", "update"),
                ("out",),
                {"dim": 0, "alpha": 1.0},
            ),
        ),
        ("out",),
    )

    with pytest.raises(NotImplementedError, match="1-D index tensor"):
        compile_graph(graph)


def test_scatter_reduce_mean_divides_by_the_number_of_writes() -> None:
    """`mean` accumulates the values and a matching write count, then divides.

    Index 0 receives two writes in row 0, which is the case a single
    accumulation would get wrong, and row 3 receives none.
    """

    def fn(source, indices, update):
        return source.scatter_reduce(0, indices, update, reduce="mean")

    source = torch.zeros(4, 2)
    indices = torch.tensor([[0, 1], [0, 1], [2, 0]], dtype=torch.int64)
    update = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = compile_callable(fn, (source, indices, update))

    assert "scatter" in [op.op for op in result.lowered_graph.ops]

    actual = result.plan(
        _torch_to_mlx(source), _torch_to_mlx(indices), _torch_to_mlx(update)
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    # Row 0 column 0 is (1 + 3) / 2, row 2 column 0 is 5 / 1, row 3 is
    # untouched and keeps the zero source value.
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert mx.allclose(actual[3], mx.zeros((2,), dtype=mx.float32))


def test_scatter_reduce_mean_without_self_only_counts_writes() -> None:
    def fn(source, indices, update):
        return source.scatter_reduce(
            1, indices, update, reduce="mean", include_self=False
        )

    source = torch.full((3, 4), 7.0)
    indices = torch.tensor([[0, 1], [2, 2], [0, 0]], dtype=torch.int64)
    update = torch.tensor([[1.0, 3.0], [5.0, 9.0], [2.0, 4.0]])
    result = compile_callable(fn, (source, indices, update))

    actual = result.plan(
        _torch_to_mlx(source), _torch_to_mlx(indices), _torch_to_mlx(update)
    )[0]
    expected = _torch_to_mlx(fn(source, indices, update))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    # Columns 1-3 received no write in row 2 and keep the source value.
    assert mx.allclose(actual[2][1:], mx.array([7.0, 7.0, 7.0]))


def test_scatter_reduce_mean_rejects_integer_inputs() -> None:
    def fn(source, indices, update):
        return source.scatter_reduce(0, indices, update, reduce="mean")

    with pytest.raises(NotImplementedError, match="needs a floating-point input"):
        compile_callable(
            fn,
            (
                torch.zeros(4, dtype=torch.int64),
                torch.tensor([0, 1], dtype=torch.int64),
                torch.ones(2, dtype=torch.int64),
            ),
        )


@pytest.mark.parametrize(
    ("shape", "pads", "value"),
    [
        ((3,), (2, 1), 0.5),
        ((2, 3), (1, 1, 2, 0), -1.25),
        ((2, 3, 4), (1, 0, 2, 1, 0, 3), 2.5),
    ],
)
def test_constant_pad_matches_pytorch(shape, pads, value) -> None:
    def fn(x):
        return torch.nn.functional.pad(x, pads, value=value)

    torch.manual_seed(50)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["pad"]
    assert result.plan.kernels == ()

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected)


def test_constant_pad_fills_only_the_requested_border() -> None:
    def fn(x):
        # Padding pairs index the trailing dimensions first, so this pads the
        # rows with two copies on top and one below while the columns are
        # untouched.
        return torch.nn.functional.pad(x, (0, 0, 2, 1), value=-3.0)

    torch.manual_seed(51)
    x = torch.randn(2, 4)
    result = compile_callable(fn, (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)
    source = mx.array(x.numpy())
    assert mx.array_equal(actual[:2], mx.full((2, 4), -3.0, dtype=actual.dtype))
    assert mx.array_equal(actual[-1], mx.full((4,), -3.0, dtype=actual.dtype))
    assert mx.array_equal(actual[2:4], source)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (torch.float16, 1.5),
        (torch.float32, 1.5),
        (torch.bfloat16, 1.5),
        (torch.int64, 4),
        (torch.bool, True),
    ],
)
def test_constant_pad_preserves_the_source_dtype(dtype, value) -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (1, 2, 1, 1), value=value)

    torch.manual_seed(52)
    if dtype == torch.bool:
        x = torch.rand(3, 2) > 0.5
    elif dtype.is_floating_point:
        x = torch.randn(3, 2).to(dtype)
    else:
        x = torch.randint(0, 8, (3, 2)).to(dtype)
    result = compile_callable(fn, (x,))
    actual = result.plan(_torch_to_mlx(x))[0]
    expected = mx.array(fn(x).float().numpy()).astype(actual.dtype)
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_constant_pad_with_no_growth_is_the_input() -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (0, 0, 0, 0), value=7.0)

    x = torch.randn(2, 3)
    result = compile_callable(fn, (x,))
    assert result.lowered_graph.ops == ()
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert mx.array_equal(actual, mx.array(x.numpy()))


def test_constant_pad_validates_pairs_against_the_rank_at_runtime() -> None:
    values = {
        "x": Value("x", TensorType((2, 3), DType.F32)),
        "out": Value("out", TensorType((3, 4), DType.F32)),
    }
    graph = Graph(
        ("x",),
        values,
        (Operation("pad", ("x",), ("out",), {"pads": (1, 1, 1, 1, 1, 1)}),),
        ("out",),
    )
    plan = compile_graph(graph).plan
    with pytest.raises(ValueError, match="6 values for a rank-2 input"):
        plan(mx.zeros((2, 3)))


@pytest.mark.parametrize(
    ("shape", "pads", "value"),
    [
        ((2, 6), (-1, -2), 0.5),
        ((2, 6), (-2, 0), -1.0),
        ((2, 6), (-1, 2), 3.0),
        ((2, 6), (2, -1), 3.0),
        ((2, 3, 4), (-1, 1, 0, -1), 2.0),
        ((2, 3, 4), (1, -1, -2, 2), 0.0),
    ],
)
def test_constant_pad_crops_negative_entries(shape, pads, value) -> None:
    # ATen derives the fill region from the distance past the source, so a
    # negative entry crops the padded extent instead of failing.
    def fn(x):
        return torch.nn.functional.pad(x, pads, value=value)

    torch.manual_seed(56)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    assert {op.op for op in result.lowered_graph.ops} <= {"slice", "pad"}
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected)


def test_constant_pad_crop_and_pad_share_one_axis() -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (-1, 2), value=-7.0)

    source = mx.array([[1.0, 2.0, 3.0, 4.0]])
    result = compile_callable(fn, (torch.zeros(1, 4),))
    actual = result.plan(source)[0]
    mx.eval(actual)
    # One element is cropped from the front, then two fills are appended, which
    # is what `F.pad(x, (-1, 2))` produces.
    assert actual.tolist() == [[2.0, 3.0, 4.0, -7.0, -7.0]]
    assert [op.op for op in result.lowered_graph.ops] == ["slice", "pad"]


def test_constant_pad_with_only_crops_is_a_slice() -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (-1, -1))

    torch.manual_seed(57)
    x = torch.randn(3, 6)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["slice"]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_constant_pad_rejects_a_crop_that_empties_the_axis() -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (-2, -1))

    with pytest.raises(NotImplementedError, match="cannot represent"):
        compile_callable(fn, (torch.randn(1, 3),))


@pytest.mark.parametrize(
    ("shape", "pads", "mode"),
    [
        ((3, 8), (2, 1), "reflect"),
        ((4, 6), (0, 3), "reflect"),
        ((3, 8), (3, 2), "replicate"),
        ((2, 5, 7), (1, 2, 0, 1), "reflect"),
        ((2, 5, 7), (0, 2, 3, 1), "replicate"),
        ((2, 3, 4, 5), (1, 0, 2, 1), "reflect"),
        ((2, 3, 4, 5), (2, 2, 1, 3), "replicate"),
    ],
)
def test_reflect_and_replicate_pad_match_pytorch(shape, pads, mode) -> None:
    def fn(x):
        return torch.nn.functional.pad(x, pads, mode=mode)

    torch.manual_seed(53)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    # The padding is a pure view chain: no kernel copies the border.
    assert result.plan.kernels == ()
    assert {op.op for op in result.lowered_graph.ops} <= {
        "slice",
        "flip",
        "expand",
        "cat",
    }
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected)


def test_reflect_pad_mirrors_the_border_without_repeating_it() -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (2, 2), mode="reflect")

    source = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = compile_callable(fn, (torch.zeros(1, 5),))
    actual = result.plan(source)[0]
    mx.eval(actual)
    # [1, 2, 3, 4, 5] padded by two mirrors the inner values on both sides and
    # never writes the edge value twice in a row, which is what separates
    # reflection from replication.
    assert actual.tolist() == [[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]]


def test_replicate_pad_repeats_the_border() -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (2, 1), mode="replicate")

    source = mx.array([[1.0, 2.0, 3.0]])
    result = compile_callable(fn, (torch.zeros(1, 3),))
    actual = result.plan(source)[0]
    mx.eval(actual)
    assert actual.tolist() == [[1.0, 1.0, 1.0, 2.0, 3.0, 3.0]]


@pytest.mark.parametrize(
    ("shape", "pads", "mode"),
    [
        ((2, 6), (-1, -2), "reflect"),
        ((2, 6), (0, -2), "reflect"),
        ((2, 6), (-2, -1), "replicate"),
        ((2, 4, 6), (-1, 2, 1, -1), "replicate"),
        ((2, 4, 6), (-2, 1, 0, 0), "reflect"),
        ((3, 5, 7), (1, -2, -1, 2), "reflect"),
    ],
)
def test_reflect_and_replicate_pad_crop_like_pytorch(shape, pads, mode) -> None:
    # Negative entries crop the padded extent, so a mirror window stays anchored
    # on the uncropped border instead of the surviving slice.
    def fn(x):
        return torch.nn.functional.pad(x, pads, mode=mode)

    torch.manual_seed(54)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    assert result.plan.kernels == ()
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected)


def test_circular_pad_wraps_the_border() -> None:
    def fn(x):
        return torch.nn.functional.pad(x, (2, 1), mode="circular")

    source = mx.array([[1.0, 2.0, 3.0]])
    result = compile_callable(fn, (torch.zeros(1, 3),))
    actual = result.plan(source)[0]
    mx.eval(actual)
    # Wrap-around takes the far border rather than mirroring or repeating the
    # near one, which is what separates circular padding from the other modes.
    assert actual.tolist() == [[2.0, 3.0, 1.0, 2.0, 3.0, 1.0]]


@pytest.mark.parametrize(
    ("shape", "pads"),
    [
        ((3, 8), (2, 1)),
        ((4, 6), (3, 3)),
        ((2, 5, 7), (1, 2)),
        ((2, 3, 4, 5), (1, 2, 2, 1)),
        ((1, 1, 5, 5), (2, 2, 1, 1)),
        ((2, 6), (-1, -2)),
        ((3, 5, 7), (1, -2, 0, 2)),
    ],
)
def test_circular_pad_matches_pytorch(shape, pads) -> None:
    # Torch builds circular padding as an allocation it fills a slice at a
    # time. The lowering rebuilds that buffer as a cat grid instead, so this
    # also covers the in-place write that no rebinding could represent.
    def fn(x):
        return torch.nn.functional.pad(x, pads, mode="circular")

    torch.manual_seed(56)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    # Every piece is a view or a cat. The single allocation kernel is the
    # buffer Torch opens with `new_empty`, which the first write's pieces still
    # read before later writes cover them.
    assert {op.op for op in result.lowered_graph.ops} <= {"slice", "cat", "zeros"}
    assert all(kernel.kind == "zeros" for kernel in result.plan.kernels)
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected)


def test_write_through_a_view_refuses_when_the_storage_is_read_elsewhere() -> None:
    def fn(x):
        written = x[0].t()
        other = x[1]
        written.copy_(torch.zeros(3, 4))
        return written + other.sum()

    with pytest.raises(NotImplementedError, match="in-place write through a view"):
        compile_callable(fn, (torch.randn(2, 4, 3),))


def test_write_through_a_view_rebinds_when_nothing_else_reads_it() -> None:
    def fn(x):
        written = x[0].t()
        written.copy_(torch.zeros(3, 4))
        return written.T

    values = torch.randn(2, 4, 3)
    result = compile_callable(fn, (values,))
    actual = result.plan(mx.array(values.numpy()))[0]
    expected = mx.array(fn(values).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_reflect_pad_composes_with_a_matmul_tail() -> None:
    # A padded activation feeding a linear layer stays in one graph, so the
    # border pieces are concatenated once and read straight into the kernel.
    def fn(x, w):
        return torch.nn.functional.pad(x, (1, 2), mode="reflect") @ w

    torch.manual_seed(55)
    x = torch.randn(3, 4)
    w = torch.randn(7, 5)
    result = compile_callable(fn, (x, w))
    actual = result.plan(mx.array(x.numpy()), mx.array(w.numpy()))[0]
    expected = mx.array(fn(x, w).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    # A matmul tail accumulates in a different order than torch, so compare the
    # fused graph at a relative tolerance that still catches a wrong border.
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "fn",
    [
        lambda a, b: a / b,
        lambda a, b: a / 4,
        lambda a, b: b / a,
    ],
)
def test_integer_true_division_promotes_to_float(fn) -> None:
    a = torch.tensor([4, 6, 9])
    b = torch.tensor([2, 4, 3])
    result = compile_callable(fn, (a, b))
    assert "cast" in [op.op for op in result.lowered_graph.ops]
    actual = result.plan(_torch_to_mlx(a), _torch_to_mlx(b))[0]
    expected_value = fn(a, b)
    expected = mx.array(expected_value.numpy()).astype(actual.dtype)
    mx.eval(actual, expected)
    assert expected_value.dtype == torch.float32
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    "fn",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a // b,
        lambda a, b: a % b,
        lambda a, b: torch.maximum(a, b),
        lambda a, b: a < b,
        lambda a, b: a == b,
    ],
)
def test_mixed_integer_pointwise_promotes_to_int64(fn) -> None:
    a = torch.tensor([7, 5, 9], dtype=torch.int32)
    b = torch.tensor([2, 4, 9], dtype=torch.int64)
    result = compile_callable(fn, (a, b))
    ops = [op.op for op in result.lowered_graph.ops]
    assert ops[0] == "cast"
    actual = result.plan(_torch_to_mlx(a), _torch_to_mlx(b))[0]
    expected = mx.array(fn(a, b).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    "fn",
    [
        lambda a, b: a + b,
        lambda a, b: a * b,
        lambda a, b: a < b,
    ],
)
def test_integer_operand_with_float_promotes_to_the_float_dtype(fn) -> None:
    a = torch.tensor([1, 5, 9])
    b = torch.tensor([1.5, 4.25, 9.0])
    result = compile_callable(fn, (a, b))
    actual = result.plan(_torch_to_mlx(a), _torch_to_mlx(b))[0]
    expected_value = fn(a, b)
    expected = mx.array(expected_value.numpy())
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_where_promotes_mixed_branch_dtypes() -> None:
    def fn(condition, left, right):
        return torch.where(condition, left, right)

    condition = torch.tensor([True, False, True])
    left = torch.tensor([1, 2, 3], dtype=torch.int32)
    right = torch.tensor([4, 5, 6], dtype=torch.int64)
    result = compile_callable(fn, (condition, left, right))
    ops = [op.op for op in result.lowered_graph.ops]
    assert ops == ["cast", "where"]
    actual = result.plan(
        _torch_to_mlx(condition), _torch_to_mlx(left), _torch_to_mlx(right)
    )[0]
    expected = mx.array(fn(condition, left, right).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_matching_dtypes_are_not_cast() -> None:
    def fn(a, b):
        return (a + b) * b

    a = torch.tensor([1, 2], dtype=torch.int64)
    b = torch.tensor([3, 4], dtype=torch.int64)
    result = compile_callable(fn, (a, b))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    actual = result.plan(_torch_to_mlx(a), _torch_to_mlx(b))[0]
    mx.eval(actual)
    assert mx.array_equal(actual, mx.array(fn(a, b).numpy()))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_pow_matches_pytorch_exactly(dtype) -> None:
    def fn(base, exponent):
        return base**exponent

    base = torch.tensor([2, 3, -2, 0, 7], dtype=dtype)
    exponent = torch.tensor([3, 4, 3, 0, 1], dtype=dtype)
    result = compile_callable(fn, (base, exponent))
    assert [op.op for op in result.lowered_graph.ops] == ["pow"]
    actual = result.plan(_torch_to_mlx(base), _torch_to_mlx(exponent))[0]
    expected = mx.array(fn(base, exponent).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("fn", "values"),
    [
        (lambda base: base**3, [2, -3, 0]),
        (lambda base: base**0, [2, -3, 0]),
        (lambda base: 2**base, [3, 4, 0]),
    ],
)
def test_integer_pow_accepts_a_scalar_operand(fn, values) -> None:
    base = torch.tensor(values, dtype=torch.int64)
    result = compile_callable(fn, (base,))
    actual = result.plan(_torch_to_mlx(base))[0]
    expected = mx.array(fn(base).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_integer_pow_wraps_like_pytorch() -> None:
    def fn(base):
        return base**70

    base = torch.tensor([2, 3], dtype=torch.int64)
    result = compile_callable(fn, (base,))
    actual = result.plan(_torch_to_mlx(base))[0]
    expected_value = fn(base)
    expected = mx.array(expected_value.numpy())
    mx.eval(actual, expected)
    assert expected_value.tolist() == [0, 2293070008301402073]
    assert mx.array_equal(actual, expected)


def test_integer_pow_rejects_a_negative_scalar_exponent() -> None:
    def fn(base):
        return base**-1

    with pytest.raises(ValueError, match="negative exponent raises"):
        compile_callable(fn, (torch.tensor([2], dtype=torch.int64),))


@pytest.mark.parametrize(
    ("fn", "shape", "op_name"),
    [
        (lambda x: torch.prod(x, dim=-1), (2, 3, 4), "prod"),
        (lambda x: torch.prod(x, dim=1, keepdim=True), (2, 3, 4), "prod"),
        (lambda x: torch.prod(x), (2, 3), "prod"),
        (lambda x: torch.amax(x, dim=1), (2, 3, 4), "amax"),
        (lambda x: torch.amax(x, dim=(0, 2), keepdim=True), (2, 3, 4), "amax"),
        (lambda x: torch.amin(x, dim=-2), (2, 3, 4), "amin"),
        (lambda x: torch.amin(x, dim=(1, 2)), (2, 3, 4), "amin"),
    ],
)
def test_reduction_family_matches_pytorch(fn, shape, op_name) -> None:
    torch.manual_seed(53)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    assert result.plan.kernels == ()
    assert [selection.implementation for selection in result.plan.selections] == ["mlx"]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    if op_name == "prod":
        # Multiplying in a different order rounds a different intermediate,
        # so a product only matches PyTorch within a few ulps.
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    else:
        assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("fn", "values"),
    [
        (lambda x: torch.prod(x, dim=1), [[2, 3], [4, 5]]),
        (lambda x: torch.amax(x, dim=0), [[3, 1], [4, 2]]),
        (lambda x: torch.amin(x, dim=1), [[3, -1], [-4, 2]]),
    ],
)
def test_integer_reductions_match_pytorch(fn, values) -> None:
    x = torch.tensor(values, dtype=torch.int64)
    result = compile_callable(fn, (x,))
    actual = result.plan(_torch_to_mlx(x))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_prod_wraps_integer_overflow_like_pytorch() -> None:
    x = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.int64)

    def fn(value):
        return torch.prod(value, dim=1)

    result = compile_callable(fn, (x,))
    actual = result.plan(_torch_to_mlx(x))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dim", "shape"),
    [(1, (2, 3, 4)), (-1, (2, 3, 4)), (0, (3, 5))],
)
def test_log_softmax_matches_pytorch(dim, shape) -> None:
    torch.manual_seed(54)
    x = torch.randn(shape)
    result = compile_callable(lambda value: torch.log_softmax(value, dim), (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["log_softmax"]
    assert result.plan.kernels == ()

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.log_softmax(x, dim).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("dim", lambda value: value.argmax(-1)),
        ("keepdim", lambda value: value.argmax(1, keepdim=True)),
        ("whole tensor", lambda value: value.argmax()),
        ("argmin", lambda value: value.argmin(0)),
        ("argmin keepdim", lambda value: value.argmin(-2, keepdim=True)),
    ],
)
def test_arg_reductions_match_pytorch(name, fn) -> None:
    torch.manual_seed(56)
    x = torch.randn(3, 4, 5)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "argmin" if "argmin" in name else "argmax"
    ]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.int64
    assert actual.shape == tuple(expected.shape)
    assert mx.array_equal(actual, expected)


def test_argmax_on_integers_and_rank_zero_output() -> None:
    x = torch.tensor([[3, 9, 1], [7, 2, 5]])
    result = compile_callable(lambda value: value.argmax(), (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert actual.shape == ()
    assert int(actual) == int(x.argmax())


@pytest.mark.parametrize(
    ("dim", "descending"),
    [(1, False), (-1, False), (0, True), (1, True)],
)
def test_sort_values_and_indices_match_pytorch(dim, descending) -> None:
    torch.manual_seed(57)
    x = torch.randn(4, 6)

    def fn(value):
        sorted_result = torch.sort(value, dim=dim, descending=descending)
        return sorted_result.values, sorted_result.indices

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["sort", "argsort"]
    values, indices = result.plan(mx.array(x.numpy()))
    expected_values, expected_indices = fn(x)
    mx.eval(values, indices)
    assert values.dtype == mx.float32
    assert indices.dtype == mx.int64
    assert mx.array_equal(values, mx.array(expected_values.numpy()))
    assert mx.array_equal(indices, mx.array(expected_indices.numpy()))


@pytest.mark.parametrize(
    ("fn", "op"),
    [
        pytest.param(
            lambda value: torch.sort(value, dim=1, stable=True).values,
            "sort",
            id="stable_sort",
        ),
        pytest.param(
            lambda value: torch.sort(value, descending=True, stable=True).values,
            "sort",
            id="stable_sort_descending",
        ),
        pytest.param(
            lambda value: torch.argsort(value, dim=-1, stable=True),
            "argsort",
            id="stable_argsort",
        ),
    ],
)
def test_a_stable_sort_spelling_lowers_like_the_plain_one(fn, op) -> None:
    # ``stable=True`` traces to ``aten.sort.stable``, a target of its own whose
    # arguments Torch records entirely as keywords. Ties are the one place the
    # two spellings could disagree, so the input below has them.
    x = torch.tensor([[3.0, 1.0, 2.0, 1.0], [0.5, -1.0, 4.0, 2.0]])
    result = compile_callable(fn, (x,))
    assert [operation.op for operation in result.lowered_graph.ops] == [op]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_sort_emits_only_the_selection_the_graph_reads() -> None:
    torch.manual_seed(58)
    x = torch.randn(3, 5)

    values_only = compile_callable(lambda value: torch.sort(value, dim=-1).values, (x,))
    assert [op.op for op in values_only.lowered_graph.ops] == ["sort"]

    indices_only = compile_callable(
        lambda value: torch.sort(value, dim=-1).indices, (x,)
    )
    assert [op.op for op in indices_only.lowered_graph.ops] == ["argsort"]
    actual = indices_only.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert mx.array_equal(actual, mx.array(torch.sort(x, dim=-1).indices.numpy()))


def test_sort_orders_nan_like_pytorch() -> None:
    x = torch.tensor([[1.0, float("nan"), 3.0, 2.0]])
    for descending in (False, True):
        result = compile_callable(
            lambda value: torch.sort(value, dim=-1, descending=descending).values,
            (x,),
        )
        actual = result.plan(mx.array(x.numpy()))[0]
        expected = mx.array(torch.sort(x, dim=-1, descending=descending).values.numpy())
        mx.eval(actual, expected)
        same = mx.equal(actual, expected) | (mx.isnan(actual) & mx.isnan(expected))
        assert bool(mx.all(same))


@pytest.mark.parametrize(
    ("k", "dim", "largest"),
    [(2, -1, True), (3, 0, False), (5, 1, True), (1, 1, True), (6, 1, False)],
)
def test_topk_values_and_indices_match_pytorch(k, dim, largest) -> None:
    torch.manual_seed(59)
    x = torch.randn(4, 6) if dim in {-1, 1} else torch.randn(5, 4)

    def fn(value):
        selected = torch.topk(value, k, dim=dim, largest=largest)
        return selected.values, selected.indices

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "topk",
        "topk_indices",
    ]
    values, indices = result.plan(mx.array(x.numpy()))
    expected_values, expected_indices = fn(x)
    mx.eval(values, indices)
    assert values.shape == tuple(expected_values.shape)
    assert indices.dtype == mx.int64
    assert mx.array_equal(values, mx.array(expected_values.numpy()))
    assert mx.array_equal(indices, mx.array(expected_indices.numpy()))


def test_topk_emits_each_selection_independently() -> None:
    torch.manual_seed(60)
    x = torch.randn(2, 7)

    indices_only = compile_callable(
        lambda value: torch.topk(value, 3, dim=-1).indices, (x,)
    )
    assert [op.op for op in indices_only.lowered_graph.ops] == ["topk_indices"]
    actual = indices_only.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert mx.array_equal(actual, mx.array(torch.topk(x, 3, dim=-1).indices.numpy()))


def test_topk_rejects_k_beyond_the_dimension() -> None:
    input_type = TensorType((2, 3), DType.F32)
    output_type = TensorType((2, 4), DType.F32)
    graph = Graph(
        ("x",),
        {"x": Value("x", input_type), "out": Value("out", output_type)},
        (
            Operation(
                "topk",
                ("x",),
                ("out",),
                {"dim": 1, "k": 4, "largest": True},
            ),
        ),
        ("out",),
    )
    with pytest.raises(ValueError, match="exceeds dimension"):
        compile_graph(graph)


@pytest.mark.parametrize(
    ("reduction", "dim", "keepdim"),
    [
        ("max", 1, False),
        ("max", -1, False),
        ("max", 0, True),
        ("max", 2, True),
        ("min", 1, False),
        ("min", -2, False),
        ("min", 0, True),
        ("min", 2, True),
    ],
)
def test_extremum_with_dim_matches_pytorch(reduction, dim, keepdim) -> None:
    torch.manual_seed(62)
    x = torch.randn(2, 3, 4)
    reference = getattr(torch, reduction)(x, dim=dim, keepdim=keepdim)

    values = compile_callable(
        lambda value: getattr(torch, reduction)(value, dim=dim, keepdim=keepdim).values,
        (x,),
    )
    indices = compile_callable(
        lambda value: (
            getattr(torch, reduction)(value, dim=dim, keepdim=keepdim).indices
        ),
        (x,),
    )
    assert [op.op for op in values.lowered_graph.ops] == [f"a{reduction}"]
    assert [op.op for op in indices.lowered_graph.ops] == [f"arg{reduction}"]

    actual_values = values.plan(mx.array(x.numpy()))[0]
    actual_indices = indices.plan(mx.array(x.numpy()))[0]
    mx.eval(actual_values, actual_indices)
    assert actual_values.shape == tuple(reference.values.shape)
    assert actual_indices.dtype == mx.int64
    assert mx.array_equal(actual_values, mx.array(reference.values.numpy()))
    assert mx.array_equal(actual_indices, mx.array(reference.indices.numpy()))


def test_extremum_with_dim_returns_both_selections() -> None:
    torch.manual_seed(63)
    x = torch.randn(4, 5)

    def fn(value):
        top = torch.max(value, dim=-1)
        bottom = torch.min(value, dim=-1)
        return top.values, top.indices, bottom.values, bottom.indices

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "amax",
        "argmax",
        "amin",
        "argmin",
    ]
    actual = result.plan(mx.array(x.numpy()))
    expected = fn(x)
    mx.eval(*actual)
    for produced, reference in zip(actual, expected, strict=True):
        assert mx.array_equal(produced, mx.array(reference.numpy()))


def test_extremum_with_dim_breaks_ties_and_nan_like_pytorch() -> None:
    x = torch.tensor(
        [
            [1.0, float("nan"), 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0],
            [float("-inf"), 0.0, -0.0, 0.0],
        ]
    )

    def fn(value):
        top = torch.max(value, dim=1)
        bottom = torch.min(value, dim=1)
        return top.values, top.indices, bottom.values, bottom.indices

    actual = compile_callable(fn, (x,)).plan(mx.array(x.numpy()))
    expected = fn(x)
    mx.eval(*actual)
    for produced, reference in zip(actual, expected, strict=True):
        reference = mx.array(reference.numpy())
        assert produced.shape == reference.shape
        if reference.dtype == mx.int64:
            assert produced.dtype == mx.int64
            assert mx.array_equal(produced, reference)
        else:
            same = mx.equal(produced, reference) | (
                mx.isnan(produced) & mx.isnan(reference)
            )
            assert bool(mx.all(same))


def test_extremum_without_dim_reduces_every_axis() -> None:
    torch.manual_seed(64)
    x = torch.randn(3, 4, 2)
    for reduction in ("max", "min"):
        result = compile_callable(lambda value: getattr(torch, reduction)(value), (x,))
        assert [op.op for op in result.lowered_graph.ops] == [f"a{reduction}"]
        actual = result.plan(mx.array(x.numpy()))[0]
        mx.eval(actual)
        assert actual.shape == ()
        assert float(actual) == float(getattr(torch, reduction)(x))


def test_extremum_with_dim_survives_an_artifact_round_trip(tmp_path) -> None:
    torch.manual_seed(65)
    x = torch.randn(4, 6)

    def fn(value):
        top = torch.max(value, dim=-1)
        return top.indices.to(value.dtype) + torch.min(value, dim=1).values

    result = compile_callable(fn, (x,))
    path = tmp_path / "extremum.aoto"
    save_compiled(result, path)
    loaded = load_compiled_module(path)
    actual = loaded(mx.array(x.numpy()))[0]
    expected = fn(x)
    mx.eval(actual)
    assert mx.array_equal(actual, mx.array(expected.numpy()))


def test_sorting_survives_an_artifact_round_trip(tmp_path) -> None:
    torch.manual_seed(61)
    x = torch.randn(3, 6)

    def fn(value):
        values, _ = torch.sort(value, dim=-1, descending=True)
        return values.argmax(-1), torch.topk(value, 2, dim=-1).values

    result = compile_callable(fn, (x,))
    path = tmp_path / "sorting.aoto"
    save_compiled(result, path)
    loaded = load_compiled_module(path)
    actual = loaded(mx.array(x.numpy()))
    expected = fn(x)
    mx.eval(*actual)
    for produced, reference in zip(actual, expected, strict=True):
        assert mx.array_equal(produced, mx.array(reference.numpy()))


@pytest.mark.parametrize(
    ("op_name", "fn"),
    [
        ("var", lambda value: torch.var(value)),
        ("var", lambda value: torch.var(value, 1)),
        ("var", lambda value: torch.var(value, -1, correction=0)),
        ("var", lambda value: torch.var(value, -1, keepdim=True)),
        ("var", lambda value: torch.var(value, [0, 2])),
        ("std", lambda value: torch.std(value, -1)),
        ("std", lambda value: torch.std(value, -1, correction=0)),
        ("std", lambda value: torch.std(value, [1, 2], keepdim=True)),
    ],
)
def test_variance_reductions_match_pytorch(op_name, fn) -> None:
    torch.manual_seed(62)
    x = torch.randn(3, 4, 5)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_variance_uses_population_correction_when_asked() -> None:
    torch.manual_seed(63)
    x = torch.randn(4, 6)
    for correction in (0, 1, 2):
        fn = lambda value: torch.var(value, -1, correction=correction)
        result = compile_callable(fn, (x,))
        assert result.lowered_graph.ops[0].attrs["correction"] == correction
        actual = result.plan(mx.array(x.numpy()))[0]
        expected = mx.array(fn(x).numpy())
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "fn",
    [
        lambda value: torch.var(value, -1),
        lambda value: torch.std(value, -1),
    ],
)
def test_variance_supports_low_precision(fn) -> None:
    torch.manual_seed(64)
    x = torch.randn(2, 6, dtype=torch.float16)
    result = compile_callable(fn, (x,))
    actual = result.plan(mx.array(x.float().numpy(), dtype=mx.float16))[0]
    expected = mx.array(fn(x.float()).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.float16
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_variance_rejects_a_non_finite_correction() -> None:
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="must be finite"):
        compile_callable(
            lambda value: torch.var(value, -1, correction=float("inf")), (x,)
        )


@pytest.mark.parametrize(
    ("op_name", "correction"),
    [
        ("var", 0.5),
        ("var", 1.5),
        ("var", -0.5),
        ("std", 0.5),
        ("std", 1.5),
        ("std", -0.5),
    ],
)
def test_variance_supports_a_fractional_correction(op_name, correction) -> None:
    # ATen applies a fractional correction as biased variance * n / (n - c);
    # MLX's ddof is integral, so the backend scales the biased pass instead.
    def fn(value):
        return getattr(torch, op_name)(value, -1, correction=correction)

    torch.manual_seed(65)
    x = torch.randn(3, 5, 4)
    result = compile_callable(fn, (x,))
    assert result.lowered_graph.ops[0].attrs["correction"] == correction
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_fractional_correction_uses_the_reduced_element_count() -> None:
    # The scale reads the number of elements actually reduced, so reducing two
    # axes at once gives 20 / 19.5 rather than the trailing-axis 4 / 3.5.
    def fn(value):
        return torch.var(value, dim=(1, 2), correction=0.5, keepdim=True)

    torch.manual_seed(66)
    x = torch.randn(3, 4, 5)
    result = compile_callable(fn, (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (3, 1, 1)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_fractional_correction_equal_to_the_count_yields_inf() -> None:
    x = torch.randn(2, 4)

    def fn(value):
        return torch.var(value, -1, correction=4.0)

    result = compile_callable(fn, (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert bool(mx.all(mx.isinf(actual)))


@pytest.mark.parametrize(
    "fn",
    [
        lambda value: torch.logsumexp(value, -1),
        lambda value: torch.logsumexp(value, 0, keepdim=True),
        lambda value: torch.logsumexp(value, dim=(1, 2)),
        lambda value: torch.logsumexp(value, dim=[0, 2], keepdim=True),
    ],
)
def test_logsumexp_matches_pytorch(fn) -> None:
    torch.manual_seed(67)
    x = torch.randn(3, 4, 5)
    result = compile_callable(fn, (x,))
    # The composite is a max shift plus a log-sum, so no MLX logsumexp needed.
    assert "amax" in [op.op for op in result.lowered_graph.ops]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_logsumexp_keeps_saturated_limits() -> None:
    # ATen replaces an infinite maximum with zero before subtracting, so a
    # +inf row stays +inf and an all -inf row stays -inf instead of going nan.
    x = torch.tensor(
        [
            [float("inf"), 0.0],
            [float("-inf"), float("-inf")],
            [1000.0, 1001.0],
        ]
    )
    result = compile_callable(lambda value: torch.logsumexp(value, -1), (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = torch.logsumexp(x, -1)
    mx.eval(actual)
    assert actual.tolist()[:2] == [math.inf, -math.inf]
    assert actual.tolist()[2] == pytest.approx(expected[2].item(), rel=1e-6)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cumprod_matches_pytorch(dim) -> None:
    def fn(value):
        return torch.cumprod(value, dim)

    torch.manual_seed(68)
    x = torch.randn(3, 4)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["cumprod"]
    assert result.lowered_graph.ops[0].attrs == {"dim": dim}
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cumprod_preserves_an_integer_dtype() -> None:
    def fn(value):
        return torch.cumprod(value, 1)

    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    result = compile_callable(fn, (x,))
    actual = result.plan(_torch_to_mlx(x))[0]
    expected = _torch_to_mlx(fn(x))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("shape", "offset", "dim1", "dim2"),
    [
        ((4, 4), 0, 0, 1),
        ((4, 4), 2, 0, 1),
        ((4, 4), -2, 0, 1),
        ((3, 4), 1, 0, 1),
        ((3, 4), -1, 0, 1),
        ((3, 4, 5), 0, -2, -1),
        ((3, 4, 5), -2, 1, 2),
        ((3, 4, 5), 1, 0, 2),
        ((2, 3, 4, 5), 0, 1, 3),
    ],
)
def test_diagonal_matches_pytorch(shape, offset, dim1, dim2) -> None:
    def fn(value):
        return torch.diagonal(value, offset, dim1, dim2)

    torch.manual_seed(69)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    # The diagonal is a strided slice of a flattened trailing pair, so the
    # plan copies nothing and launches no kernel.
    assert result.plan.kernels == ()
    assert {op.op for op in result.lowered_graph.ops} <= {
        "permute",
        "reshape",
        "slice",
    }
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert mx.array_equal(actual, expected)


def test_diagonal_reads_the_expected_elements() -> None:
    source = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = compile_callable(lambda value: torch.diagonal(value), (torch.zeros(2, 3),))
    mx.eval(result.plan(source))
    assert result.plan(source)[0].tolist() == [1.0, 5.0]
    shifted = compile_callable(
        lambda value: torch.diagonal(value, 1), (torch.zeros(2, 3),)
    )
    assert shifted.plan(source)[0].tolist() == [2.0, 6.0]


@pytest.mark.parametrize(
    ("shape", "dim", "size", "step"),
    [
        ((7,), 0, 2, 1),
        ((2, 3, 7), 0, 2, 2),
        ((2, 3, 7), -1, 3, 1),
        ((1, 2, 5, 5), 2, 3, 1),
        ((2, 3, 4, 5), 1, 2, 3),
        ((2, 3, 6, 6), 3, 2, 2),
    ],
)
def test_unfold_matches_pytorch(shape, dim, size, step) -> None:
    def fn(value):
        return value.unfold(dim, size, step)

    torch.manual_seed(43)
    x = torch.randn(shape)
    result = compile_callable(fn, (x,))
    # A window is one strided read, so the plan copies nothing until the view
    # is made contiguous and launches no kernel.
    assert result.plan.kernels == ()
    assert [op.op for op in result.lowered_graph.ops] == ["unfold"]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert mx.array_equal(actual, expected)


def test_unfold_puts_the_window_last_and_the_count_in_place() -> None:
    # ATen keeps the axes after the unfolded one where they were and appends the
    # window, so (2, 5) unfolded by two with a step of one is (2, 4, 2).
    graph = compile_callable(
        lambda value: value.unfold(1, 2, 1), (torch.zeros(2, 5),)
    ).input_graph
    (unfold,) = graph.ops
    assert unfold.op == "unfold"
    assert unfold.attrs == {"dim": 1, "size": 2, "step": 1}
    assert graph.value(unfold.outputs[0]).type.shape == (2, 4, 2)


def test_unfold_reads_the_expected_elements() -> None:
    source = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stepped = compile_callable(
        lambda value: value.unfold(1, 2, 2), (torch.zeros(2, 3),)
    )
    assert stepped.plan(source)[0].tolist() == [[[1.0, 2.0]], [[4.0, 5.0]]]


def test_unfold_rejects_a_window_wider_than_the_axis() -> None:
    # Torch rejects it while tracing, so the error surfaces before lowering.
    with pytest.raises(RuntimeError, match="maximum size for tensor"):
        compile_callable(lambda value: value.unfold(0, 4, 1), (torch.zeros(3),))


def test_diagonal_rejects_an_offset_with_no_elements() -> None:
    with pytest.raises(NotImplementedError, match="no elements"):
        compile_callable(lambda value: torch.diagonal(value, 4), (torch.randn(3, 3),))


def test_masked_fill_with_a_tensor_value_lowers_to_where() -> None:
    def fn(value, mask, fill):
        return value.masked_fill(mask, fill)

    torch.manual_seed(70)
    x = torch.randn(3, 4)
    mask = torch.rand(3, 4) > 0.5
    fill = torch.tensor(-2.5)
    result = compile_callable(fn, (x, mask, fill))
    assert [op.op for op in result.lowered_graph.ops] == ["where"]
    actual = result.plan(
        mx.array(x.numpy()), mx.array(mask.numpy()), mx.array(fill.numpy())
    )[0]
    expected = mx.array(fn(x, mask, fill).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("shape", "repeats"),
    [
        ((3,), (4,)),
        ((3, 4), (2, 3)),
        ((3, 4), (2, 1, 3)),
        ((2, 3, 4), (2, 1, 2)),
        ((2, 3, 4), (1, 1, 1)),
        ((1, 5), (3, 2)),
    ],
)
def test_repeat_and_tile_lower_to_layout_operations(shape, repeats) -> None:
    torch.manual_seed(65)
    x = torch.randn(shape)
    for fn in (
        lambda value: value.repeat(*repeats),
        lambda value: value.tile(*repeats),
    ):
        result = compile_callable(fn, (x,))
        assert all(op.op in {"reshape", "expand"} for op in result.lowered_graph.ops)
        actual = result.plan(mx.array(x.numpy()))[0]
        expected = mx.array(fn(x).numpy())
        mx.eval(actual, expected)
        assert actual.shape == tuple(expected.shape)
        assert mx.array_equal(actual, expected)


def test_repeat_with_only_ones_disappears() -> None:
    x = torch.randn(3, 4)
    result = compile_callable(lambda value: value.repeat(1, 1), (x,))
    assert result.lowered_graph.ops == ()


def test_repeat_rejects_empty_dimensions() -> None:
    x = torch.randn(3, 4)
    with pytest.raises(NotImplementedError, match="empty dimension"):
        compile_callable(lambda value: value.repeat(0, 2), (x,))


@pytest.mark.parametrize(
    ("ord_value", "dims", "keepdim"),
    [
        (2.0, (-1,), True),
        (2.0, None, False),
        (1.0, (0,), False),
        (float("inf"), None, False),
        (float("-inf"), (-1,), False),
        (3.0, (1, 2), True),
        (0.0, (-1,), False),
    ],
)
def test_vector_norm_matches_pytorch(ord_value, dims, keepdim) -> None:
    torch.manual_seed(66)
    x = torch.randn(2, 3, 4)

    def fn(value):
        return torch.linalg.vector_norm(value, ord_value, dims, keepdim)

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["vector_norm"]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_vector_norm_folds_into_functional_normalize() -> None:
    torch.manual_seed(67)
    x = torch.randn(4, 5)
    result = compile_callable(
        lambda value: torch.nn.functional.normalize(value, dim=-1, eps=1e-9),
        (x,),
    )
    assert [op.op for op in result.lowered_graph.ops] == [
        "vector_norm",
        "clamp_min",
        "expand",
        "div",
    ]
    assert result.lowered_graph.ops[0].attrs["ord"] == 2.0
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.nn.functional.normalize(x, dim=-1, eps=1e-9).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_vector_norm_supports_low_precision_artifact_round_trip(tmp_path) -> None:
    torch.manual_seed(68)
    x = torch.randn(2, 6, dtype=torch.float16)

    def fn(value):
        return torch.linalg.vector_norm(value, 2, (-1,), True) * value

    result = compile_callable(fn, (x,))
    path = tmp_path / "vector_norm.aoto"
    save_compiled(result, path)
    loaded = load_compiled_module(path)
    actual = loaded(mx.array(x.float().numpy(), dtype=mx.float16))[0]
    expected = mx.array(fn(x.float()).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.float16
    assert mx.allclose(actual.astype(mx.float32), expected, atol=2e-3, rtol=2e-3)


def test_log_softmax_supports_low_precision() -> None:
    torch.manual_seed(55)
    x = torch.randn(2, 6, dtype=torch.float16)
    result = compile_callable(lambda value: torch.log_softmax(value, -1), (x,))
    actual = result.plan(mx.array(x.float().numpy(), dtype=mx.float16))[0]
    expected = mx.array(torch.log_softmax(x.float(), -1).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.float16
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_log_softmax_rejects_integer_inputs() -> None:
    with pytest.raises(NotImplementedError, match="needs a floating-point input"):
        compile_callable(
            lambda value: torch.log_softmax(value, -1),
            (torch.tensor([[1, 2]]),),
        )


def test_reduction_rejects_repeated_axes() -> None:
    values = {
        "x": Value("x", TensorType((2, 3), DType.F32)),
        "out": Value("out", TensorType((3,), DType.F32)),
    }
    graph = Graph(
        ("x",),
        values,
        (
            Operation(
                "amax",
                ("x",),
                ("out",),
                {"dims": (1, -1), "keepdim": False},
            ),
        ),
        ("out",),
    )
    with pytest.raises(ValueError, match="repeats an axis"):
        compile_graph(graph)


def test_embedding_composes_with_matmul_and_pointwise_fusion() -> None:
    def fn(indices, embedding, projection):
        hidden = torch.nn.functional.embedding(indices, embedding)
        return torch.nn.functional.silu(hidden @ projection)

    torch.manual_seed(45)
    inputs = (
        torch.tensor([1, 3, 0, 2], dtype=torch.int64),
        torch.randn(4, 8, dtype=torch.float16),
        torch.randn(8, 6, dtype=torch.float16),
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == [
        "embedding",
        "matmul",
        "silu",
    ]

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert actual.shape == (4, 6)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_embedding_requires_integer_indices() -> None:
    weight_type = TensorType((4, 8), DType.F32)
    index_type = TensorType((2,), DType.F32)
    values = {
        "weight": Value("weight", weight_type),
        "indices": Value("indices", index_type),
        "out": Value("out", TensorType((2, 8), DType.F32)),
    }
    graph = Graph(
        ("weight", "indices"),
        values,
        (Operation("embedding", ("weight", "indices"), ("out",)),),
        ("out",),
    )
    with pytest.raises(ValueError, match="indices must be int32 or int64"):
        compile_graph(graph)


def test_flip_lowers_to_mlx_layout_without_a_kernel() -> None:
    def fn(x):
        return torch.flip(x, dims=[1, -1])

    torch.manual_seed(46)
    x = torch.randn(2, 3, 4)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["flip"]
    assert result.plan.kernels == ()

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    "roll",
    [
        lambda x: torch.roll(x, shifts=1, dims=-1),
        lambda x: torch.roll(x, shifts=(1, -2), dims=(0, 2)),
        lambda x: torch.roll(x, shifts=3),
    ],
)
def test_roll_lowers_to_mlx_layout_without_a_kernel(roll) -> None:
    torch.manual_seed(47)
    x = torch.randn(2, 3, 4)
    result = compile_callable(roll, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["roll"]
    assert result.plan.kernels == ()

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(roll(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(("op_name", "diagonal"), [("triu", 1), ("tril", -1)])
def test_triangular_ops_use_mlx_without_a_kernel(op_name, diagonal) -> None:
    operation = getattr(torch, op_name)

    def fn(value):
        return operation(value, diagonal=diagonal)

    torch.manual_seed(48)
    value = torch.randn(2, 3, 4, 5)
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    assert result.lowered_graph.ops[0].attrs["diagonal"] == diagonal
    assert result.plan.kernels == ()

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_inverse_flip_and_roll_layouts_are_eliminated() -> None:
    def fn(x):
        flipped = torch.flip(torch.flip(x, dims=[1, -1]), dims=[-1, 1])
        rolled = torch.roll(
            torch.roll(x, shifts=(1, -2), dims=(0, 2)),
            shifts=(-1, 2),
            dims=(0, 2),
        )
        flat_rolled = torch.roll(torch.roll(x, shifts=3), shifts=-3)
        return flipped, rolled, flat_rolled

    torch.manual_seed(48)
    x = torch.randn(2, 3, 4)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.input_graph.ops] == [
        "flip",
        "flip",
        "roll",
        "roll",
        "roll",
        "roll",
    ]
    assert result.lowered_graph.ops == ()
    assert result.lowered_graph.outputs == ("arg0", "arg0", "arg0")

    actual = result.plan(mx.array(x.numpy()))
    mx.eval(*actual)
    for value in actual:
        assert mx.array_equal(value, mx.array(x.numpy()))


def test_singleton_flip_and_zero_roll_are_identities() -> None:
    def fn(x):
        return torch.flip(x, dims=[1]), torch.roll(x, shifts=2, dims=1)

    torch.manual_seed(49)
    x = torch.randn(2, 1, 4)
    result = compile_callable(fn, (x,))
    assert result.lowered_graph.ops == ()
    assert result.lowered_graph.outputs == ("arg0", "arg0")


def test_out_of_range_slice_is_not_removed_as_identity() -> None:
    tensor_type = TensorType((4,), DType.F32)
    values = {
        "x": Value("x", tensor_type),
        "out": Value("out", tensor_type),
    }
    graph = Graph(
        ("x",),
        values,
        (
            Operation(
                "slice",
                ("x",),
                ("out",),
                {"dim": 1, "start": None, "end": None, "step": 1},
            ),
        ),
        ("out",),
    )
    with pytest.raises(ValueError, match="slice.dim 1 exceeds rank 1"):
        compile_graph(graph)


def test_folded_batched_matmul_is_restored_to_rank_four() -> None:
    def fn(query, key):
        return torch.matmul(query, key.transpose(-2, -1))

    query = torch.randn(2, 4, 6, 8)
    key = torch.randn(2, 4, 3, 8)
    result = compile_callable(fn, (query, key))
    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]
    assert result.lowered_graph.ops[0].attrs == {"transpose_b": True}

    inputs = (mx.array(query.numpy()), mx.array(key.numpy()))
    actual = result.plan(*inputs)[0]
    expected = mx.matmul(inputs[0], mx.swapaxes(inputs[1], -1, -2))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_matmul_input_transpose_folds_into_left_operand() -> None:
    left = torch.randn(8, 3)
    right = torch.randn(8, 5)
    result = compile_callable(lambda lhs, rhs: lhs.T @ rhs, (left, right))

    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]
    assert result.lowered_graph.ops[0].attrs == {"transpose_a": True}

    inputs = (mx.array(left.numpy()), mx.array(right.numpy()))
    actual = result.plan(*inputs)[0]
    expected = mx.matmul(mx.swapaxes(inputs[0], -1, -2), inputs[1])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected)


def test_matmul_input_transpose_is_kept_when_shared() -> None:
    left = torch.randn(3, 8)
    right = torch.randn(3, 5)

    def fn(lhs, rhs):
        transposed = lhs.T
        return transposed @ rhs, transposed

    result = compile_callable(fn, (left, right))
    assert [op.op for op in result.lowered_graph.ops] == ["permute", "matmul"]
    assert "transpose_a" not in result.lowered_graph.ops[1].attrs


def test_direct_scaled_dot_product_attention_with_mask() -> None:
    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )

    torch.manual_seed(47)
    inputs = (
        torch.randn(2, 3, 4, 8),
        torch.randn(2, 3, 5, 8),
        torch.randn(2, 3, 5, 8),
        torch.randn(2, 1, 4, 5) * 0.2,
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.input_graph.ops] == ["attention"]
    attention = result.input_graph.ops[0]
    assert attention.attrs == {
        "scale": pytest.approx(8**-0.5),
        "has_mask": True,
        "causal": False,
        "safe_softmax": True,
    }
    assert result.plan.kernels == ()

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_direct_scaled_dot_product_attention_causal() -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

    torch.manual_seed(53)
    inputs = tuple(torch.randn(2, 3, 6, 8) for _ in range(3))
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.input_graph.ops] == ["attention"]
    attention = result.input_graph.ops[0]
    assert attention.attrs["scale"] == pytest.approx(8**-0.5)
    assert attention.attrs["has_mask"] is False
    assert attention.attrs["causal"] is True

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(("query_length", "key_length"), [(3, 5), (5, 3)])
def test_direct_scaled_dot_product_attention_causal_with_uneven_lengths(
    query_length: int, key_length: int
) -> None:
    """Torch aligns a causal mask at the top left, which is not MLX's mask.

    MLX's own causal mask is aligned at the bottom right, so a query shorter or
    longer than its keys has to be masked explicitly to agree with Torch.
    """

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

    torch.manual_seed(31)
    inputs = (
        torch.randn(2, 2, query_length, 8),
        torch.randn(2, 2, key_length, 8),
        torch.randn(2, 2, key_length, 8),
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.input_graph.ops] == ["attention"]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_direct_scaled_dot_product_attention_explicit_scale() -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=0.25
        )

    torch.manual_seed(59)
    inputs = tuple(torch.randn(2, 3, 4, 8) for _ in range(3))
    result = compile_callable(fn, inputs)
    assert result.input_graph.ops[0].attrs["scale"] == 0.25

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("codegen_attention", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_direct_scaled_dot_product_attention_distinct_value_head_dim(
    codegen_attention, dtype
) -> None:
    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            scale=0.25,
        )

    torch.manual_seed(62)
    inputs = (
        torch.randn(2, 2, 3, 8, dtype=dtype),
        torch.randn(2, 2, 4, 8, dtype=dtype),
        torch.randn(2, 2, 4, 6, dtype=dtype),
        torch.randn(2, 1, 3, 4, dtype=dtype) * 0.2,
    )
    result = compile_callable(fn, inputs, codegen_attention=codegen_attention)

    input_ops = [op.op for op in result.input_graph.ops]
    assert input_ops.count("mul") == 2
    assert "safe_softmax" in input_ops
    assert [op.op for op in result.lowered_graph.ops] == ["attention"]
    attention = result.lowered_graph.ops[0]
    assert attention.attrs == {
        "scale": 0.25,
        "has_mask": True,
        "causal": False,
        "safe_softmax": True,
    }
    assert [kernel.kind for kernel in result.plan.kernels] == (
        ["attention"] if codegen_attention else []
    )

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
    assert actual.shape == (2, 2, 3, 6)
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("codegen_attention", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_attention_zeroes_fully_masked_rows(codegen_attention, dtype) -> None:
    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=0.25
        )

    torch.manual_seed(65)
    inputs = (
        torch.randn(2, 2, 3, 8, dtype=dtype),
        torch.randn(2, 2, 4, 8, dtype=dtype),
        torch.randn(2, 2, 4, 6, dtype=dtype),
        torch.full((2, 1, 3, 4), -torch.inf, dtype=dtype),
    )
    result = compile_callable(fn, inputs, codegen_attention=codegen_attention)
    assert "safe_softmax" in [op.op for op in result.input_graph.ops]
    assert result.lowered_graph.ops[0].attrs["safe_softmax"] is True

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)

    assert mx.all(mx.isfinite(actual)).item()
    assert mx.array_equal(actual, mx.zeros_like(actual))
    assert mx.allclose(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("codegen_attention", [False, True])
def test_attention_preserves_explicit_low_precision_cast(codegen_attention) -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query.bfloat16(), key.bfloat16(), value.bfloat16()
        )

    torch.manual_seed(63)
    inputs = (
        torch.randn(2, 3, 4, 8, dtype=torch.float16),
        torch.randn(2, 3, 5, 8, dtype=torch.float16),
        torch.randn(2, 3, 5, 6, dtype=torch.float16),
    )
    result = compile_callable(fn, inputs, codegen_attention=codegen_attention)

    lowered_ops = result.lowered_graph.ops
    assert [op.op for op in lowered_ops] == ["cast", "cast", "cast", "attention"]
    attention = lowered_ops[-1]
    assert all(
        result.lowered_graph.value(value_id).type.dtype == DType.BF16
        for value_id in attention.inputs + attention.outputs
    )

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("codegen_attention", [False, True])
def test_attention_preserves_explicit_float32_compute(codegen_attention) -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float()
        )

    torch.manual_seed(64)
    inputs = (
        torch.randn(2, 3, 4, 8, dtype=torch.float16),
        torch.randn(2, 3, 5, 8, dtype=torch.float16),
        torch.randn(2, 3, 5, 6, dtype=torch.float16),
    )
    result = compile_callable(fn, inputs, codegen_attention=codegen_attention)

    lowered_ops = result.lowered_graph.ops
    assert [op.op for op in lowered_ops] == ["cast", "cast", "cast", "attention"]
    attention = lowered_ops[-1]
    assert all(
        result.lowered_graph.value(value_id).type.dtype == DType.F32
        for value_id in attention.inputs + attention.outputs
    )

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_direct_scaled_dot_product_attention_grouped_query() -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, enable_gqa=True
        )

    torch.manual_seed(61)
    inputs = (
        torch.randn(2, 8, 5, 16),
        torch.randn(2, 2, 5, 16),
        torch.randn(2, 2, 5, 16),
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["attention"]
    assert result.plan.kernels == ()

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("key_value_heads", [8, 2, 1])
def test_native_attention_rank4_mha_and_gqa(key_value_heads) -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            enable_gqa=key_value_heads != 8,
        )

    torch.manual_seed(63)
    inputs = (
        torch.randn(2, 8, 5, 16),
        torch.randn(2, key_value_heads, 7, 16),
        torch.randn(2, key_value_heads, 7, 16),
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]
    source = result.plan.kernels[0].source
    assert "metal_attention_flash_8x32" in result.plan.kernels[0].name
    assert "threadgroup T tile_q[128]" in source
    assert "threadgroup T tile_k[512]" in source
    assert "simd_max(score)" in source
    assert "simd_sum(probability)" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    ("dtype", "simdgroup_type"),
    [(torch.float16, "half"), (torch.bfloat16, "bfloat")],
)
def test_native_attention_uses_simdgroup_qk_and_pv(dtype, simdgroup_type) -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    torch.manual_seed(64)
    inputs = (
        torch.randn(2, 2, 8, 64, dtype=dtype),
        torch.randn(2, 2, 7, 64, dtype=dtype),
        torch.randn(2, 2, 7, 64, dtype=dtype),
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    source = result.plan.kernels[0].source

    assert "threadgroup float tile_scores[256]" in source
    assert "if (warp < 4u)" in source
    assert "vec<T," not in source
    assert "simdgroup_float8x8 qk_accumulator" in source
    assert f"simdgroup_{simdgroup_type}8x8 qk_frag_q" in source
    assert "simdgroup_load(qk_frag_k" in source
    assert "key_fragment * 8u * 64u" in source
    assert "threadgroup T tile_probabilities_low[256]" in source
    assert "threadgroup float tile_pv[512]" in source
    assert "simdgroup_float8x8 pv_accumulator" in source
    assert f"simdgroup_{simdgroup_type}8x8 pv_frag_probability" in source
    assert f"simdgroup_{simdgroup_type}8x8 pv_frag_value" in source
    assert "simdgroup_multiply_accumulate" in source
    assert "tile_k[index] = global_key" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float16, 5e-3), (torch.bfloat16, 2e-2)],
)
def test_native_attention_uses_tensor_core_when_query_tile_fits(
    dtype, tolerance
) -> None:
    _require_tensor_cores()

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    torch.manual_seed(66)
    inputs = (
        torch.randn(1, 2, 17, 64, dtype=dtype),
        torch.randn(1, 2, 40, 64, dtype=dtype),
        torch.randn(1, 2, 40, 64, dtype=dtype),
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    kernel = result.plan.kernels[0]
    source = kernel.source

    assert "metal_attention_tensorcore_16x32_32" in kernel.name
    assert "threadgroup T tile_q[1024]" in source
    assert "16, 32, 16, false, true, true," in source
    assert "tsc_qk.run(tsc_q, tsc_k, tsc_scores)" in source
    assert "tsc_pv.run(tsc_p, tsc_v, tsc_acc_" in source
    assert "simd_shuffle(block_max_0" in source
    assert "simd_shuffle(block_sum_0" in source
    assert "simdgroup_multiply_accumulate" not in source
    assert "threadgroup float tile_scores" not in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_native_attention_tensor_core_grouped_query_with_mask() -> None:
    _require_tensor_cores()

    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, enable_gqa=True
        )

    torch.manual_seed(69)
    mask = torch.zeros(1, 8, 64, 80, dtype=torch.float16)
    mask[..., 3::4] = float("-inf")
    mask[:, :, 5, :] = float("-inf")
    inputs = (
        torch.randn(1, 8, 64, 64, dtype=torch.float16),
        torch.randn(1, 2, 80, 64, dtype=torch.float16),
        torch.randn(1, 2, 80, 64, dtype=torch.float16),
        mask,
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    kernel = result.plan.kernels[0]
    source = kernel.source

    assert "metal_attention_tensorcore_32x32_64" in kernel.name
    assert "head / 4u" in source
    assert source.count("in_3[") == 16
    assert "if (key_base + 32u > 80u)" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    empty_row = mx.zeros((1, 8, 64), dtype=mx.float16)
    mx.eval(actual, expected, empty_row)
    assert mx.array_equal(actual[:, :, 5, :], empty_row)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_native_attention_drops_the_key_clamp_on_aligned_tiles() -> None:
    """An aligned key length makes the per-element clamp provably a no-op."""

    _require_tensor_cores()

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

    torch.manual_seed(71)
    inputs = tuple(torch.randn(1, 2, 256, 64, dtype=torch.float16) for _ in range(3))
    result = compile_callable(fn, inputs, codegen_attention=True)
    kernel = result.plan.kernels[0]
    source = kernel.source

    assert "metal_attention_tensorcore_32x32_64" in kernel.name
    # The key rows of both operands are addressed directly instead of going
    # through `metal::min`, which the kernel then never has to evaluate.
    assert "metal::min(key_base" not in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_native_attention_matches_pytorch_on_an_unaligned_key_length() -> None:
    """The clamp is kept whenever the tiles can run past the last key."""

    _require_tensor_cores()

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

    torch.manual_seed(72)
    inputs = tuple(torch.randn(1, 2, 1000, 64, dtype=torch.float16) for _ in range(3))
    result = compile_callable(fn, inputs, codegen_attention=True)
    source = result.plan.kernels[0].source

    assert "metal::min(key_base" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_native_attention_drops_the_key_clamp_for_cross_attention() -> None:
    """A causal query shorter than an aligned key still stops on a tile edge."""

    _require_tensor_cores()

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

    torch.manual_seed(73)
    inputs = (
        torch.randn(1, 2, 64, 64, dtype=torch.float16),
        torch.randn(1, 2, 128, 64, dtype=torch.float16),
        torch.randn(1, 2, 128, 64, dtype=torch.float16),
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    source = result.plan.kernels[0].source

    assert "metal::min(key_base" not in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float16, 5e-3), (torch.bfloat16, 2e-2)],
)
def test_native_attention_uses_long_sequence_query_tile(dtype, tolerance) -> None:
    _require_tensor_cores()

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )

    torch.manual_seed(67)
    inputs = tuple(torch.randn(1, 1, 769, 128, dtype=dtype) for _ in range(3))
    result = compile_callable(fn, inputs, codegen_attention=True)
    kernel = result.plan.kernels[0]
    source = kernel.source

    assert "metal_attention_tensorcore_32x32_64" in kernel.name
    assert "threadgroup T tile_q[4096]" in source
    assert "uint query_row_0 = sg * 16u + 0u + fm" in source
    assert "uint key_row_3 = metal::min(key_base + 24u + fm, 768u)" in source
    assert "metal::fast::exp(" in source
    assert "if (key_base + 32u > query_block * 32u + sg * 16u)" in source
    assert "if (key_base + 32u > 769u)" in source
    assert "if (alpha_0 != 1.0f || alpha_1 != 1.0f)" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("shape", "expected_kernel"),
    [
        ((1, 1, 16, 256), "metal_attention_flash_8x32_256"),
        ((1, 1, 768, 160), "metal_attention_flash_16x64_512"),
    ],
)
def test_native_attention_limits_query_tile_by_shared_memory(
    shape, expected_kernel
) -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    inputs = tuple(torch.randn(*shape, dtype=torch.float16) for _ in range(3))
    result = compile_callable(fn, inputs, codegen_attention=True)

    assert expected_kernel in result.plan.kernels[0].name


def test_native_attention_long_sequence_boolean_mask_uses_64_key_tile() -> None:
    dtype = DType.F16
    values = {
        "query": Value("query", TensorType((1, 769, 16), dtype)),
        "key": Value("key", TensorType((1, 64, 16), dtype)),
        "value": Value("value", TensorType((1, 64, 16), dtype)),
        "mask": Value("mask", TensorType((769, 64), DType.BOOL)),
        "output": Value("output", TensorType((1, 769, 16), dtype)),
    }
    graph = Graph(
        ("query", "key", "value", "mask"),
        values,
        (
            Operation(
                "attention",
                ("query", "key", "value", "mask"),
                ("output",),
                {"scale": 16**-0.5, "has_mask": True, "causal": False},
            ),
        ),
        ("output",),
    )
    result = compile_graph(graph, codegen_attention=True)
    kernel = result.plan.kernels[0]
    assert "metal_attention_flash_24x64_768" in kernel.name
    assert "float score0 = -INFINITY" in kernel.source
    assert "float score1 = -INFINITY" in kernel.source

    torch.manual_seed(77)
    mask = torch.rand(769, 64) > 0.25
    mask[1, :] = False
    inputs = (
        torch.randn(1, 769, 16, dtype=torch.float16),
        torch.randn(1, 64, 16, dtype=torch.float16),
        torch.randn(1, 64, 16, dtype=torch.float16),
        mask,
    )
    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(
        torch.nn.functional.scaled_dot_product_attention(
            inputs[0], inputs[1], inputs[2], attn_mask=inputs[3]
        ).numpy()
    )
    empty_row = mx.zeros((1, 16), dtype=mx.float16)
    mx.eval(actual, expected, empty_row)
    assert mx.array_equal(actual[:, 1, :], empty_row)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_native_attention_long_sequence_partial_value_chunk() -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
        )

    torch.manual_seed(68)
    inputs = (
        torch.randn(1, 1, 769, 64, dtype=torch.float16),
        torch.randn(1, 1, 769, 64, dtype=torch.float16),
        torch.randn(1, 1, 769, 72, dtype=torch.float16),
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    kernel = result.plan.kernels[0]

    assert "metal_attention_flash_24x64_768" in kernel.name
    assert "value_chunk_fragments" in kernel.source
    assert "vec<T, 8>" in kernel.source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_native_attention_additive_mask_dtypes(dtype) -> None:
    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            scale=0.17,
        )

    torch.manual_seed(65)
    inputs = (
        torch.randn(2, 2, 3, 20, dtype=dtype),
        torch.randn(2, 2, 4, 20, dtype=dtype),
        torch.randn(2, 2, 4, 20, dtype=dtype),
        torch.randn(2, 1, 3, 4, dtype=dtype) * 0.2,
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_native_attention_all_masked_row_is_zero() -> None:
    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
        )

    torch.manual_seed(67)
    mask = torch.randn(2, 1, 3, 4) * 0.2
    mask[:, :, 0, :] = -torch.inf
    inputs = (
        torch.randn(2, 2, 3, 16),
        torch.randn(2, 2, 4, 16),
        torch.randn(2, 2, 4, 16),
        mask,
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    zero_row = mx.zeros((2, 2, 16))
    mx.eval(actual, expected, zero_row)
    assert mx.array_equal(actual[:, :, 0, :], zero_row)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_native_attention_boolean_mask_and_zeroes_empty_rows() -> None:
    dtype = DType.F32
    values = {
        "query": Value("query", TensorType((2, 2, 3, 8), dtype)),
        "key": Value("key", TensorType((2, 2, 4, 8), dtype)),
        "value": Value("value", TensorType((2, 2, 4, 8), dtype)),
        "mask": Value("mask", TensorType((1, 1, 3, 4), DType.BOOL)),
        "output": Value("output", TensorType((2, 2, 3, 8), dtype)),
    }
    graph = Graph(
        ("query", "key", "value", "mask"),
        values,
        (
            Operation(
                "attention",
                ("query", "key", "value", "mask"),
                ("output",),
                {
                    "scale": 8**-0.5,
                    "has_mask": True,
                    "causal": False,
                    "safe_softmax": True,
                },
            ),
        ),
        ("output",),
    )
    result = compile_graph(graph, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]

    torch.manual_seed(68)
    mask = torch.rand(1, 1, 3, 4) > 0.25
    mask[:, :, 1, :] = False
    inputs = (
        torch.randn(2, 2, 3, 8),
        torch.randn(2, 2, 4, 8),
        torch.randn(2, 2, 4, 8),
        mask,
    )
    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(
        torch.nn.functional.scaled_dot_product_attention(
            inputs[0], inputs[1], inputs[2], attn_mask=inputs[3]
        ).numpy()
    )
    empty_row = mx.zeros((2, 2, 8))
    mx.eval(actual, expected, empty_row)
    assert mx.array_equal(actual[:, :, 1, :], empty_row)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_frontend_boolean_attention_mask_uses_additive_pointwise_path() -> None:
    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )

    torch.manual_seed(70)
    mask = torch.rand(1, 1, 3, 4) > 0.25
    mask[:, :, 2, :] = False
    inputs = (
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 2, 4, 8),
        torch.randn(1, 2, 4, 8),
        mask,
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    input_ops = [op.op for op in result.input_graph.ops]
    assert "scalar_tensor" in input_ops
    assert "where" in input_ops
    assert [op.op for op in result.lowered_graph.ops] == [
        "fused_elementwise",
        "attention",
    ]
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "fused_elementwise",
        "attention",
    ]

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    empty_row = mx.zeros((1, 2, 8))
    mx.eval(actual, expected, empty_row)
    assert mx.array_equal(actual[:, :, 2, :], empty_row)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_native_attention_causal_with_rectangular_sequence() -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )

    torch.manual_seed(69)
    inputs = (
        torch.randn(2, 2, 4, 12),
        torch.randn(2, 2, 6, 12),
        torch.randn(2, 2, 6, 12),
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_native_attention_supports_distinct_value_head_dim() -> None:
    dtype = DType.F32
    values = {
        "query": Value("query", TensorType((2, 2, 3, 8), dtype)),
        "key": Value("key", TensorType((2, 2, 4, 8), dtype)),
        "value": Value("value", TensorType((2, 2, 4, 6), dtype)),
        "output": Value("output", TensorType((2, 2, 3, 6), dtype)),
    }
    graph = Graph(
        ("query", "key", "value"),
        values,
        (
            Operation(
                "attention",
                ("query", "key", "value"),
                ("output",),
                {"scale": 8**-0.5, "has_mask": False, "causal": False},
            ),
        ),
        ("output",),
    )
    result = compile_graph(graph, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]

    torch.manual_seed(71)
    inputs = (
        torch.randn(2, 2, 3, 8),
        torch.randn(2, 2, 4, 8),
        torch.randn(2, 2, 4, 6),
    )
    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(
        torch.nn.functional.scaled_dot_product_attention(*inputs).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_native_attention_squeezes_rank3_output() -> None:
    def fn(query, key, value, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
        )

    torch.manual_seed(73)
    inputs = (
        torch.randn(2, 5, 16),
        torch.randn(2, 7, 16),
        torch.randn(2, 7, 16),
        torch.randn(2, 5, 7) * 0.2,
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    assert "squeeze" in [op.op for op in result.lowered_graph.ops]
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (2, 5, 16)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("mask_shape", [(5, 7), (2, 5, 7)])
def test_native_attention_rank3_direct_ir_supports_broadcast_masks(mask_shape) -> None:
    dtype = DType.F32
    values = {
        "query": Value("query", TensorType((2, 5, 16), dtype)),
        "key": Value("key", TensorType((2, 7, 16), dtype)),
        "value": Value("value", TensorType((2, 7, 16), dtype)),
        "mask": Value("mask", TensorType(mask_shape, dtype)),
        "output": Value("output", TensorType((2, 5, 16), dtype)),
    }
    graph = Graph(
        ("query", "key", "value", "mask"),
        values,
        (
            Operation(
                "attention",
                ("query", "key", "value", "mask"),
                ("output",),
                {"scale": 16**-0.5, "has_mask": True, "causal": False},
            ),
        ),
        ("output",),
    )
    result = compile_graph(graph, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]

    torch.manual_seed(75)
    inputs = (
        torch.randn(2, 5, 16),
        torch.randn(2, 7, 16),
        torch.randn(2, 7, 16),
        torch.randn(mask_shape) * 0.2,
    )
    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(
        torch.nn.functional.scaled_dot_product_attention(
            inputs[0], inputs[1], inputs[2], attn_mask=inputs[3]
        ).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_native_attention_rank3_boolean_mask_on_simd_path() -> None:
    dtype = DType.F16
    values = {
        "query": Value("query", TensorType((2, 8, 16), dtype)),
        "key": Value("key", TensorType((2, 7, 16), dtype)),
        "value": Value("value", TensorType((2, 7, 16), dtype)),
        "mask": Value("mask", TensorType((8, 7), DType.BOOL)),
        "output": Value("output", TensorType((2, 8, 16), dtype)),
    }
    graph = Graph(
        ("query", "key", "value", "mask"),
        values,
        (
            Operation(
                "attention",
                ("query", "key", "value", "mask"),
                ("output",),
                {"scale": 16**-0.5, "has_mask": True, "causal": False},
            ),
        ),
        ("output",),
    )
    result = compile_graph(graph, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]
    assert "tile_probabilities_low" in result.plan.kernels[0].source

    torch.manual_seed(76)
    mask = torch.rand(8, 7) > 0.25
    mask[1, :] = False
    inputs = (
        torch.randn(2, 8, 16, dtype=torch.float16),
        torch.randn(2, 7, 16, dtype=torch.float16),
        torch.randn(2, 7, 16, dtype=torch.float16),
        mask,
    )
    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = mx.array(
        torch.nn.functional.scaled_dot_product_attention(
            inputs[0], inputs[1], inputs[2], attn_mask=inputs[3]
        ).numpy()
    )
    empty_row = mx.zeros((2, 16), dtype=mx.float16)
    mx.eval(actual, expected, empty_row)
    assert mx.array_equal(actual[:, 1, :], empty_row)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("batch", [1, 2])
def test_native_decode_attention_launches_one_threadgroup_per_head(
    batch, monkeypatch
) -> None:
    head_dim = 8
    heads = 4
    key_length = 8
    dtype = DType.F32
    values = {
        "query": Value("query", TensorType((batch, heads, 1, head_dim), dtype)),
        "key": Value("key", TensorType((batch, heads, key_length, head_dim), dtype)),
        "value": Value(
            "value", TensorType((batch, heads, key_length, head_dim), dtype)
        ),
        "mask": Value("mask", TensorType((batch, 1, 1, key_length), DType.BOOL)),
        "output": Value("output", TensorType((batch, heads, 1, head_dim), dtype)),
    }
    graph = Graph(
        ("query", "key", "value", "mask"),
        values,
        (
            Operation(
                "attention",
                ("query", "key", "value", "mask"),
                ("output",),
                {"scale": head_dim**-0.5, "has_mask": True, "causal": False},
            ),
        ),
        ("output",),
    )

    launches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    cached_kernel = metal_msl._metal_kernel

    def recording_kernel(name, input_names, output_names, source, header):
        kernel = cached_kernel(name, input_names, output_names, source, header)

        def launch(**kwargs):
            launches.append((tuple(kwargs["grid"]), tuple(kwargs["threadgroup"])))
            return kernel(**kwargs)

        return launch

    monkeypatch.setattr(metal_msl, "_metal_kernel", recording_kernel)
    result = compile_graph(graph, codegen_attention=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["attention"]
    assert "threadgroup float scores[8]" in result.plan.kernels[0].source

    torch.manual_seed(11)
    query = torch.randn(batch, heads, 1, head_dim)
    key = torch.randn(batch, heads, key_length, head_dim)
    value = torch.randn(batch, heads, key_length, head_dim)
    reference_query = query.double()
    reference_key = key.double()
    reference_value = value.double()
    for valid_lengths in ((key_length, key_length), (1, key_length), (5, 2)):
        valid_lengths = valid_lengths[:batch]
        mask = torch.zeros(batch, 1, 1, key_length, dtype=torch.bool)
        for row, valid_length in enumerate(valid_lengths):
            mask[row, 0, 0, key_length - valid_length :] = True
        scores = reference_query @ reference_key.transpose(-1, -2)
        scores = scores * (head_dim**-0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        expected = (weights @ reference_value).to(torch.float32)

        actual = result.plan(
            *(mx.array(tensor.numpy()) for tensor in (query, key, value, mask))
        )[0]
        mx.eval(actual)
        assert mx.isfinite(actual).all().item()
        assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-4, rtol=1e-4)

    assert len(launches) == 3
    for grid, threadgroup in launches:
        threadgroups = tuple(
            -(-size // block) for size, block in zip(grid, threadgroup, strict=True)
        )
        # One threadgroup per (head, batch) pair: extra threadgroups would run
        # the same head twice and write past the end of the output buffer.
        assert threadgroups == (heads, batch, 1)


def test_native_attention_falls_back_above_threadgroup_limit() -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    inputs = tuple(torch.randn(1, 2, 3, 512) for _ in range(3))
    result = compile_callable(fn, inputs, codegen_attention=True)
    assert result.plan.kernels == ()


@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_decomposed_grouped_query_attention_is_fused(key_value_heads) -> None:
    def fn(query, key, value):
        batch, query_heads, sequence, head_dim = query.shape
        groups = query_heads // key_value_heads
        key = (
            key[:, :, None]
            .expand(batch, key_value_heads, groups, sequence, head_dim)
            .reshape(batch, query_heads, sequence, head_dim)
        )
        value = (
            value[:, :, None]
            .expand(batch, key_value_heads, groups, sequence, head_dim)
            .reshape(batch, query_heads, sequence, head_dim)
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, value)

    torch.manual_seed(67)
    inputs = (
        torch.randn(2, 8, 5, 16),
        torch.randn(2, key_value_heads, 5, 16),
        torch.randn(2, key_value_heads, 5, 16),
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["attention"]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_attention_rejects_non_divisible_grouped_query_heads() -> None:
    dtype = DType.F32
    values = {
        "q": Value("q", TensorType((2, 6, 4, 8), dtype)),
        "k": Value("k", TensorType((2, 4, 4, 8), dtype)),
        "v": Value("v", TensorType((2, 4, 4, 8), dtype)),
        "out": Value("out", TensorType((2, 6, 4, 8), dtype)),
    }
    graph = Graph(
        ("q", "k", "v"),
        values,
        (
            Operation(
                "attention",
                ("q", "k", "v"),
                ("out",),
                {"scale": 8**-0.5, "has_mask": False, "causal": False},
            ),
        ),
        ("out",),
    )
    with pytest.raises(ValueError, match="multiple of key/value heads"):
        compile_graph(graph)


def test_direct_scaled_dot_product_attention_rejects_dropout() -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.1
        )

    inputs = tuple(torch.randn(2, 3, 4, 8) for _ in range(3))
    with pytest.raises(NotImplementedError, match="dropout"):
        compile_callable(fn, inputs)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(torch.nn.Dropout(0.3), id="Dropout"),
        pytest.param(torch.nn.Dropout1d(0.3), id="Dropout1d"),
    ],
)
def test_dropout_in_training_mode_names_dropout_in_the_refusal(module) -> None:
    # The mask shows up as a bare ``aten.bernoulli_.float``, so the message has
    # to say what the user did and what fixes it rather than name the operator.
    module.train()
    with pytest.raises(NotImplementedError) as caught:
        compile_callable(module, (torch.randn(2, 3, 4),))
    message = str(caught.value)
    assert "dropout" in message
    assert "aten.bernoulli_.float" in message
    assert "eval" in message


def test_dropout_is_the_identity_in_eval_mode() -> None:
    module = torch.nn.Dropout(0.5)
    module.eval()
    values = torch.randn(2, 3, 4)
    compiled = compile_callable(module, (values,))
    _assert_values_match(compiled.plan(_torch_to_mlx(values))[0], values)


@pytest.mark.parametrize(
    ("fn", "name"),
    [
        pytest.param(lambda x: x + torch.rand(3), "aten.rand.default", id="rand"),
        pytest.param(lambda x: x + torch.randn(3), "aten.randn.default", id="randn"),
        pytest.param(torch.rand_like, "aten.rand_like.default", id="rand_like"),
        pytest.param(torch.randn_like, "aten.randn_like.default", id="randn_like"),
        pytest.param(
            lambda x: x + torch.randint(0, 5, (3,)).float(),
            "aten.randint.low",
            id="randint",
        ),
        pytest.param(
            lambda x: torch.multinomial(torch.ones(4), 2),
            "aten.multinomial.default",
            id="multinomial",
        ),
    ],
)
def test_random_creation_operators_are_refused_by_name(fn, name) -> None:
    # A plan carries no RNG state, so these cannot be made faithful. The
    # refusal has to name the operator so the message reads as a decision
    # rather than a missing case.
    with pytest.raises(ValueError, match=name):
        compile_callable(fn, (torch.randn(4, 3),))


@pytest.mark.parametrize(
    ("fn", "name"),
    [
        pytest.param(torch.nonzero, "nonzero", id="nonzero"),
        pytest.param(torch.unique, "unique", id="unique"),
        pytest.param(lambda x: torch.unique(x, dim=1), "unique", id="unique_dim"),
        pytest.param(
            lambda x: torch.masked_select(x, x > 0), "masked_select", id="masked_select"
        ),
        pytest.param(lambda x: x[x > 0], "boolean-mask indexing", id="mask_index"),
        pytest.param(
            lambda x: torch.bincount(torch.zeros(8, dtype=torch.int64)),
            "bincount",
            id="bincount",
        ),
    ],
)
def test_data_dependent_lengths_are_refused_by_name(fn, name) -> None:
    # The IR wants positive constant dimensions, so a result whose length is a
    # function of the values has nowhere to go. The node's metadata holds that
    # length as a symbol, and the refusal reads it there so the message names
    # the operator instead of leaking Torch's ``Eq(u0, 0)`` guard.
    with pytest.raises(ValueError, match=f"{name} .*returns a length"):
        compile_callable(fn, (torch.randn(4, 4),))


def test_one_hot_without_num_classes_names_the_argument_that_fixes_it() -> None:
    # ``one_hot`` reads its class count from the largest value, which is a guard
    # on the data. With ``num_classes`` given the operator is a comparison and
    # lowers normally, so the refusal has to point at that argument.
    labels = torch.randint(0, 5, (2, 3))
    with pytest.raises(ValueError, match="num_classes"):
        compile_callable(lambda t: torch.nn.functional.one_hot(t).float(), (labels,))
    compiled = compile_callable(
        lambda t: torch.nn.functional.one_hot(t, 5).float(), (labels,)
    )
    _assert_values_match(
        compiled.plan(_torch_to_mlx(labels))[0],
        torch.nn.functional.one_hot(labels, 5).float(),
    )


def test_a_comparison_over_the_values_is_refused_with_advice() -> None:
    # ``equal`` and ``allclose`` reduce over the data, so fake tracing has no
    # example result to hand back and the plan has no shape to compile to. The
    # message has to say where the comparison belongs instead of leaking
    # Torch's ``DataDependentOutputException``.
    value = torch.randn(4)
    for fn in (torch.equal, torch.allclose):
        with pytest.raises(ValueError, match="compare the plan's outputs"):
            compile_callable(fn, (value, value))


def test_a_float64_decomposition_says_where_the_float64_came_from() -> None:
    # ``quantile`` builds its result out of float64 constants, so the dtype
    # refusal fires even though every input is float32. Saying that in the
    # message is the difference between "pass float32" and "this operator is
    # out of range", and only one of them is actionable.
    for fn in (
        lambda x: torch.quantile(x, 0.5),
        lambda x: torch.nanquantile(x, 0.5),
        lambda x: torch.linalg.matrix_rank(x),
    ):
        with pytest.raises(ValueError, match="float64.*decomposition"):
            compile_callable(fn, (torch.randn(4, 4),))


def test_attention_is_not_fused_when_probabilities_are_observed() -> None:
    def fn(query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) * 0.1 + mask
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, value), probabilities

    inputs = (
        *(torch.randn(2, 4, 6, 8) for _ in range(3)),
        torch.zeros(2, 1, 6, 6),
    )
    result = compile_callable(fn, inputs)
    lowered_ops = [op.op for op in result.lowered_graph.ops]
    assert "attention" not in lowered_ops
    assert "softmax" in lowered_ops
    assert len(result.plan(*(mx.array(value.numpy()) for value in inputs))) == 2


def test_multi_head_attention_end_to_end_on_metal() -> None:
    def attention(query, key, value):
        batch, sequence, heads, head_dim = query.shape
        query = query.transpose(1, 2).reshape(batch * heads, sequence, head_dim)
        key = key.transpose(1, 2).reshape(batch * heads, sequence, head_dim)
        value = value.transpose(1, 2).reshape(batch * heads, sequence, head_dim)
        scores = torch.matmul(query, key.transpose(1, 2)) / (head_dim**0.5)
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.matmul(probabilities, value)
        return output.reshape(batch, heads, sequence, head_dim).transpose(1, 2)

    torch.manual_seed(37)
    inputs = tuple(torch.randn(2, 4, 3, 8) for _ in range(3))
    result = compile_callable(attention, inputs, codegen_matmul=True)
    lowered_ops = [op.op for op in result.lowered_graph.ops]
    assert "bmm" not in {op.op for op in result.input_graph.ops}
    assert "softmax" in {op.op for op in result.input_graph.ops}
    assert lowered_ops.count("attention") == 1
    assert "softmax" not in lowered_ops

    mlx_inputs = tuple(mx.array(value.numpy()) for value in inputs)
    actual = result.plan(*mlx_inputs)[0]
    query, key, value = mlx_inputs
    batch, sequence, heads, head_dim = query.shape
    query = mx.reshape(
        mx.transpose(query, (0, 2, 1, 3)), (batch * heads, sequence, head_dim)
    )
    key = mx.reshape(
        mx.transpose(key, (0, 2, 1, 3)), (batch * heads, sequence, head_dim)
    )
    value = mx.reshape(
        mx.transpose(value, (0, 2, 1, 3)), (batch * heads, sequence, head_dim)
    )
    scores = mx.matmul(query, mx.swapaxes(key, 1, 2)) / (head_dim**0.5)
    probabilities = mx.softmax(scores, axis=-1)
    expected = mx.matmul(probabilities, value)
    expected = mx.transpose(
        mx.reshape(expected, (batch, heads, sequence, head_dim)), (0, 2, 1, 3)
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_two_layer_decoder_transformer_end_to_end_on_metal() -> None:
    hidden_size = 32
    num_heads = 4
    head_dim = 8
    intermediate_size = 64
    vocabulary_size = 48
    batch_size = 2
    sequence_length = 5

    def rms_norm(value, weight):
        return torch.nn.functional.rms_norm(value, (hidden_size,), weight, eps=1e-6)

    def apply_rope(value, cosine, sine):
        half = value.shape[-1] // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        return value * cosine + rotated * sine

    def decoder_layer(value, *parameters):
        (
            attention_norm,
            qkv_weight,
            output_weight,
            mlp_norm,
            gate_weight,
            up_weight,
            down_weight,
            cosine,
            sine,
        ) = parameters
        normalized = rms_norm(value, attention_norm)
        query, key, attention_value = (normalized @ qkv_weight).split(
            hidden_size, dim=-1
        )
        query = query.reshape(
            batch_size, sequence_length, num_heads, head_dim
        ).transpose(1, 2)
        key = key.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(
            1, 2
        )
        attention_value = attention_value.reshape(
            batch_size, sequence_length, num_heads, head_dim
        ).transpose(1, 2)
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, attention_value, is_causal=True
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, hidden_size
        )
        value = value + attention_output @ output_weight

        normalized = rms_norm(value, mlp_norm)
        gate = normalized @ gate_weight
        up = normalized @ up_weight
        mlp_output = (torch.nn.functional.silu(gate) * up) @ down_weight
        return value + mlp_output

    def transformer(token_ids, embedding, final_norm, lm_head, *layers):
        hidden = torch.nn.functional.embedding(token_ids, embedding)
        parameters_per_layer = 9
        for layer_index in range(2):
            start = layer_index * parameters_per_layer
            hidden = decoder_layer(
                hidden,
                *layers[start : start + parameters_per_layer],
            )
        return rms_norm(hidden, final_norm) @ lm_head

    torch.manual_seed(123)
    token_ids = torch.randint(
        0, vocabulary_size, (batch_size, sequence_length), dtype=torch.int32
    )
    inputs = [
        token_ids,
        torch.randn(vocabulary_size, hidden_size),
        torch.randn(hidden_size),
        torch.randn(hidden_size, vocabulary_size),
    ]

    frequencies = torch.arange(sequence_length, dtype=torch.float32)[
        :, None
    ] / torch.pow(
        torch.tensor(10000.0),
        torch.arange(0, head_dim, 2, dtype=torch.float32)[None, :] / head_dim,
    )
    frequencies = torch.cat((frequencies, frequencies), dim=-1)[None, None]
    cosine = frequencies.cos()
    sine = frequencies.sin()
    for _ in range(2):
        inputs.extend(
            (
                torch.randn(hidden_size),
                torch.randn(hidden_size, 3 * hidden_size),
                torch.randn(hidden_size, hidden_size),
                torch.randn(hidden_size),
                torch.randn(hidden_size, intermediate_size),
                torch.randn(hidden_size, intermediate_size),
                torch.randn(intermediate_size, hidden_size),
                cosine,
                sine,
            )
        )

    result = compile_callable(
        transformer,
        tuple(inputs),
        codegen_matmul=True,
        codegen_attention=True,
    )
    lowered_ops = [op.op for op in result.lowered_graph.ops]
    assert lowered_ops.count("attention") == 2
    assert lowered_ops.count("rms_norm") == 5
    assert lowered_ops.count("matmul") == 7
    assert lowered_ops.count("addmm") == 4
    assert "add" not in lowered_ops
    assert lowered_ops.count("rope") == 4
    assert lowered_ops.count("fused_elementwise") == 0
    assert lowered_ops.count("silu") == 0
    assert "mul" not in lowered_ops
    epilogue_ops = [
        op for op in result.lowered_graph.ops if op.attrs.get("epilogue") == "silu_mul"
    ]
    assert len(epilogue_ops) == 2
    assert "embedding" in lowered_ops
    assert "attention" in {kernel.kind for kernel in result.plan.kernels}
    assert "matmul" in {kernel.kind for kernel in result.plan.kernels}
    assert "rope" in {kernel.kind for kernel in result.plan.kernels}

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(transformer(*inputs).detach().numpy())
    mx.eval(actual, expected)
    assert actual.shape == (batch_size, sequence_length, vocabulary_size)
    assert mx.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_cat_lowering_preserves_concatenation_order() -> None:
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    torch.manual_seed(78)
    x = torch.randn(2, 3, 8)
    result = compile_callable(rotate_half, (x,))

    ops = [op.op for op in result.lowered_graph.ops]
    assert ops == ["slice", "neg", "slice", "cat"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["neg"]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(rotate_half(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected)


@pytest.mark.parametrize("head_dim", [8, 12, 6])
def test_rope_matches_rotate_half_reference(head_dim) -> None:
    """Fused rope must match the composite form, natively or via MLX."""

    def apply_rotary_embedding(x, cosine, sine):
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cosine + rotated * sine

    torch.manual_seed(91)
    inputs = (
        torch.randn(2, 3, head_dim),
        torch.randn(1, 1, head_dim),
        torch.randn(1, 1, head_dim),
    )
    result = compile_callable(apply_rotary_embedding, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["rope"]
    native = head_dim % 8 == 0
    assert [kernel.kind for kernel in result.plan.kernels] == (
        ["rope"] if native else []
    )

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(apply_rotary_embedding(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rope_keeps_shared_source_for_following_uses() -> None:
    """Fusing rope must not drop a source tensor that other ops still read."""

    def fn(x, cosine, sine):
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        rotated_input = x * cosine + rotated * sine
        return rotated_input, x.sum(dim=-1)

    torch.manual_seed(92)
    inputs = (
        torch.randn(2, 3, 8),
        torch.randn(1, 1, 8),
        torch.randn(1, 1, 8),
    )
    result = compile_callable(fn, inputs)
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))
    expected = fn(*inputs)
    mx.eval(*actual)
    for got, want in zip(actual, expected, strict=True):
        assert mx.allclose(got, mx.array(want.numpy()), atol=1e-6, rtol=1e-6)


def test_residual_add_folds_into_projection_epilogue() -> None:
    """``hidden + hidden @ w`` should become one ``addmm``, not an add kernel."""

    def block(hidden, weight):
        return hidden + hidden @ weight

    torch.manual_seed(93)
    hidden = torch.randn(2, 3, 8)
    weight = torch.randn(8, 8)
    result = compile_callable(block, (hidden, weight), codegen_matmul=True)
    lowered = [op.op for op in result.lowered_graph.ops]
    assert "add" not in lowered
    assert lowered.count("addmm") == 1
    assert "addmm" in {kernel.kind for kernel in result.plan.kernels}

    actual = result.plan(*(mx.array(value.numpy()) for value in (hidden, weight)))[0]
    expected = mx.array(block(hidden, weight).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_residual_add_survives_rank_changing_view() -> None:
    """A squeeze between the projection and the add must still fold."""

    def block(hidden, weight):
        projected = (hidden @ weight).reshape(hidden.shape[0], hidden.shape[1], -1)
        return hidden + projected

    torch.manual_seed(94)
    hidden = torch.randn(1, 1, 16)
    weight = torch.randn(16, 16)
    result = compile_callable(block, (hidden, weight), codegen_matmul=True)
    lowered = [op.op for op in result.lowered_graph.ops]
    assert "add" not in lowered
    assert lowered.count("addmm") == 1

    actual = result.plan(*(mx.array(value.numpy()) for value in (hidden, weight)))[0]
    expected = mx.array(block(hidden, weight).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_residual_add_keeps_shared_projection_output() -> None:
    """Folding must not fire when the projection output has other consumers."""

    def block(hidden, weight):
        projected = hidden @ weight
        return hidden + projected, projected.sum()

    torch.manual_seed(95)
    hidden = torch.randn(1, 1, 16)
    weight = torch.randn(16, 16)
    result = compile_callable(block, (hidden, weight), codegen_matmul=True)
    lowered = [op.op for op in result.lowered_graph.ops]
    assert "add" in lowered
    assert "addmm" not in lowered

    actual = result.plan(*(mx.array(value.numpy()) for value in (hidden, weight)))
    expected = block(hidden, weight)
    mx.eval(*actual)
    for got, want in zip(actual, expected, strict=True):
        assert mx.allclose(got, mx.array(want.numpy()), atol=1e-5, rtol=1e-5)


def test_activation_folds_into_projection_epilogue() -> None:
    """``silu(x @ gate) * (x @ up)`` should be one epilogue matmul, not two."""

    def mlp(x, gate_weight, up_weight):
        return torch.nn.functional.silu(x @ gate_weight) * (x @ up_weight)

    torch.manual_seed(96)
    inputs = (torch.randn(2, 3, 8), torch.randn(8, 16), torch.randn(8, 16))
    result = compile_callable(mlp, inputs, codegen_matmul=True)
    lowered = [op.op for op in result.lowered_graph.ops]
    assert "fused_elementwise" not in lowered
    assert "silu" not in lowered
    assert lowered.count("matmul") == 2
    epilogue_matmuls = [
        op
        for op in result.lowered_graph.ops
        if op.op == "matmul" and op.attrs.get("epilogue") == "silu_mul"
    ]
    assert len(epilogue_matmuls) == 1
    assert [kernel.kind for kernel in result.plan.kernels] == ["matmul", "matmul"]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(mlp(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_activation_folds_through_reshape_chain() -> None:
    """The epilogue must survive a pure reshape between projection and multiply."""

    def mlp(x, gate_weight, up_weight):
        gate = torch.nn.functional.silu(x @ gate_weight)
        up = (x @ up_weight).reshape(2, 48).reshape(2, 3, 16)
        return gate * up

    torch.manual_seed(97)
    inputs = (torch.randn(2, 3, 8), torch.randn(8, 16), torch.randn(8, 16))
    result = compile_callable(mlp, inputs, codegen_matmul=True)
    lowered = [op.op for op in result.lowered_graph.ops]
    assert "fused_elementwise" not in lowered
    assert any(
        op.attrs.get("epilogue") == "silu_mul" for op in result.lowered_graph.ops
    )

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(mlp(*inputs).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_activation_epilogue_requires_single_use_gate() -> None:
    """A gate with another consumer keeps the standalone activation for now."""

    def mlp(x, gate_weight, up_weight):
        gate = x @ gate_weight
        activated = torch.nn.functional.silu(gate)
        return activated * (x @ up_weight), gate.sum()

    torch.manual_seed(98)
    inputs = (torch.randn(1, 1, 8), torch.randn(8, 16), torch.randn(8, 16))
    result = compile_callable(mlp, inputs, codegen_matmul=True)
    assert not any(
        op.attrs.get("epilogue") == "silu_mul" for op in result.lowered_graph.ops
    )

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))
    expected = mlp(*inputs)
    mx.eval(*actual)
    for got, want in zip(actual, expected, strict=True):
        assert mx.allclose(got, mx.array(want.numpy()), atol=1e-5, rtol=1e-5)


def test_activation_epilogue_skips_non_projection_operand() -> None:
    """Both operands must trace back to a projection for the fold to apply."""

    def mlp(x, gate_weight, bias):
        return torch.nn.functional.silu(x @ gate_weight) * bias

    torch.manual_seed(99)
    inputs = (torch.randn(2, 3, 8), torch.randn(8, 16), torch.randn(2, 3, 16))
    result = compile_callable(mlp, inputs, codegen_matmul=True)
    assert not any(
        op.attrs.get("epilogue") == "silu_mul" for op in result.lowered_graph.ops
    )

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(mlp(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_rotary_embedding_rotate_half_compiles_and_runs() -> None:
    def apply_rotary_embedding(x, cosine, sine):
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cosine + rotated * sine

    torch.manual_seed(79)
    inputs = (
        torch.randn(2, 3, 8),
        torch.randn(1, 1, 8),
        torch.randn(1, 1, 8),
    )
    result = compile_callable(apply_rotary_embedding, inputs)
    lowered_ops = [op.op for op in result.lowered_graph.ops]
    assert lowered_ops == ["rope"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["rope"]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(apply_rotary_embedding(*inputs).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (2, 3, 8)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(("dim", "index"), [(1, -1), (-2, 0)])
def test_select_lowers_to_slice_and_squeeze(dim, index) -> None:
    def fn(x):
        return x.select(dim, index)

    torch.manual_seed(80)
    x = torch.randn(2, 3, 4)
    result = compile_callable(fn, (x,))

    axis = dim % x.ndim
    resolved_index = index % x.shape[axis]
    assert [op.op for op in result.lowered_graph.ops] == ["slice", "squeeze"]
    assert result.lowered_graph.ops[0].attrs == {
        "dim": axis,
        "start": resolved_index,
        "end": resolved_index + 1,
        "step": 1,
    }
    assert result.lowered_graph.ops[1].attrs == {"dim": axis}
    assert result.plan.kernels == ()

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x, a, b: x[a, :, b, :],
        lambda x, a, b: x[:, a, :, b],
        lambda x, a, b: x[a, :, b, a],
        lambda x, a, b: x[a, :, 0, b],
        lambda x, a, b: x[a, 0, :, b],
    ],
)
def test_non_adjacent_advanced_indexing_matches_pytorch(fn) -> None:
    # PyTorch moves the broadcast index dimensions to the front when index
    # tensors are separated by a fixed axis, so the frontend permutes the
    # indexed axes first instead of rejecting the spelling.
    torch.manual_seed(58)
    sample = torch.randn(3, 4, 5, 6)
    a = torch.randint(0, 3, (3,))
    b = torch.randint(0, 5, (3,))
    expected = fn(sample, a, b)
    result = compile_callable(fn, (sample, a, b))
    assert "index" in [op.op for op in result.lowered_graph.ops]
    actual = result.plan(_torch_to_mlx(sample), _torch_to_mlx(a), _torch_to_mlx(b))[0]
    mx.eval(actual)
    assert actual.shape == tuple(expected.shape)
    assert mx.array_equal(actual, _torch_to_mlx(expected))


def test_non_adjacent_advanced_indexing_permutes_before_the_gather() -> None:
    def fn(x, a, b):
        return x[a, :, b, :]

    sample = torch.randn(3, 4, 5, 6)
    a = torch.zeros(2, dtype=torch.int64)
    b = torch.zeros(2, dtype=torch.int64)
    result = compile_callable(fn, (sample, a, b))
    ops = result.lowered_graph.ops
    assert [op.op for op in ops] == ["permute", "index"]
    assert ops[0].attrs == {"dims": (0, 2, 1, 3)}
    assert ops[1].attrs == {"axes": (0, 1)}


def test_non_adjacent_advanced_indexing_rejects_out_of_range() -> None:
    def fn(x, i):
        return x[i, :, 0]

    x = torch.randn(3, 4, 5)
    result = compile_callable(fn, (x, torch.zeros(2, dtype=torch.int64)))
    with pytest.raises(IndexError):
        result.plan(mx.zeros((3, 4, 5)), mx.array([0, 5], dtype=mx.int64))


def test_adjacent_advanced_indexing_stays_one_op() -> None:
    def fn(x, i, j):
        return x[:, i, j]

    x = torch.randn(2, 3, 4)
    result = compile_callable(
        fn, (x, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    )
    assert [op.op for op in result.lowered_graph.ops] == ["index"]


def test_select_can_produce_a_scalar() -> None:
    def fn(x):
        return x.select(0, -2)

    x = torch.arange(3, dtype=torch.float32)
    result = compile_callable(fn, (x,))

    assert [op.op for op in result.lowered_graph.ops] == ["slice", "squeeze"]
    actual = result.plan(_torch_to_mlx(x))[0]
    expected = _torch_to_mlx(fn(x))
    mx.eval(actual, expected)
    assert actual.shape == ()
    assert mx.array_equal(actual, expected)


def test_chunk_lowering_fuses_qkv_pointwise_region() -> None:
    def qkv_gate(x):
        query, key, value = torch.chunk(x, 3, dim=-1)
        return torch.nn.functional.silu(query) * value + key

    torch.manual_seed(80)
    x = torch.randn(2, 3, 12)
    result = compile_callable(qkv_gate, (x,))

    assert [op.op for op in result.input_graph.ops].count("slice") == 3
    assert [op.op for op in result.lowered_graph.ops] == [
        "slice",
        "slice",
        "slice",
        "fused_elementwise",
    ]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(qkv_gate(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_split_with_sizes_supports_uneven_parts() -> None:
    def split_uneven(x):
        first, middle, last = torch.split(x, [1, 3, 2], dim=-1)
        return torch.cat((first + 1.0, middle * 2.0, last - 3.0), dim=-1)

    torch.manual_seed(81)
    x = torch.randn(2, 6)
    result = compile_callable(split_uneven, (x,))

    lowered_ops = [op.op for op in result.lowered_graph.ops]
    assert lowered_ops.count("slice") == 3
    assert "cat" in lowered_ops

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(split_uneven(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_unbind_lowering_returns_each_squeezed_tensor() -> None:
    def unbind_heads(x):
        return torch.unbind(x, dim=1)

    torch.manual_seed(82)
    x = torch.randn(2, 3, 4)
    result = compile_callable(unbind_heads, (x,))

    assert [op.op for op in result.lowered_graph.ops] == [
        "slice",
        "squeeze",
        "slice",
        "squeeze",
        "slice",
        "squeeze",
    ]

    actual = result.plan(mx.array(x.numpy()))
    expected = tuple(mx.array(value.numpy()) for value in unbind_heads(x))
    mx.eval(*actual, *expected)
    assert len(actual) == len(expected) == 3
    assert all(mx.allclose(a, e) for a, e in zip(actual, expected, strict=True))


def test_split_defaults_to_the_leading_axis() -> None:
    # ``torch.split(x, n)`` and ``torch.chunk(x, n)`` trace with ``dim=0``
    # elided, the same way ``unbind`` does, so the elided spelling has to work.
    torch.manual_seed(84)
    x = torch.randn(4, 6)

    split = compile_callable(lambda value: torch.split(value, 2), (x,))
    chunked = compile_callable(lambda value: torch.chunk(value, 2), (x,))

    assert [op.op for op in split.lowered_graph.ops] == ["slice", "slice"]
    assert [op.op for op in chunked.lowered_graph.ops] == ["slice", "slice"]

    actual = split.plan(mx.array(x.numpy()))
    expected = tuple(mx.array(value.numpy()) for value in torch.split(x, 2))
    mx.eval(*actual, *expected)
    assert len(actual) == len(expected) == 2
    assert all(mx.array_equal(a, e) for a, e in zip(actual, expected, strict=True))


def test_unbind_defaults_to_the_leading_axis() -> None:
    def unbind_leading(x):
        return torch.unbind(x)

    torch.manual_seed(83)
    x = torch.randn(3, 4, 5)
    result = compile_callable(unbind_leading, (x,))

    assert [op.op for op in result.lowered_graph.ops] == [
        "slice",
        "squeeze",
        "slice",
        "squeeze",
        "slice",
        "squeeze",
    ]

    actual = result.plan(mx.array(x.numpy()))
    expected = tuple(mx.array(value.numpy()) for value in unbind_leading(x))
    mx.eval(*actual, *expected)
    assert len(actual) == len(expected) == 3
    assert all(mx.allclose(a, e) for a, e in zip(actual, expected, strict=True))


@pytest.mark.parametrize(
    ("dim", "keepdim"),
    [
        (-1, False),
        (1, False),
        (0, True),
        ((0, 2), False),
        (None, False),
        (None, True),
    ],
)
def test_nansum_ignores_nan_like_pytorch(dim, keepdim) -> None:
    torch.manual_seed(84)
    x = torch.randn(2, 3, 4)
    x[0, 0, 0] = float("nan")
    x[1, 2, :] = float("nan")

    def fn(value):
        return torch.nansum(value, dim=dim, keepdim=keepdim)

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "fused_elementwise",
        "sum",
    ]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == tuple(expected.shape)
    assert mx.allclose(actual, expected)


def test_nansum_of_an_all_nan_slice_is_zero() -> None:
    x = torch.tensor([[1.0, float("nan")], [float("nan"), float("nan")]])
    result = compile_callable(lambda value: torch.nansum(value, dim=-1), (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert [float(value) for value in actual] == [1.0, 0.0]


def test_nansum_on_integers_skips_the_mask() -> None:
    x = torch.tensor([[3, 1], [2, 5]])
    result = compile_callable(lambda value: torch.nansum(value), (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["sum"]
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert int(actual) == int(torch.nansum(x))


def test_nansum_honours_the_requested_dtype() -> None:
    x = torch.randn(2, 4)
    x[0, 0] = float("nan")
    result = compile_callable(
        lambda value: torch.nansum(value, dim=-1, dtype=torch.float32), (x,)
    )
    assert [op.op for op in result.lowered_graph.ops] == [
        "fused_elementwise",
        "sum",
    ]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.nansum(x, dim=-1, dtype=torch.float32).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.float32
    assert mx.allclose(actual, expected)


@pytest.mark.parametrize(
    ("dim", "keepdim"),
    [(-1, False), (1, True), (0, False), (None, False), (None, True)],
)
def test_nanmean_matches_pytorch(dim, keepdim) -> None:
    torch.manual_seed(85)
    x = torch.randn(3, 4, 2)
    x[0, 1, 1] = float("nan")
    x[2, :, 0] = float("nan")

    def fn(value):
        return torch.nanmean(value, dim=dim, keepdim=keepdim)

    actual = compile_callable(fn, (x,)).plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    one_nan = mx.isnan(actual) ^ mx.isnan(expected)
    filled_actual = mx.where(mx.isnan(actual), 0.0, actual)
    filled_expected = mx.where(mx.isnan(expected), 0.0, expected)
    assert actual.shape == tuple(expected.shape)
    assert not bool(mx.any(one_nan))
    assert bool(mx.allclose(filled_actual, filled_expected, atol=1e-6, rtol=1e-6))


def test_nanmean_of_an_all_nan_slice_stays_nan() -> None:
    x = torch.tensor([[float("nan"), float("nan")], [1.0, 3.0]])
    result = compile_callable(lambda value: torch.nanmean(value, dim=-1), (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert bool(mx.isnan(actual[0]))
    assert float(actual[1]) == 2.0


def test_sum_accepts_a_none_dim_like_the_decompositions() -> None:
    torch.manual_seed(86)
    x = torch.randn(2, 3, 4)
    for keepdim in (False, True):
        result = compile_callable(lambda value: torch.sum(value, None, keepdim), (x,))
        actual = result.plan(mx.array(x.numpy()))[0]
        expected = mx.array(torch.sum(x, None, keepdim).numpy())
        mx.eval(actual, expected)
        assert actual.shape == tuple(expected.shape)
        assert mx.allclose(actual, expected)


@pytest.mark.parametrize("dim", [-1, 1, 0, (0, 2), None])
def test_count_nonzero_matches_pytorch(dim) -> None:
    torch.manual_seed(87)
    x = torch.randn(2, 3, 4)
    x[0, 0, 0] = 0.0
    x[1] = 0.0

    def fn(value):
        return torch.count_nonzero(value, dim=dim)

    result = compile_callable(fn, (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = fn(x)
    mx.eval(actual)
    assert actual.dtype == mx.int64
    assert actual.shape == tuple(expected.shape)
    assert actual.tolist() == expected.tolist()


def test_count_nonzero_reuses_an_existing_boolean_mask() -> None:
    torch.manual_seed(88)
    x = torch.randn(3, 5)
    result = compile_callable(
        lambda value: torch.count_nonzero(value > 0.0, dim=1), (x,)
    )
    # The comparison already produces the mask, so no extra ``ne`` is emitted.
    assert [op.op for op in result.lowered_graph.ops] == ["gt", "sum"]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = torch.count_nonzero(x > 0.0, dim=1)
    mx.eval(actual)
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("reduction", "dim", "keepdim"),
    [
        ("all", None, False),
        ("all", -1, False),
        ("all", 1, True),
        ("any", None, False),
        ("any", 0, False),
        ("any", (0, 2), False),
    ],
)
def test_all_and_any_match_pytorch(reduction, dim, keepdim) -> None:
    torch.manual_seed(89)
    x = torch.randn(2, 3, 4)
    x[0, 0, 0] = float("nan")
    x[1, 2, :] = 0.0

    def fn(value):
        return getattr(torch, reduction)(value, dim=dim, keepdim=keepdim)

    result = compile_callable(fn, (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = fn(x)
    mx.eval(actual)
    assert actual.dtype == mx.bool_
    assert actual.shape == tuple(expected.shape)
    assert actual.tolist() == expected.tolist()


def test_all_and_any_accept_a_boolean_input() -> None:
    torch.manual_seed(90)
    mask = torch.randn(4, 5) > 0.0

    for reduction in ("all", "any"):
        result = compile_callable(
            lambda value: getattr(torch, reduction)(value, dim=1), (mask,)
        )
        actual = result.plan(mx.array(mask.numpy()))[0]
        expected = getattr(torch, reduction)(mask, dim=1)
        mx.eval(actual)
        assert actual.tolist() == expected.tolist()


def test_squeeze_without_a_dim_drops_every_singleton_axis() -> None:
    torch.manual_seed(91)
    x = torch.randn(1, 3, 1, 4)

    result = compile_callable(lambda value: value.squeeze(), (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["reshape"]
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert actual.shape == (3, 4)
    assert mx.array_equal(actual, mx.array(x.squeeze().numpy()))


def test_squeeze_without_a_dim_keeps_a_fully_singleton_input() -> None:
    x = torch.tensor([[2.5]])
    result = compile_callable(lambda value: value.squeeze(), (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    mx.eval(actual)
    assert actual.shape == ()
    assert float(actual) == 2.5


def test_transformer_decoder_block_compiles_and_runs() -> None:
    def decoder_block(
        x,
        norm_weight,
        query_weight,
        key_weight,
        value_weight,
        output_weight,
        gate_weight,
        up_weight,
        down_weight,
        causal_mask,
    ):
        batch, sequence, hidden = x.shape
        heads, head_dim = 4, hidden // 4
        normalized = torch.nn.functional.rms_norm(x, (hidden,), norm_weight, eps=1e-6)
        query = (
            (normalized @ query_weight)
            .reshape(batch, sequence, heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            (normalized @ key_weight)
            .reshape(batch, sequence, heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            (normalized @ value_weight)
            .reshape(batch, sequence, heads, head_dim)
            .transpose(1, 2)
        )
        scores = (
            torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5) + causal_mask
        )
        probabilities = torch.softmax(scores, dim=-1)
        attention = torch.matmul(probabilities, value)
        attention = attention.transpose(1, 2).reshape(batch, sequence, hidden)
        x = x + attention @ output_weight

        normalized = torch.nn.functional.rms_norm(x, (hidden,), norm_weight, eps=1e-6)
        mlp = torch.nn.functional.silu(normalized @ gate_weight) * (
            normalized @ up_weight
        )
        x = x + mlp @ down_weight
        return torch.nn.functional.rms_norm(x, (hidden,), norm_weight, eps=1e-6)

    torch.manual_seed(43)
    batch, sequence, hidden, heads = 2, 6, 16, 4
    head_dim = hidden // heads
    inputs = (
        torch.randn(batch, sequence, hidden),
        torch.randn(hidden),
        *(torch.randn(hidden, hidden) for _ in range(4)),
        torch.randn(hidden, 32),
        torch.randn(hidden, 32),
        torch.randn(32, hidden),
        torch.zeros(batch, 1, sequence, sequence),
    )
    result = compile_callable(decoder_block, inputs)
    op_names = [op.op for op in result.input_graph.ops]
    lowered_ops = [op.op for op in result.lowered_graph.ops]
    assert op_names.count("rms_norm") == 3
    assert lowered_ops.count("attention") == 1
    assert "softmax" not in lowered_ops
    assert {"matmul", "rms_norm"} <= set(lowered_ops)
    assert "fused_elementwise" not in lowered_ops
    assert any(
        op.attrs.get("epilogue") == "silu_mul" for op in result.lowered_graph.ops
    )

    mlx_inputs = tuple(mx.array(value.numpy()) for value in inputs)
    actual = result.plan(*mlx_inputs)[0]
    (
        x,
        norm_weight,
        query_weight,
        key_weight,
        value_weight,
        output_weight,
        gate_weight,
        up_weight,
        down_weight,
        causal_mask,
    ) = mlx_inputs

    def normalized(value):
        return (
            value
            * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + 1e-6)
            * norm_weight
        )

    hidden_state = normalized(x)
    query = mx.transpose(
        mx.reshape(hidden_state @ query_weight, (batch, sequence, heads, head_dim)),
        (0, 2, 1, 3),
    )
    key = mx.transpose(
        mx.reshape(hidden_state @ key_weight, (batch, sequence, heads, head_dim)),
        (0, 2, 1, 3),
    )
    value = mx.transpose(
        mx.reshape(hidden_state @ value_weight, (batch, sequence, heads, head_dim)),
        (0, 2, 1, 3),
    )
    probabilities = mx.softmax(
        mx.matmul(query, mx.swapaxes(key, -1, -2)) * (head_dim**-0.5) + causal_mask,
        axis=-1,
    )
    attention = mx.reshape(
        mx.transpose(mx.matmul(probabilities, value), (0, 2, 1, 3)),
        (batch, sequence, hidden),
    )
    hidden_state = x + attention @ output_weight
    normalized_state = normalized(hidden_state)
    gate = normalized_state @ gate_weight
    mlp = gate / (1.0 + mx.exp(-gate)) * (normalized_state @ up_weight)
    expected = normalized(hidden_state + mlp @ down_weight)

    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fused_kernel_supports_float16_and_bfloat16(dtype) -> None:
    ir_dtype = DType.F16 if dtype == mx.float16 else DType.BF16
    graph = fuse_pointwise(_manual_gate_graph(ir_dtype))
    result = compile_graph(graph)
    x = mx.array([1.0, 2.0, 3.0], dtype=dtype)
    gate = mx.array([0.0, 1.0, -1.0], dtype=dtype)
    actual = result.plan(x, gate)[0]
    expected = x * (gate / (1.0 + mx.exp(-gate)))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_addmm_and_add_scalar_attributes_are_preserved() -> None:
    def fn(x, weight, bias):
        projected = torch.addmm(bias, x, weight, beta=0.5, alpha=2.0)
        return torch.add(projected, bias, alpha=3.0)

    torch.manual_seed(11)
    torch_inputs = (
        torch.randn(2, 4),
        torch.randn(4, 8),
        torch.randn(8),
    )
    result = compile_callable(fn, torch_inputs)
    assert result.input_graph.ops[0].attrs == {"alpha": 2.0, "beta": 0.5}
    assert result.input_graph.ops[1].attrs == {"alpha": 3.0}

    inputs = tuple(mx.array(value.numpy()) for value in torch_inputs)
    actual = result.plan(*inputs)[0]
    expected = 0.5 * inputs[2] + 2.0 * mx.matmul(inputs[0], inputs[1])
    expected = expected + 3.0 * inputs[2]
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_fused_add_alpha_is_preserved() -> None:
    tensor_type = TensorType((3,), DType.F32)
    values = {
        "x": Value("x", tensor_type),
        "gate": Value("gate", tensor_type),
        "silu": Value("silu", tensor_type),
        "out": Value("out", tensor_type),
    }
    ops = (
        Operation("silu", ("gate",), ("silu",)),
        Operation("add", ("x", "silu"), ("out",), {"alpha": 2.0}),
    )
    result = compile_graph(fuse_pointwise(Graph(("x", "gate"), values, ops, ("out",))))
    x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    gate = mx.array([0.0, 1.0, -1.0], dtype=mx.float32)
    actual = result.plan(x, gate)[0]
    expected = x + 2.0 * (gate / (1.0 + mx.exp(-gate)))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_single_tensor_output_is_supported() -> None:
    def fn(x):
        return torch.nn.functional.silu(x)

    result = compile_callable(fn, (torch.randn(4),))
    assert result.input_graph.outputs == ("v0",)


def test_runtime_rejects_mismatched_input_metadata() -> None:
    graph = fuse_pointwise(_manual_gate_graph())
    result = compile_graph(graph)
    with pytest.raises(ValueError, match="shape"):
        result.plan(mx.zeros((2,), dtype=mx.float32), mx.zeros((3,), dtype=mx.float32))
    with pytest.raises(TypeError, match="dtype"):
        result.plan(mx.zeros((3,), dtype=mx.float16), mx.zeros((3,), dtype=mx.float16))


def test_broadcast_pointwise_inputs_are_fused() -> None:
    output_type = TensorType((2, 3), DType.F32)
    bias_type = TensorType((1, 3), DType.F32)
    values = {
        "gate": Value("gate", output_type),
        "bias": Value("bias", bias_type),
        "activated": Value("activated", output_type),
        "out": Value("out", output_type),
    }
    graph = Graph(
        ("gate", "bias"),
        values,
        (
            Operation("silu", ("gate",), ("activated",)),
            Operation("add", ("bias", "activated"), ("out",)),
        ),
        ("out",),
    )
    result = compile_graph(fuse_pointwise(graph))
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]

    gate = mx.array([[0.0, 1.0, -1.0], [2.0, -2.0, 3.0]], dtype=mx.float32)
    bias = mx.array([[10.0, 20.0, 30.0]], dtype=mx.float32)
    actual = result.plan(gate, bias)[0]
    expected = bias + gate / (1.0 + mx.exp(-gate))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("name", "fn", "expected_literals"),
    [
        ("elu", lambda x: torch.nn.functional.elu(x), ()),
        ("elu alpha", lambda x: torch.nn.functional.elu(x, 1.5), ()),
        (
            "elu scale",
            lambda x: torch.ops.aten.elu.default(x, 1.5, 0.75, 1.0),
            ("0.75",),
        ),
        (
            "elu input_scale",
            lambda x: torch.ops.aten.elu.default(x, 1.2, 0.9, 2.0),
            ("0.9", "2.0"),
        ),
        (
            "selu",
            lambda x: torch.nn.functional.selu(x),
            ("1.0507009873554805", "1.6732632423543772"),
        ),
    ],
)
def test_elu_honours_the_aten_scale_arguments(name, fn, expected_literals) -> None:
    torch.manual_seed(4)
    x = torch.randn(3, 5)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["elu"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["elu"]
    source = result.plan.kernels[0].source
    for literal in expected_literals:
        assert literal in source

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_selu_keeps_the_positive_branch_unscaled_by_input_scale() -> None:
    """ATen scales only the negative branch by ``input_scale``."""

    positive = torch.tensor([0.5, 2.0])
    negative = torch.tensor([-0.5, -2.0])

    def fn(value):
        return torch.ops.aten.elu.default(value, 1.2, 0.9, 2.0)

    result = compile_callable(fn, (positive,))
    actual = result.plan(mx.array(positive.numpy()))[0]
    mx.eval(actual)
    assert mx.allclose(actual, mx.array([0.45, 1.8]))

    negative_result = compile_callable(fn, (negative,))
    actual_negative = negative_result.plan(mx.array(negative.numpy()))[0]
    mx.eval(actual_negative)
    expected_negative = mx.array(fn(negative).numpy())
    assert mx.allclose(actual_negative, expected_negative, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("approximate", [None, "tanh"])
def test_gelu_is_generated_as_a_metal_kernel(approximate) -> None:
    def fn(x):
        if approximate is None:
            return torch.nn.functional.gelu(x)
        return torch.nn.functional.gelu(x, approximate=approximate)

    torch.manual_seed(3)
    x = torch.randn(17)
    result = compile_callable(fn, (x,))
    assert [kernel.kind for kernel in result.plan.kernels] == ["gelu"]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_common_pointwise_chain_is_one_metal_kernel() -> None:
    def fn(x, y, ones, scale):
        first = torch.sigmoid(torch.tanh(torch.relu(x)))
        second = torch.rsqrt(torch.abs(y) + ones)
        third = second / scale
        return first + third - torch.erf(y)

    torch.manual_seed(5)
    inputs = (
        torch.randn(23),
        torch.randn(23),
        torch.ones(23),
        torch.full((23,), 3.0),
    )
    result = compile_callable(fn, inputs)
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("op_name", ["sin", "cos"])
def test_trigonometric_ops_are_generated_as_metal_kernels(op_name) -> None:
    operation = getattr(torch, op_name)

    def fn(angle):
        return operation(angle)

    torch.manual_seed(6)
    angle = torch.randn(2, 3, 11) * 4.0
    result = compile_callable(fn, (angle,))
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    assert [kernel.kind for kernel in result.plan.kernels] == [op_name]
    assert f"metal::{op_name}" in result.plan.kernels[0].source

    actual = result.plan(mx.array(angle.numpy()))[0]
    expected = mx.array(fn(angle).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("op_name", "values", "source_expression"),
    [
        ("log", (0.125, 0.5, 1.0, 2.0, 7.5), "metal::log("),
        ("log2", (0.125, 0.5, 1.0, 2.0, 7.5), "metal::log2("),
        ("log10", (0.125, 0.5, 1.0, 2.0, 7.5), "metal::log10("),
        ("log1p", (0.125, 0.5, 1.0, 2.0, 7.5), "log1p("),
        ("expm1", (-2.0, -0.5, -0.001, 0.0, 0.001, 0.5, 2.0), "tsc_expm1("),
        ("exp2", (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0), "metal::exp2("),
        ("floor", (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5), "metal::floor("),
        ("ceil", (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5), "metal::ceil("),
        ("round", (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5), "metal::rint("),
        ("trunc", (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5), "metal::trunc("),
        ("sign", (-2.0, -0.0, 0.0, 0.5, 3.0), "metal::sign("),
    ],
)
def test_additional_unary_ops_are_generated_as_metal_kernels(
    op_name, values, source_expression
) -> None:
    operation = getattr(torch, op_name)

    def fn(value):
        return operation(value)

    value = torch.tensor(values, dtype=torch.float32)
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    assert [kernel.kind for kernel in result.plan.kernels] == [op_name]
    assert source_expression in result.plan.kernels[0].source

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("op_name", "fn", "source_expression", "expected_attrs"),
    [
        (
            "softplus",
            lambda value: torch.nn.functional.softplus(value),
            "tsc_softplus(",
            {"beta": 1.0, "threshold": 20.0},
        ),
        (
            "softplus",
            lambda value: torch.nn.functional.softplus(value, beta=2.0, threshold=10.0),
            "tsc_softplus(",
            {"beta": 2.0, "threshold": 10.0},
        ),
        (
            "mish",
            lambda value: torch.nn.functional.mish(value),
            "tsc_mish(",
            {},
        ),
        (
            "logsigmoid",
            lambda value: torch.nn.functional.logsigmoid(value),
            "tsc_logsigmoid(",
            {},
        ),
        (
            "hardsigmoid",
            lambda value: torch.nn.functional.hardsigmoid(value),
            "metal::min(metal::max(",
            {},
        ),
    ],
    ids=[
        "softplus-default",
        "softplus-custom",
        "mish",
        "logsigmoid",
        "hardsigmoid",
    ],
)
def test_smooth_activations_are_generated_as_stable_metal_kernels(
    op_name, fn, source_expression, expected_attrs
) -> None:
    value = torch.tensor(
        [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0],
        dtype=torch.float32,
    )
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    assert result.lowered_graph.ops[0].attrs == expected_attrs
    assert [kernel.kind for kernel in result.plan.kernels] == [op_name]
    assert source_expression in result.plan.kernels[0].source

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_smooth_activations_fuse_with_surrounding_pointwise_ops() -> None:
    def fn(value):
        mish = torch.nn.functional.mish(value)
        softplus = torch.nn.functional.softplus(value, beta=0.75, threshold=5.0)
        return mish + softplus + torch.nn.functional.logsigmoid(value)

    torch.manual_seed(32)
    x = torch.randn(2, 3, 17)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]
    body = result.lowered_graph.ops[0].attrs["body"]
    assert [op.op for op in body] == [
        "mish",
        "softplus",
        "add",
        "logsigmoid",
        "add",
    ]
    source = result.plan.kernels[0].source
    assert "tsc_softplus(" in source
    assert "tsc_logsigmoid(" in source

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "fn",
    [
        lambda value: torch.nn.functional.softplus(value),
        lambda value: torch.nn.functional.mish(value),
        lambda value: torch.nn.functional.logsigmoid(value),
        lambda value: torch.nn.functional.hardsigmoid(value),
    ],
    ids=["softplus", "mish", "logsigmoid", "hardsigmoid"],
)
def test_smooth_activations_preserve_nonfinite_values(fn) -> None:
    value = torch.tensor(
        [-float("inf"), float("inf"), float("nan")], dtype=torch.float32
    )
    result = compile_callable(fn, (value,))
    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    for actual_value, expected_value in zip(
        actual.tolist(), expected.tolist(), strict=True
    ):
        if math.isnan(expected_value):
            assert math.isnan(actual_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_smooth_activations_support_low_precision(dtype) -> None:
    torch.manual_seed(33)
    x = torch.randn(2, 3, 13, dtype=dtype)

    def fn(value):
        return torch.nn.functional.mish(
            torch.nn.functional.softplus(value, beta=0.75, threshold=5.0)
        ) + torch.nn.functional.logsigmoid(value)

    result = compile_callable(fn, (x,))
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected = mx.array(fn(x.float()).numpy(), dtype=mlx_dtype)
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def _numerical_pointwise_chain(left, right):
    upper = torch.maximum(left, right)
    lower = torch.minimum(left, right)
    stable_sum = torch.logaddexp(upper, lower)
    angle = torch.atan2(lower, upper)
    gate = torch.nn.functional.hardsigmoid(upper - lower)
    active = torch.nn.functional.threshold(gate, 0.25, -1.0)
    return torch.exp2(active) + stable_sum + angle


@pytest.mark.parametrize("op_name", ["maximum", "minimum"])
def test_maximum_and_minimum_preserve_signed_zero_and_nan(op_name) -> None:
    operation = getattr(torch, op_name)
    left = torch.tensor([-0.0, 0.0, -0.0, 1.0, float("nan")])
    right = torch.tensor([0.0, -0.0, -0.0, 2.0, 3.0])
    result = compile_callable(lambda lhs, rhs: operation(lhs, rhs), (left, right))
    assert [op.op for op in result.lowered_graph.ops] == [op_name]
    assert [kernel.kind for kernel in result.plan.kernels] == [op_name]
    assert "metal::isnan(" in result.plan.kernels[0].source

    actual = result.plan(mx.array(left.numpy()), mx.array(right.numpy()))[0]
    expected = operation(left, right)
    mx.eval(actual)
    for actual_value, expected_value in zip(
        actual.tolist(), expected.tolist(), strict=True
    ):
        if math.isnan(expected_value):
            assert math.isnan(actual_value)
            continue
        assert actual_value == expected_value
        if expected_value == 0.0:
            assert math.copysign(1.0, actual_value) == math.copysign(
                1.0, expected_value
            )


def test_logaddexp_preserves_nonfinite_values() -> None:
    left = torch.tensor(
        [
            -float("inf"),
            float("inf"),
            -float("inf"),
            float("inf"),
            0.0,
            float("nan"),
        ]
    )
    right = torch.tensor(
        [
            -float("inf"),
            float("inf"),
            float("inf"),
            -float("inf"),
            -float("inf"),
            1.0,
        ]
    )
    result = compile_callable(torch.logaddexp, (left, right))
    assert [op.op for op in result.lowered_graph.ops] == ["logaddexp"]
    assert "tsc_logaddexp(" in result.plan.kernels[0].source

    actual = result.plan(mx.array(left.numpy()), mx.array(right.numpy()))[0]
    expected = torch.logaddexp(left, right)
    mx.eval(actual)
    for actual_value, expected_value in zip(
        actual.tolist(), expected.tolist(), strict=True
    ):
        if math.isnan(expected_value):
            assert math.isnan(actual_value)
        else:
            assert actual_value == expected_value


def test_atan2_preserves_signed_zero_and_infinite_quadrants() -> None:
    left = torch.tensor(
        [
            0.0,
            -0.0,
            0.0,
            -0.0,
            1.0,
            -1.0,
            float("inf"),
            -float("inf"),
            float("inf"),
            -float("inf"),
            3.0,
        ]
    )
    right = torch.tensor(
        [
            0.0,
            0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            float("inf"),
            float("inf"),
            -float("inf"),
            -float("inf"),
            4.0,
        ]
    )
    result = compile_callable(torch.atan2, (left, right))
    assert [op.op for op in result.lowered_graph.ops] == ["atan2"]
    assert "tsc_atan2(" in result.plan.kernels[0].source

    actual = result.plan(mx.array(left.numpy()), mx.array(right.numpy()))[0]
    expected = torch.atan2(left, right)
    mx.eval(actual)
    for actual_value, expected_value in zip(
        actual.tolist(), expected.tolist(), strict=True
    ):
        if math.isnan(expected_value):
            assert math.isnan(actual_value)
            continue
        assert actual_value == pytest.approx(expected_value, abs=1e-6)
        if expected_value == 0.0:
            assert math.copysign(1.0, actual_value) == math.copysign(
                1.0, expected_value
            )


@pytest.mark.parametrize(
    "keyword_arguments", [False, True], ids=["positional", "keyword"]
)
def test_threshold_matches_pytorch_arguments_and_boundaries(
    keyword_arguments,
) -> None:
    def fn(value):
        if keyword_arguments:
            return torch.ops.aten.threshold.default(value, threshold=0.25, value=-2.0)
        return torch.ops.aten.threshold.default(value, 0.25, -2.0)

    value = torch.tensor(
        [-float("inf"), -0.1, 0.25, 0.2501, float("inf"), float("nan")]
    )
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["threshold"]
    assert result.lowered_graph.ops[0].attrs == {
        "threshold": 0.25,
        "value": -2.0,
    }
    assert "> 0.25 ?" in result.plan.kernels[0].source

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = fn(value)
    mx.eval(actual)
    for actual_value, expected_value in zip(
        actual.tolist(), expected.tolist(), strict=True
    ):
        if math.isnan(expected_value):
            assert math.isnan(actual_value)
        else:
            assert actual_value == expected_value


def test_numerical_pointwise_ops_fuse_into_one_metal_kernel() -> None:
    torch.manual_seed(34)
    inputs = (torch.randn(2, 3, 17), torch.randn(2, 3, 17) * 0.5)
    result = compile_callable(_numerical_pointwise_chain, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]
    body = result.lowered_graph.ops[0].attrs["body"]
    assert [op.op for op in body] == [
        "maximum",
        "minimum",
        "logaddexp",
        "atan2",
        "sub",
        "hardsigmoid",
        "threshold",
        "exp2",
        "add",
        "add",
    ]
    source = result.plan.kernels[0].source
    for expression in (
        "tsc_logaddexp(",
        "tsc_atan2(",
        "metal::exp2(",
        "metal::isnan(",
    ):
        assert expression in source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(_numerical_pointwise_chain(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_numerical_pointwise_ops_support_low_precision(dtype) -> None:
    torch.manual_seed(35)
    inputs = (
        torch.randn(2, 3, 13, dtype=dtype),
        torch.randn(2, 3, 13, dtype=dtype) * 0.5,
    )
    result = compile_callable(_numerical_pointwise_chain, inputs)
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]

    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(
        *(mx.array(value.float().numpy(), dtype=mlx_dtype) for value in inputs)
    )[0]
    expected = mx.array(
        _numerical_pointwise_chain(*(value.float() for value in inputs)).numpy(),
        dtype=mlx_dtype,
    )
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_logsigmoid_rejects_backward_buffer_output() -> None:
    def fn(value):
        return torch.ops.aten.log_sigmoid_forward.default(value)

    with pytest.raises(NotImplementedError, match="backward buffer"):
        compile_callable(fn, (torch.randn(5),))


def test_square_ir_op_is_generated_as_a_metal_kernel() -> None:
    tensor_type = TensorType((5,), DType.F32)
    graph = Graph(
        ("x",),
        {
            "x": Value("x", tensor_type),
            "out": Value("out", tensor_type),
        },
        (Operation("square", ("x",), ("out",)),),
        ("out",),
    )
    result = compile_graph(graph)
    assert [kernel.kind for kernel in result.plan.kernels] == ["square"]
    assert "(input_0) * (input_0)" in result.plan.kernels[0].source

    value = mx.array([-2.0, -0.5, 0.0, 0.5, 3.0], dtype=mx.float32)
    actual = result.plan(value)[0]
    expected = value * value
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_additional_unary_ops_fuse_with_surrounding_pointwise_ops() -> None:
    def fn(x, y):
        return torch.log1p(torch.square(x)) + torch.expm1(torch.sign(y))

    x = torch.tensor([-0.5, 0.0, 0.25, 1.0], dtype=torch.float32)
    y = torch.tensor([-2.0, -0.0, 0.0, 3.0], dtype=torch.float32)
    result = compile_callable(fn, (x, y))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]
    source = result.plan.kernels[0].source
    assert "log1p(" in source
    assert "tsc_expm1(" in source
    assert "metal::sign(" in source

    actual = result.plan(mx.array(x.numpy()), mx.array(y.numpy()))[0]
    expected = mx.array(fn(x, y).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rope_style_trigonometric_region_is_one_metal_kernel() -> None:
    def fn(x1, x2, angle):
        return torch.cos(angle) * x1 + torch.sin(angle) * x2

    torch.manual_seed(8)
    inputs = (
        torch.randn(2, 3, 8),
        torch.randn(2, 3, 8),
        torch.randn(1, 1, 8) * 3.0,
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]
    source = result.plan.kernels[0].source
    assert "metal::cos" in source
    assert "metal::sin" in source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_metal_matmul_and_addmm_codegen() -> None:
    torch.manual_seed(11)
    inputs = (torch.randn(3, 4), torch.randn(4, 5), torch.randn(5))
    x, weight, bias = inputs

    def fn(left, right, bias_value):
        return torch.addmm(bias_value, left, right, beta=0.5, alpha=2.0)

    result = compile_callable(fn, inputs, codegen_matmul=True)
    assert [kernel.kind for kernel in result.plan.kernels] == ["addmm"]
    assert "in_0[" in result.plan.kernels[0].source
    assert "in_1[" in result.plan.kernels[0].source
    assert "in_2[" in result.plan.kernels[0].source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(x, weight, bias).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("codegen_matmul", [False, True])
def test_baddbmm_matches_pytorch_with_broadcast_batches(codegen_matmul) -> None:
    def fn(bias, left, right):
        return torch.baddbmm(bias, left, right, beta=0.25, alpha=-1.5)

    torch.manual_seed(12)
    inputs = (
        torch.randn(1, 2, 4),
        torch.randn(3, 2, 5),
        torch.randn(3, 5, 4),
    )
    result = compile_callable(fn, inputs, codegen_matmul=codegen_matmul)
    assert [op.op for op in result.lowered_graph.ops] == ["addmm"]
    expected_kind = "addmm" if codegen_matmul else None
    assert [kernel.kind for kernel in result.plan.kernels] == (
        [expected_kind] if expected_kind is not None else []
    )

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=5e-3)


def test_full_mlp_can_use_only_generated_metal_kernels() -> None:
    def mlp(x, gate_w, up_w, down_w):
        gate = x @ gate_w
        up = x @ up_w
        return (torch.nn.functional.silu(gate) * up) @ down_w

    torch.manual_seed(7)
    torch_inputs = (
        torch.randn(2, 16),
        torch.randn(16, 32),
        torch.randn(16, 32),
        torch.randn(32, 16),
    )
    result = compile_callable(mlp, torch_inputs, codegen_matmul=True)
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "matmul",
        "matmul",
        "matmul",
    ]

    actual = result.plan(*(mx.array(value.numpy()) for value in torch_inputs))[0]
    expected = mx.array(mlp(*torch_inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "mode", ["square", "scalar_exponent", "scalar_base", "tensor_exponent"]
)
def test_pow_variants_match_pytorch(mode) -> None:
    torch.manual_seed(21)
    if mode == "square":
        fn = lambda value: value.square()
        inputs = (torch.randn(17),)
    elif mode == "scalar_exponent":
        fn = lambda value: torch.pow(value, 2.5)
        inputs = (torch.rand(17) + 0.5,)
    elif mode == "scalar_base":
        fn = lambda value: torch.pow(2.0, value)
        inputs = (torch.randn(17),)
    else:
        fn = lambda left, right: torch.pow(left, right)
        inputs = (torch.rand(17) + 0.5, torch.rand(17) + 0.5)

    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["pow"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["pow"]
    assert "metal::pow(" in result.plan.kernels[0].source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_pow_fuses_with_surrounding_pointwise_ops() -> None:
    def fn(value):
        return torch.pow(torch.sigmoid(value), 2.0) + torch.exp(value)

    torch.manual_seed(22)
    x = torch.randn(31)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]
    assert result.plan.kernels[0].source.count("metal::pow(") == 1

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "mode", ["min_max", "min_only", "max_only", "tensor_bounds", "hardtanh"]
)
def test_clamp_variants_match_pytorch(mode) -> None:
    torch.manual_seed(24)
    x = torch.randn(2, 3)
    if mode == "min_max":
        fn = lambda value: torch.clamp(value, min=-0.25, max=0.5)
        inputs = (x,)
    elif mode == "min_only":
        fn = lambda value: torch.clamp(value, min=0.0)
        inputs = (x,)
    elif mode == "max_only":
        fn = lambda value: torch.clamp(value, max=0.0)
        inputs = (x,)
    elif mode == "tensor_bounds":
        fn = lambda value, lower, upper: torch.clamp(value, min=lower, max=upper)
        inputs = (x, torch.randn(1, 3), torch.randn(2, 1))
    else:
        fn = lambda value: torch.nn.functional.hardtanh(value, -0.5, 0.75)
        inputs = (x,)

    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["clamp"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["clamp"]
    source = result.plan.kernels[0].source
    assert "metal::max(" in source or mode == "max_only"
    assert "metal::min(" in source or mode == "min_only"

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_clamp_fuses_with_surrounding_pointwise_ops() -> None:
    def fn(value):
        return torch.clamp(torch.sigmoid(value), 0.2, 0.8) + torch.tanh(value)

    torch.manual_seed(25)
    x = torch.randn(31)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert result.plan.kernels[0].source.count("metal::min(") == 1

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_clamp_fuses_with_tensor_bounds() -> None:
    def fn(value, lower, upper):
        return torch.clamp(torch.sigmoid(value), min=lower, max=upper) + value

    torch.manual_seed(27)
    inputs = (torch.randn(2, 3), torch.randn(1, 3) - 0.5, torch.randn(2, 1))
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    body = result.lowered_graph.ops[0].attrs["body"]
    assert body[-2].attrs["min_input"] == 1
    assert body[-2].attrs["max_input"] == 2

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("nan_bound", [False, True])
def test_clamp_propagates_nan(nan_bound) -> None:
    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0])
    if nan_bound:
        fn = lambda value: torch.clamp(value, min=float("nan"), max=1.0)
    else:
        fn = lambda value: torch.clamp(value, min=-1.0, max=1.0)

    result = compile_callable(fn, (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(mx.isnan(actual), mx.isnan(expected))
    actual_values = mx.where(mx.isnan(actual), 0.0, actual)
    expected_values = mx.where(mx.isnan(expected), 0.0, expected)
    assert mx.allclose(actual_values, expected_values, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_clamp_supports_low_precision(dtype) -> None:
    torch.manual_seed(26)
    x = torch.randn(2, 3, 7, dtype=dtype)
    result = compile_callable(lambda value: torch.clamp(value, min=-0.5, max=0.5), (x,))
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected = mx.array(
        torch.clamp(x.float(), min=-0.5, max=0.5).numpy(),
        dtype=mlx_dtype,
    )
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("activation", ["leaky_relu", "elu", "hardswish"])
def test_additional_activations_match_pytorch(activation) -> None:
    torch.manual_seed(28)
    x = torch.randn(2, 3, 31)
    if activation == "leaky_relu":
        fn = lambda value: torch.nn.functional.leaky_relu(value, 0.17)
    elif activation == "elu":
        fn = lambda value: torch.nn.functional.elu(value, 1.3)
    else:
        fn = lambda value: torch.nn.functional.hardswish(value)

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [activation]
    assert [kernel.kind for kernel in result.plan.kernels] == [activation]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_additional_activations_fuse_with_surrounding_pointwise_ops() -> None:
    def fn(value):
        activated = torch.nn.functional.leaky_relu(value, 0.2)
        activated = torch.nn.functional.elu(activated, 0.7)
        return torch.nn.functional.hardswish(activated) + value

    torch.manual_seed(29)
    x = torch.randn(2, 3, 17)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    body = result.lowered_graph.ops[0].attrs["body"]
    assert [op.op for op in body] == ["leaky_relu", "elu", "hardswish", "add"]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("activation", ["leaky_relu", "elu", "hardswish"])
def test_additional_activations_preserve_signed_zero(activation) -> None:
    inputs = torch.tensor([-0.0, 0.0, float("nan"), float("inf"), float("-inf")])
    functions = {
        "leaky_relu": lambda value: torch.nn.functional.leaky_relu(value, 0.25),
        "elu": lambda value: torch.nn.functional.elu(value, 1.5),
        "hardswish": lambda value: torch.nn.functional.hardswish(value),
    }
    fn = functions[activation]
    result = compile_callable(fn, (inputs,))
    actual = result.plan(mx.array(inputs.numpy()))[0]
    expected = mx.array(fn(inputs).numpy())
    mx.eval(actual, expected)
    actual_values = actual.tolist()
    expected_values = expected.tolist()
    for actual_value, expected_value in zip(
        actual_values[:2], expected_values[:2], strict=True
    ):
        assert math.copysign(1.0, actual_value) == math.copysign(1.0, expected_value)
    assert math.isnan(actual_values[2]) == math.isnan(expected_values[2])
    finite = mx.isfinite(actual) & mx.isfinite(expected)
    assert mx.allclose(
        mx.where(finite, actual, 0.0),
        mx.where(finite, expected, 0.0),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("operation", ["clamp_min", "clamp_max"])
@pytest.mark.parametrize("tensor_bound", [False, True])
def test_one_sided_clamp_variants_match_pytorch(operation, tensor_bound) -> None:
    torch.manual_seed(30)
    x = torch.randn(2, 3, 7)
    reduce = torch.clamp_min if operation == "clamp_min" else torch.clamp_max
    if tensor_bound:
        bound = torch.randn(1, 3, 1) * 0.5
        fn = lambda value, limit: reduce(value * limit, limit)
        inputs = (x, bound)
    else:
        fn = lambda value: reduce(value, 0.25 if operation == "clamp_min" else -0.25)
        inputs = (x,)

    result = compile_callable(fn, inputs)
    if tensor_bound:
        assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
        body = result.lowered_graph.ops[0].attrs["body"]
        assert body[-1].op == operation
        key = "min_input" if operation == "clamp_min" else "max_input"
        assert body[-1].attrs[key] == 1
    else:
        assert [op.op for op in result.lowered_graph.ops] == [operation]
        assert [kernel.kind for kernel in result.plan.kernels] == [operation]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_extended_activations_support_low_precision(dtype) -> None:
    torch.manual_seed(31)
    x = torch.randn(2, 3, 13, dtype=dtype)

    def fn(value):
        value = torch.nn.functional.leaky_relu(value, 0.2)
        value = torch.nn.functional.elu(value, 0.7)
        return torch.nn.functional.hardswish(value)

    result = compile_callable(fn, (x,))
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected = mx.array(fn(x.float()).numpy(), dtype=mlx_dtype)
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pow_supports_low_precision(dtype) -> None:
    torch.manual_seed(23)
    x = torch.rand(2, 3, 7, dtype=dtype) + 0.5
    result = compile_callable(lambda value: value.square(), (x,))
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected_input = mx.array(x.float().numpy(), dtype=mlx_dtype)
    expected = expected_input * expected_input
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_scalar_pointwise_ops_are_fused() -> None:
    def fn(x):
        return torch.sigmoid((3.0 - x) * 2.0 + 1.0) / 4.0

    torch.manual_seed(19)
    x = torch.randn(31)
    result = compile_callable(fn, (x,))
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]
    inputs = result.input_graph.ops
    assert inputs[0].attrs["scalar_position"] == "lhs"
    assert "scalar" in inputs[1].attrs

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_static_size_fill_ops_generate_constants(dtype) -> None:
    def fn():
        return (
            torch.zeros((2, 3), dtype=dtype),
            torch.ones((2, 3), dtype=dtype),
            torch.full((2, 3), 2.5, dtype=dtype),
        )

    result = compile_callable(fn, ())
    assert [op.op for op in result.input_graph.ops] == ["zeros", "ones", "full"]
    assert [op.op for op in result.lowered_graph.ops] == ["zeros", "ones", "full"]
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "zeros",
        "ones",
        "full",
    ]

    actual = result.plan()
    expected = tuple(_torch_to_mlx(output) for output in fn())
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.shape == expected_value.shape
        assert mx.array_equal(actual_value, expected_value)


def test_new_fill_ops_use_static_output_metadata() -> None:
    template = torch.empty(4, 5, dtype=torch.float16)

    def fn(value):
        return (
            value.new_zeros((2, 3)),
            value.new_ones((2, 3), dtype=torch.float32),
            value.new_full((2, 3), 2.5, dtype=torch.bfloat16),
        )

    result = compile_callable(fn, (template,))
    assert [op.op for op in result.input_graph.ops] == [
        "new_zeros",
        "new_ones",
        "new_full",
    ]
    assert all(not op.inputs for op in result.input_graph.ops)
    assert [
        result.input_graph.value(op.outputs[0]).type.shape
        for op in result.input_graph.ops
    ] == [(2, 3), (2, 3), (2, 3)]

    actual = result.plan(_torch_to_mlx(template))
    expected = tuple(_torch_to_mlx(output) for output in fn(template))
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.shape == expected_value.shape
        assert mx.array_equal(actual_value, expected_value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_like_fill_ops_generate_static_constants(dtype) -> None:
    value = torch.zeros(2, 3, 5, dtype=dtype)

    def fn(template):
        return (
            torch.zeros_like(template),
            torch.ones_like(
                template,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            ),
            torch.full_like(template, 2.5),
        )

    result = compile_callable(fn, (value,))
    assert [op.op for op in result.input_graph.ops] == [
        "zeros_like",
        "ones_like",
        "full_like",
    ]
    assert [op.op for op in result.lowered_graph.ops] == [
        "zeros_like",
        "ones_like",
        "full_like",
    ]
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "zeros_like",
        "ones_like",
        "full_like",
    ]

    actual = result.plan(_torch_to_mlx(value))
    expected = tuple(_torch_to_mlx(output) for output in fn(value))
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.shape == expected_value.shape
        assert mx.array_equal(actual_value, expected_value)


def test_full_like_fuses_into_pointwise_region() -> None:
    def fn(value):
        return torch.full_like(value, 2.5) + value

    torch.manual_seed(47)
    value = torch.randn(3, 7)
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.input_graph.ops] == ["full_like", "add"]
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert len(result.plan.kernels) == 1
    assert "2.5" in result.plan.kernels[0].source

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_full_fuses_into_pointwise_region() -> None:
    def fn(value):
        return torch.full((3, 7), 2.5) + value

    torch.manual_seed(48)
    value = torch.randn(3, 7)
    result = compile_callable(fn, (value,))
    assert [op.op for op in result.input_graph.ops] == ["full", "add"]
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert len(result.plan.kernels) == 1
    assert "2.5" in result.plan.kernels[0].source

    actual = result.plan(mx.array(value.numpy()))[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_static_fill_cse_ignores_template_and_preserves_output_type() -> None:
    def fn(left, right):
        return (
            torch.ones((2, 3)),
            torch.ones((2, 3)),
            left.new_ones((2, 3)),
            right.new_ones((2, 3)),
            left.new_ones((2, 3), dtype=torch.float16),
        )

    inputs = (
        torch.empty(4, 5, dtype=torch.float32),
        torch.empty(6, 7, dtype=torch.float16),
    )
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.input_graph.ops] == [
        "ones",
        "ones",
        "new_ones",
        "new_ones",
        "new_ones",
    ]
    assert [op.op for op in result.lowered_graph.ops] == [
        "ones",
        "new_ones",
        "new_ones",
    ]
    assert len(result.plan.kernels) == 3

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))
    mx.eval(*actual)
    assert mx.array_equal(actual[0], actual[1])
    assert mx.array_equal(actual[2], actual[3])
    assert actual[0].dtype == mx.float32
    assert actual[2].dtype == mx.float32
    assert actual[4].dtype == mx.float16
    assert mx.array_equal(actual[2], mx.ones((2, 3), dtype=mx.float32))
    assert mx.array_equal(actual[4], mx.ones((2, 3), dtype=mx.float16))


def test_like_fill_cse_ignores_template_and_preserves_output_type() -> None:
    def fn(left, right):
        return (
            torch.ones_like(left),
            torch.ones_like(right),
            torch.ones_like(left, dtype=torch.float16),
        )

    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.input_graph.ops] == ["ones_like"] * 3
    assert [op.op for op in result.lowered_graph.ops] == ["ones_like"] * 2
    assert len(result.plan.kernels) == 2

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))
    mx.eval(*actual)
    assert mx.array_equal(actual[0], actual[1])
    assert actual[0].dtype == mx.float32
    assert actual[2].dtype == mx.float16
    assert mx.array_equal(actual[0], mx.ones((2, 3), dtype=mx.float32))
    assert mx.array_equal(actual[2], mx.ones((2, 3), dtype=mx.float16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_full_like_preserves_signed_zero_and_nonfinite_values(dtype) -> None:
    template = torch.empty(7, dtype=dtype)
    values = (-0.0, float("nan"), float("inf"), -float("inf"))

    def fn(value):
        return tuple(torch.full_like(value, fill) for fill in values)

    result = compile_callable(fn, (template,))
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(mx.array(template.float().numpy(), dtype=mlx_dtype))
    mx.eval(*actual)

    assert all(value.dtype == mlx_dtype for value in actual)
    assert math.copysign(1.0, actual[0][0].item()) < 0.0
    assert bool(mx.all(mx.isnan(actual[1])).item())
    assert bool(mx.all(mx.isinf(actual[2]) & (actual[2] > 0)).item())
    assert bool(mx.all(mx.isinf(actual[3]) & (actual[3] < 0)).item())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_static_full_preserves_signed_zero_and_nonfinite_values(dtype) -> None:
    values = (-0.0, float("nan"), float("inf"), -float("inf"))

    def fn():
        return tuple(torch.full((7,), fill, dtype=dtype) for fill in values)

    result = compile_callable(fn, ())
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan()
    mx.eval(*actual)

    assert all(value.dtype == mlx_dtype for value in actual)
    assert math.copysign(1.0, actual[0][0].item()) < 0.0
    assert bool(mx.all(mx.isnan(actual[1])).item())
    assert bool(mx.all(mx.isinf(actual[2]) & (actual[2] > 0)).item())
    assert bool(mx.all(mx.isinf(actual[3]) & (actual[3] < 0)).item())


def test_like_fill_supports_boolean_outputs() -> None:
    template = torch.zeros(11, dtype=torch.bool)

    def fn(value):
        return (
            torch.zeros_like(value),
            torch.ones_like(value),
            torch.full_like(value, 2),
        )

    result = compile_callable(fn, (template,))
    actual = result.plan(_torch_to_mlx(template))
    mx.eval(*actual)
    assert all(value.dtype == mx.bool_ for value in actual)
    assert not bool(mx.any(actual[0]).item())
    assert bool(mx.all(actual[1]).item())
    assert bool(mx.all(actual[2]).item())


def test_static_size_fill_supports_boolean_outputs() -> None:
    def fn():
        return (
            torch.zeros((11,), dtype=torch.bool),
            torch.ones((11,), dtype=torch.bool),
            torch.full((11,), 2, dtype=torch.bool),
        )

    result = compile_callable(fn, ())
    actual = result.plan()
    mx.eval(*actual)
    assert all(value.dtype == mx.bool_ for value in actual)
    assert not bool(mx.any(actual[0]).item())
    assert bool(mx.all(actual[1]).item())
    assert bool(mx.all(actual[2]).item())


def test_static_size_fill_supports_integer_outputs() -> None:
    def fn():
        return (
            torch.zeros((2, 3), dtype=torch.int32),
            torch.ones((2, 3), dtype=torch.int64),
            torch.full((2, 3), 2**53 + 1, dtype=torch.int64),
            torch.full((2, 3), -2.9, dtype=torch.int32),
        )

    result = compile_callable(fn, ())
    assert [op.op for op in result.lowered_graph.ops] == [
        "zeros",
        "ones",
        "full",
        "full",
    ]
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "zeros",
        "ones",
        "full",
        "full",
    ]

    actual = result.plan()
    expected = tuple(_torch_to_mlx(output) for output in fn())
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == expected_value.dtype
        assert mx.array_equal(actual_value, expected_value)


def test_like_and_new_fill_support_integer_outputs() -> None:
    template = torch.zeros(3, dtype=torch.int32)

    def fn(value):
        return (
            torch.full_like(value, -2.9),
            torch.ones_like(value, dtype=torch.int64),
            value.new_full((2,), 2**53 + 1, dtype=torch.int64),
        )

    result = compile_callable(fn, (template,))
    assert all(not op.inputs for op in result.lowered_graph.ops)
    actual = result.plan(_torch_to_mlx(template))
    expected = tuple(_torch_to_mlx(output) for output in fn(template))
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == expected_value.dtype
        assert mx.array_equal(actual_value, expected_value)


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1])
def test_integer_scalar_constant_preserves_int64_precision(value: int) -> None:
    result = compile_callable(lambda: torch.scalar_tensor(value, dtype=torch.int64), ())

    actual = result.plan()[0]
    mx.eval(actual)
    assert actual.dtype == mx.int64
    assert actual.item() == value


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_pointwise_ops_fuse_and_wrap(dtype) -> None:
    info = torch.iinfo(dtype)
    value = torch.tensor(
        [info.min, info.max, -7, -1, 0, 1, 7],
        dtype=dtype,
    )

    def fn(tensor):
        return (tensor + 1) * 3

    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["fused_elementwise"]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(fn(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "scalar"),
    [
        (torch.int32, 2**32 - 1),
        (torch.int32, 2**32),
        (torch.int32, 2**53 + 1),
        (torch.int64, 2**53 + 1),
        (torch.int64, 2**64 - 1),
    ],
)
def test_integer_scalar_operands_wrap_without_float_rounding(
    dtype, scalar: int
) -> None:
    value = torch.tensor([-2, -1, 0, 1, 2], dtype=dtype)

    def fn(tensor):
        return (tensor + scalar) * 3

    result = compile_callable(fn, (value,))
    assert result.input_graph.ops[0].attrs["scalar"] == scalar
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(fn(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("divisor", [3, -3])
def test_integer_floor_divide_and_remainder_match_pytorch(dtype, divisor: int) -> None:
    info = torch.iinfo(dtype)
    value = torch.tensor(
        [info.min, info.max, -7, -6, -1, 0, 1, 6, 7],
        dtype=dtype,
    )

    def fn(tensor):
        quotient = torch.floor_divide(tensor, divisor)
        remainder = torch.remainder(tensor, divisor)
        return quotient * 100 + remainder

    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(fn(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_minimum_floor_division_by_negative_one_wraps(dtype) -> None:
    info = torch.iinfo(dtype)
    value = torch.tensor([info.min], dtype=dtype)
    result = compile_callable(
        lambda tensor: torch.floor_divide(tensor, -1),
        (value,),
    )

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(torch.floor_divide(value, -1))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("divisor", [3, -3])
def test_integer_fmod_follows_the_dividend(dtype, divisor: int) -> None:
    # ``fmod`` truncates where ``remainder`` floors, so on a negative dividend
    # the two disagree; the kernel uses C's integer remainder, which is the
    # truncated form.
    info = torch.iinfo(dtype)
    value = torch.tensor([info.min, info.max, -7, -6, -1, 0, 1, 6, 7], dtype=dtype)

    result = compile_callable(lambda tensor: torch.fmod(tensor, divisor), (value,))
    source = result.plan.kernels[0].source
    assert "tsc_fmod_" in source

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(torch.fmod(value, divisor))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("divisor", [3.0, -3.0, 2.5])
def test_float_floor_divide_and_remainder_match_pytorch(dtype, divisor) -> None:
    value = torch.tensor([-7.5, -6.0, -1.0, 0.0, 1.0, 6.0, 7.5, 9.0], dtype=dtype)

    def fn(tensor):
        quotient = torch.floor_divide(tensor, divisor)
        remainder = torch.remainder(tensor, divisor)
        return quotient * 100 + remainder

    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    source = result.plan.kernels[0].source
    assert "tsc_floor_div_f32(" in source
    assert "tsc_remainder_f32(" in source

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(fn(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_scalar_dividend_remainder_matches_pytorch() -> None:
    # ``scalar % tensor`` is the Scalar_Tensor schema, so the frontend has to
    # hand the scalar over as the dividend and keep the result on the divisor's
    # sign. Both the float and the integer spelling have to work.
    divisors = torch.tensor([-3.0, -2.0, 2.0, 3.0, 0.5, -0.5])

    result = compile_callable(lambda value: torch.remainder(2.0, value), (divisors,))
    assert [op.op for op in result.lowered_graph.ops] == ["remainder"]
    actual = result.plan(_torch_to_mlx(divisors))[0]
    expected = _torch_to_mlx(torch.remainder(2.0, divisors))
    mx.eval(actual, expected)
    assert actual.tolist() == expected.tolist()
    assert actual.tolist() == [-1.0, 0.0, 0.0, 2.0, 0.0, 0.0]

    integers = torch.tensor([-3, -2, 2, 3], dtype=torch.int32)
    int_result = compile_callable(lambda value: torch.remainder(7, value), (integers,))
    int_actual = int_result.plan(_torch_to_mlx(integers))[0]
    int_expected = _torch_to_mlx(torch.remainder(7, integers))
    mx.eval(int_actual, int_expected)
    assert int_actual.dtype == mx.int32
    assert int_actual.tolist() == int_expected.tolist()
    assert int_actual.tolist() == [-2, -1, 1, 1]


@pytest.mark.parametrize("name", ["add", "sub", "div", "mul"])
def test_scalar_operand_schema_matches_the_tensor_spelling(name: str) -> None:
    # ATen decompositions spell scalar arithmetic with the ``.Scalar`` schemas
    # directly, so those have to lower exactly like the tensor spelling.
    value = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])
    scalar = 2.0
    tensor_op = getattr(getattr(torch.ops.aten, name), "Scalar")

    result = compile_callable(lambda tensor: tensor_op(tensor, scalar), (value,))
    assert [op.op for op in result.lowered_graph.ops] == [name]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(tensor_op(value, scalar))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("mul", torch.float32),
        ("div", torch.float32),
        ("sub", torch.float32),
        ("remainder", torch.float32),
        ("add", torch.int32),
        ("bitwise_and", torch.int32),
        ("bitwise_or", torch.int32),
        ("bitwise_xor", torch.int32),
        ("bitwise_not", torch.int32),
        ("logical_not", torch.bool),
    ],
)
def test_inplace_pointwise_spellings_are_legalized(name, dtype) -> None:
    # In-place arithmetic appears in ATen decompositions and in hand-written
    # models. The mutated tensor must have no other observer, which is the same
    # rule the in-place scatter family already follows.
    if dtype == torch.bool:
        value = torch.tensor([True, False, True, True])
    elif dtype == torch.int32:
        value = torch.tensor([1, 2, 3, 4], dtype=dtype)
    else:
        value = torch.tensor([4.5, -1.0, 0.0, 2.0])

    # The operand is a Python scalar so it binds as the ``.Scalar`` schema
    # instead of a traced constant tensor.
    if name in {"bitwise_not", "logical_not"}:
        argument = None
    elif dtype == torch.int32:
        argument = 5
    else:
        argument = 2.0

    def fn(tensor):
        mutated = tensor.clone()
        if argument is None:
            return getattr(mutated, f"{name}_")()
        return getattr(mutated, f"{name}_")(argument)

    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == [name]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(fn(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    if dtype == torch.bool:
        assert actual.tolist() == expected.tolist()
    else:
        assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("name", "value", "argument"),
    [
        ("clamp", torch.tensor([4.5, -1.0, 0.0, 2.0]), (-0.5, 1.5)),
        ("clamp_min", torch.tensor([4.5, -1.0, 0.0, 2.0]), (0.25,)),
        ("clamp_max", torch.tensor([4.5, -1.0, 0.0, 2.0]), (1.0,)),
        ("clamp_min", torch.tensor([1, 2, 3, 4], dtype=torch.int32), (3,)),
        ("clamp_max", torch.tensor([1, 2, 3, 4], dtype=torch.int32), (2,)),
    ],
)
def test_inplace_clamp_spellings_are_legalized(name, value, argument) -> None:
    # ``F.cosine_similarity`` decomposes into ``clamp_min_.Tensor``, so the
    # in-place clamp family has to legalize like the arithmetic spellings.
    def fn(tensor):
        mutated = tensor.clone()
        return getattr(mutated, f"{name}_")(*argument)

    result = compile_callable(fn, (value,))
    assert [op.op for op in result.lowered_graph.ops] == [name]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(fn(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_cosine_similarity_compiles_into_fused_pointwise_ops() -> None:
    torch.manual_seed(88)
    left = torch.randn(3, 5, 4)
    right = torch.randn(3, 5, 4)

    def fn(first, second):
        return torch.nn.functional.cosine_similarity(first, second, dim=-1)

    result = compile_callable(fn, (left, right))
    # The denominator's clamp, the dot product and the division all land in
    # one pointwise region once the norms are computed.
    assert "vector_norm" in [op.op for op in result.lowered_graph.ops]
    actual = result.plan(_torch_to_mlx(left), _torch_to_mlx(right))[0]
    expected = _torch_to_mlx(fn(left, right))
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_inplace_mul_keeps_reads_that_precede_the_write() -> None:
    # ``observed`` is computed from the value *before* the write, so lowering the
    # mutation as a new binding is faithful: only a reader placed after the
    # write would see a stale value.
    def fn(value):
        observed = value + 1.0
        value.mul_(2.0)
        return observed + value

    values = torch.randn(3)
    result = compile_callable(fn, (values,))
    actual = result.plan(_torch_to_mlx(values))[0]
    _assert_values_match(actual, fn(values))


@pytest.mark.parametrize(
    ("shape", "apply"),
    [
        pytest.param((3, 4), lambda value: value.t_().contiguous(), id="t_"),
        pytest.param(
            (3, 4),
            lambda value: value.transpose_(0, 1).contiguous(),
            id="transpose_2d",
        ),
        pytest.param(
            (2, 3, 4),
            lambda value: value.transpose_(0, 2).contiguous(),
            id="transpose_3d",
        ),
    ],
)
def test_inplace_layout_spellings_match_torch(shape, apply) -> None:
    # A layout write is a rebinding too: the node becomes the transposed view,
    # which is faithful because the tracer rebinds the name it wrote through.
    values = torch.randn(*shape)
    result = compile_callable(apply, (values,))
    actual = result.plan(_torch_to_mlx(values))[0]
    _assert_values_match(actual, apply(values.clone()))


@pytest.mark.parametrize(
    "apply",
    [
        lambda value: value.relu_(),
        lambda value: value.sigmoid_(),
        lambda value: value.tanh_(),
        lambda value: value.exp_(),
        lambda value: value.expm1_(),
        lambda value: value.log_(),
        lambda value: value.log1p_(),
        lambda value: value.neg_(),
        lambda value: value.abs_(),
        lambda value: value.sqrt_(),
        lambda value: value.rsqrt_(),
        lambda value: value.reciprocal_(),
        lambda value: value.floor_(),
        lambda value: value.ceil_(),
        lambda value: value.round_(),
        lambda value: value.trunc_(),
        lambda value: value.erf_(),
    ],
)
def test_inplace_unary_spellings_match_torch(apply) -> None:
    # Torch keeps the mutated spelling these ATen kernels have, so the unary
    # table has to accept ``relu_`` and friends alongside ``relu``.
    values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 1.0, 2.5])
    result = compile_callable(apply, (values,))
    actual = result.plan(_torch_to_mlx(values))[0]
    _assert_values_match(actual, apply(values.clone()))


@pytest.mark.parametrize(
    "apply",
    [
        lambda value: torch.nn.functional.silu(value, inplace=True),
        lambda value: torch.nn.functional.hardsigmoid(value, inplace=True),
        lambda value: torch.nn.functional.hardswish(value, inplace=True),
        lambda value: torch.nn.functional.mish(value, inplace=True),
        lambda value: torch.nn.functional.elu(value, inplace=True),
        lambda value: torch.nn.functional.leaky_relu(value, inplace=True),
        lambda value: torch.nn.functional.selu(value, inplace=True),
        lambda value: torch.nn.functional.hardtanh(value, 0.0, 6.0, inplace=True),
        lambda value: torch.ops.aten.gelu_(value),
    ],
)
def test_inplace_activations_that_decompose_into_copy_match_torch(apply) -> None:
    # These have no ``*_`` ATen kernel, so the decomposition computes the
    # functional form and writes it back with ``copy_``, which legalizes into
    # the same rebinding the arithmetic in-place ops use.
    values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 1.0, 2.5])
    result = compile_callable(apply, (values,))
    actual = result.plan(_torch_to_mlx(values))[0]
    _assert_values_match(actual, apply(values.clone()))


def test_fill_zero_and_masked_fill_mutations_match_torch() -> None:
    values = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

    def fill(value):
        return value.fill_(2.5)

    def zero(value):
        value.zero_()
        return value + 1.0

    def masked(value):
        return value.masked_fill_(value < 0.0, 0.0)

    for fn in (fill, zero, masked):
        result = compile_callable(fn, (values,))
        actual = result.plan(_torch_to_mlx(values))[0]
        _assert_values_match(actual, fn(values.clone()))


def test_copy_mutation_casts_and_broadcasts_like_torch() -> None:
    def narrow(value):
        return value.copy_(value.to(torch.float16) * 3.0)

    def broadcast(value):
        return value.copy_(value.sum(dim=0))

    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    for fn in (narrow, broadcast):
        result = compile_callable(fn, (values,))
        actual = result.plan(_torch_to_mlx(values))[0]
        _assert_values_match(actual, fn(values.clone()))


def test_boolean_mul_and_add_are_conjunction_and_disjunction() -> None:
    # ATen writes mask conjunctions as ``*`` and disjunctions as ``+`` on bool
    # tensors, which is how ``isclose`` combines its two comparisons.
    value = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0, float("nan")])

    conj = compile_callable(lambda x: (x > 0) * (x < 1), (value,))
    assert [op.op for op in conj.lowered_graph.ops] == ["fused_elementwise"]
    conj_actual = conj.plan(_torch_to_mlx(value))[0]
    conj_expected = _torch_to_mlx((value > 0) * (value < 1))
    mx.eval(conj_actual, conj_expected)
    assert conj_actual.dtype == mx.bool_
    assert conj_actual.tolist() == conj_expected.tolist()
    assert conj_actual.tolist() == [False, False, False, True, False, False]

    disj = compile_callable(lambda x: (x > 0) + (x < 1), (value,))
    disj_actual = disj.plan(_torch_to_mlx(value))[0]
    disj_expected = _torch_to_mlx((value > 0) + (value < 1))
    mx.eval(disj_actual, disj_expected)
    assert disj_actual.tolist() == disj_expected.tolist()


def test_isclose_lowers_to_one_fused_kernel() -> None:
    # ``isclose`` is an ATen composite: two comparisons, a scale and an offset,
    # combined with boolean ``*``/``&``. It has to compile as one kernel and
    # keep ATen's NaN and infinity rules.
    left = torch.tensor([1.0, 1.0000001, -1.0, float("nan"), float("inf"), 0.0])
    right = torch.tensor([1.0, 1.0, 1.0, 1.0, float("inf"), 1e-9])

    result = compile_callable(torch.isclose, (left, right))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(_torch_to_mlx(left), _torch_to_mlx(right))[0]
    expected = _torch_to_mlx(torch.isclose(left, right))
    mx.eval(actual, expected)
    assert actual.dtype == mx.bool_
    assert actual.tolist() == expected.tolist()
    # NaN is never close, matching infinities are, and a difference inside the
    # default ``atol + rtol * |other|`` window counts as close.
    assert actual.tolist() == [True, True, False, False, True, True]


def test_isclose_with_equal_nan_flips_only_the_nan_comparison() -> None:
    value = torch.tensor([float("nan"), 1.0])

    result = compile_callable(
        lambda tensor: torch.isclose(tensor, tensor, equal_nan=True), (value,)
    )
    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(torch.isclose(value, value, equal_nan=True))
    mx.eval(actual, expected)
    assert actual.tolist() == expected.tolist() == [True, True]


def test_heaviside_matches_pytorch_on_both_branches() -> None:
    value = torch.tensor([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0, float("nan")])
    boundary = torch.tensor([7.0, -7.0, 0.5, -0.5, 1.5, -1.5, 3.0])

    result = compile_callable(torch.heaviside, (value, boundary))
    assert [op.op for op in result.lowered_graph.ops] == ["heaviside"]
    assert "== 0.0f ?" in result.plan.kernels[0].source

    actual = result.plan(_torch_to_mlx(value), _torch_to_mlx(boundary))[0]
    expected = _torch_to_mlx(torch.heaviside(value, boundary))
    mx.eval(actual, expected)
    assert actual.tolist() == expected.tolist()
    # ATen compares against zero before the sign, so NaN takes the zero branch
    # and -0.0 counts as zero, where the boundary value wins.
    assert actual.tolist() == [0.0, 0.0, 0.5, -0.5, 1.0, 1.0, 0.0]


def test_heaviside_keeps_an_integer_dtype() -> None:
    value = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int32)
    boundary = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32)

    result = compile_callable(torch.heaviside, (value, boundary))
    actual = result.plan(_torch_to_mlx(value), _torch_to_mlx(boundary))[0]
    expected = _torch_to_mlx(torch.heaviside(value, boundary))
    mx.eval(actual, expected)
    assert actual.dtype == mx.int32
    assert actual.tolist() == expected.tolist() == [0, 0, 7, 1, 1]


def test_heaviside_rejects_boolean_operands() -> None:
    with pytest.raises(ValueError, match="boolean tensor operands"):
        compile_callable(
            torch.heaviside,
            (torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)),
        )


def test_float_remainder_keeps_the_divisor_sign() -> None:
    # ATen adjusts the truncated remainder towards the divisor, so the result
    # of a mixed-sign pair is non-negative for a positive divisor and vice
    # versa; fmod deliberately does neither.
    dividend = torch.tensor([-7.0, -7.0, 7.0, 7.0, 0.0])
    divisor = torch.tensor([3.0, -3.0, 3.0, -3.0, -3.0])

    result = compile_callable(torch.remainder, (dividend, divisor))
    assert [op.op for op in result.lowered_graph.ops] == ["remainder"]
    assert "tsc_remainder_f32(" in result.plan.kernels[0].source

    actual = result.plan(_torch_to_mlx(dividend), _torch_to_mlx(divisor))[0]
    expected = _torch_to_mlx(torch.remainder(dividend, divisor))
    mx.eval(actual, expected)
    assert actual.tolist() == expected.tolist()
    assert actual.tolist() == [2.0, -1.0, 1.0, -2.0, 0.0]


def test_fmod_follows_the_dividend_sign() -> None:
    dividend = torch.tensor([-7.5, -1.0, 0.0, 1.0, 7.5])
    divisor = torch.tensor([3.0, 3.0, 3.0, -3.0, 3.0])

    result = compile_callable(torch.fmod, (dividend, divisor))
    assert [op.op for op in result.lowered_graph.ops] == ["fmod"]
    assert "metal::fmod(" in result.plan.kernels[0].source

    actual = result.plan(_torch_to_mlx(dividend), _torch_to_mlx(divisor))[0]
    expected = _torch_to_mlx(torch.fmod(dividend, divisor))
    mx.eval(actual, expected)
    assert actual.tolist() == expected.tolist()
    assert actual.tolist() == [-1.5, -1.0, 0.0, 1.0, 1.5]


@pytest.mark.parametrize("weight", [0.25, 0.75])
def test_lerp_matches_pytorch_for_tensor_and_scalar_weights(weight) -> None:
    torch.manual_seed(23)
    start = torch.randn(16)
    end = torch.randn(16)
    weight_tensor = torch.full((16,), weight)

    cases = (
        (
            compile_callable(
                lambda self_, other: torch.lerp(self_, other, weight),
                (start, end),
            ),
            (start, end),
            torch.lerp(start, end, weight),
        ),
        (
            compile_callable(torch.lerp, (start, end, weight_tensor)),
            (start, end, weight_tensor),
            torch.lerp(start, end, weight_tensor),
        ),
    )

    for result, inputs, expected in cases:
        assert "tsc_lerp_f32(" in result.plan.kernels[0].source
        actual = result.plan(*(_torch_to_mlx(item) for item in inputs))[0]
        mx.eval(actual)
        assert mx.allclose(actual, _torch_to_mlx(expected), atol=1e-6, rtol=1e-6)


def test_lerp_switches_branch_near_one_half() -> None:
    # Both ATen branches agree to within rounding, but the branch that avoids
    # cancellation is the one that keeps the result near `end` exact.
    start = torch.tensor([1000.0, -1000.0])
    end = torch.tensor([1.0, -1.0])

    result = compile_callable(
        lambda self_, other, weight: torch.lerp(self_, other, weight),
        (start, end, torch.tensor([0.6, 0.6])),
    )
    actual = result.plan(
        _torch_to_mlx(start),
        _torch_to_mlx(end),
        _torch_to_mlx(torch.tensor([0.6, 0.6])),
    )[0]
    expected = torch.lerp(start, end, torch.tensor([0.6, 0.6]))
    mx.eval(actual)
    assert actual.tolist() == pytest.approx(expected.tolist(), rel=1e-6)


@pytest.mark.parametrize("value", [None, 0.5])
@pytest.mark.parametrize(
    ("operation", "operator"),
    [(torch.addcmul, "*"), (torch.addcdiv, "/")],
)
def test_addcmul_and_addcdiv_match_pytorch(operation, operator, value) -> None:
    torch.manual_seed(29)
    accumulator = torch.randn(3, 4)
    left = torch.randn(3, 4)
    right = torch.randn(3, 1).abs() + 0.5
    expression = "1.0" if value is None else repr(float(value))

    if value is None:

        def fn(acc, first, second):
            # Adding the accumulator keeps the op inside a fused region, which
            # is the path a real graph takes for a non-terminal operation.
            return operation(acc, first, second) + acc

    else:

        def fn(acc, first, second):
            return operation(acc, first, second, value=value) + acc

    result = compile_callable(fn, (accumulator, left, right))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]
    source = result.plan.kernels[0].source
    # ATen scales the numerator before dividing, so addcdiv emits
    # `self + value * t1 / t2` rather than `self + value * (t1 / t2)`.
    assert f"{expression} * (" in source
    if operator == "/":
        assert ") / (" in source

    actual = result.plan(
        _torch_to_mlx(accumulator), _torch_to_mlx(left), _torch_to_mlx(right)
    )[0]
    expected = _torch_to_mlx(fn(accumulator, left, right))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_addcmul_rejects_integer_operands() -> None:
    value = torch.arange(4, dtype=torch.int64)

    with pytest.raises(ValueError, match="floating-point"):
        compile_callable(
            lambda acc, left, right: torch.addcmul(acc, left, right),
            (value, value, value),
        )


def test_logit_matches_pytorch_with_and_without_eps() -> None:
    value = torch.tensor([0.0, 1.0, 0.5, -0.0001, 0.25, 0.75, 1e-9])

    cases = (
        (torch.logit, None),
        (lambda tensor: torch.logit(tensor, eps=1e-3), 1e-3),
    )
    for fn, eps in cases:
        result = compile_callable(fn, (value,))
        assert [op.op for op in result.lowered_graph.ops] == ["logit"]
        assert "metal::log(" in result.plan.kernels[0].source
        if eps is not None:
            assert result.lowered_graph.ops[0].attrs["eps"] == eps

        actual = result.plan(_torch_to_mlx(value))[0]
        _assert_values_match(actual, fn(value))


def test_logit_promotes_integer_inputs_like_pytorch() -> None:
    value = torch.tensor([0, 1, 2], dtype=torch.int64)

    result = compile_callable(torch.logit, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["cast", "logit"]
    actual = result.plan(_torch_to_mlx(value))[0]
    _assert_values_match(actual, torch.logit(value))


def test_erfc_frac_sinc_and_hypot_match_pytorch() -> None:
    value = torch.tensor([-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5])
    other = torch.tensor([3.0, -4.0, 0.5, 2.0, -2.0, 1.0, 0.25, -0.75])

    cases = (
        (torch.erfc, (value,), "tsc_erfc("),
        (torch.frac, (value,), "metal::trunc("),
        (torch.sinc, (value,), "tsc_sinc("),
        (torch.hypot, (value, other), "hypot("),
    )
    for fn, inputs, marker in cases:
        result = compile_callable(fn, inputs)
        assert marker in result.plan.kernels[0].source
        actual = result.plan(*(_torch_to_mlx(item) for item in inputs))[0]
        _assert_values_match(actual, fn(*inputs))


def test_sinc_returns_one_at_zero_and_near_zero() -> None:
    # GPU sin can flush a 1e-8 argument to zero, so the small-argument branch
    # has to reproduce ATen's exact 1 there rather than 0/pi_x.
    value = torch.tensor([0.0, -0.0, 1e-8, -1e-8, 1e-5])

    result = compile_callable(torch.sinc, (value,))
    assert "tsc_sinc(" in result.plan.kernels[0].source
    actual = result.plan(_torch_to_mlx(value))[0]
    expected = torch.sinc(value)
    mx.eval(actual)
    assert actual.tolist() == expected.tolist()


def test_copysign_follows_the_magnitude_and_signed_zero() -> None:
    magnitude = torch.tensor([3.0, -3.0, 0.0, -0.0, 2.5])
    sign = torch.tensor([-1.0, 1.0, -0.0, -0.0, float("nan")], dtype=torch.float32)

    result = compile_callable(torch.copysign, (magnitude, sign))
    assert "metal::copysign(" in result.plan.kernels[0].source
    actual = result.plan(_torch_to_mlx(magnitude), _torch_to_mlx(sign))[0]
    _assert_values_match(actual, torch.copysign(magnitude, sign))


def test_xlogy_short_circuits_a_zero_left_operand() -> None:
    value = torch.tensor([0.0, -0.0, 1.0, 2.0])
    other = torch.tensor([-1.0, 3.0, 2.0, float("nan")])

    result = compile_callable(torch.xlogy, (value, other))
    assert "tsc_xlogy_f32(" in result.plan.kernels[0].source
    actual = result.plan(_torch_to_mlx(value), _torch_to_mlx(other))[0]
    _assert_values_match(actual, torch.xlogy(value, other))


def test_xlogy_accepts_a_scalar_left_operand() -> None:
    value = torch.tensor([0.5, 1.0, 2.0])

    result = compile_callable(lambda tensor: torch.xlogy(2.0, tensor), (value,))
    assert result.lowered_graph.ops[0].attrs["scalar"] == 2.0
    actual = result.plan(_torch_to_mlx(value))[0]
    _assert_values_match(actual, torch.xlogy(2.0, value))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_nan_to_num_uses_the_dtype_finite_limits(dtype) -> None:
    value = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 1.0, -2.5], dtype=dtype
    )

    result = compile_callable(torch.nan_to_num, (value,))
    assert [op.op for op in result.lowered_graph.ops] == ["nan_to_num"]
    actual = result.plan(_torch_to_mlx(value))[0]
    expected = torch.nan_to_num(value)
    mx.eval(actual)
    assert actual.tolist() == expected.tolist()


def test_nan_to_num_honours_explicit_replacements() -> None:
    value = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])

    def fn(tensor):
        return torch.nan_to_num(tensor, nan=7.0, posinf=8.0, neginf=-9.0)

    result = compile_callable(fn, (value,))
    assert result.lowered_graph.ops[0].attrs == {
        "nan": 7.0,
        "posinf": 8.0,
        "neginf": -9.0,
    }
    actual = result.plan(_torch_to_mlx(value))[0]
    mx.eval(actual)
    assert actual.tolist() == [7.0, 8.0, -9.0, 1.0]


def test_nan_to_num_is_a_identity_on_integer_tensors() -> None:
    value = torch.tensor([1, 2, 3], dtype=torch.int64)

    result = compile_callable(torch.nan_to_num, (value,))
    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(torch.nan_to_num(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_dot_mv_and_trace_reuse_existing_lowerings() -> None:
    torch.manual_seed(31)
    left = torch.randn(6)
    right = torch.randn(6)
    matrix = torch.randn(3, 4)
    vector = torch.randn(4)
    square = torch.randn(4, 4)
    rectangular = torch.randn(3, 5)

    cases = (
        (torch.dot, (left, right), ["mul", "sum"]),
        (torch.mv, (matrix, vector), ["reshape", "matmul", "reshape"]),
        (torch.trace, (square,), ["reshape", "slice", "sum"]),
        (torch.trace, (rectangular,), ["reshape", "slice", "sum"]),
    )
    for fn, inputs, expected_ops in cases:
        result = compile_callable(fn, inputs)
        assert [op.op for op in result.lowered_graph.ops] == expected_ops

        actual = result.plan(*(_torch_to_mlx(item) for item in inputs))[0]
        expected = fn(*inputs)
        mx.eval(actual)
        assert actual.shape == tuple(expected.shape)
        assert actual.reshape(-1).tolist() == pytest.approx(
            expected.reshape(-1).tolist(), abs=1e-5, rel=1e-5
        )


def test_trace_slices_the_short_diagonal_of_a_rectangular_matrix() -> None:
    matrix = torch.arange(15, dtype=torch.float32).reshape(3, 5)

    result = compile_callable(torch.trace, (matrix,))
    assert result.lowered_graph.ops[1].attrs == {
        "dim": 0,
        "start": 0,
        "end": 18,
        "step": 6,
    }

    actual = result.plan(_torch_to_mlx(matrix))[0]
    expected = _torch_to_mlx(torch.trace(matrix))
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)
    assert actual.item() == 0.0 + 6.0 + 12.0


@pytest.mark.parametrize("decimals", [1, 2, -1])
def test_round_with_decimals_matches_pytorch(decimals: int) -> None:
    value = torch.tensor([1.2345, -1.2345, 2.55, -2.55, 0.0, 15.0, 0.06])

    result = compile_callable(
        lambda tensor: torch.round(tensor, decimals=decimals), (value,)
    )
    assert [op.op for op in result.lowered_graph.ops] == ["round"]
    assert result.lowered_graph.ops[0].attrs == {"decimals": decimals}
    assert "metal::rint(" in result.plan.kernels[0].source

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = torch.round(value, decimals=decimals)
    mx.eval(actual)
    assert actual.tolist() == pytest.approx(expected.tolist(), abs=1e-6, rel=1e-6)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_bitwise_ops_fuse_with_tensor_and_scalar_operands(dtype) -> None:
    left = torch.tensor([-9, -4, -1, 0, 1, 4, 9], dtype=dtype)
    right = torch.tensor([3, 7, -2, 5, -4, 2, 3], dtype=dtype)

    def fn(lhs, rhs):
        return ((lhs & rhs) | (lhs ^ 3)) ^ ~rhs

    result = compile_callable(fn, (left, right))
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(_torch_to_mlx(left), _torch_to_mlx(right))[0]
    expected = _torch_to_mlx(fn(left, right))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_unary_ops_fuse_end_to_end(dtype) -> None:
    info = torch.iinfo(dtype)
    value = torch.tensor(
        [info.min, -9, -4, -1, 0, 1, 4, 9, info.max],
        dtype=dtype,
    )

    def fn(tensor):
        return -tensor + tensor.abs() + tensor.sign() * 3 + torch.square(tensor)

    result = compile_callable(fn, (value,))
    assert "square" in [op.op for op in result.input_graph.ops]
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(_torch_to_mlx(value))[0]
    expected = _torch_to_mlx(fn(value))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_integer_comparison_selection_clamp_and_fill_fuse_end_to_end() -> None:
    value = torch.tensor([-9, -4, -1, 0, 1, 4, 9], dtype=torch.int64)
    other = torch.tensor([3, 7, -2, 5, -4, 2, 3], dtype=torch.int64)
    lower = torch.tensor([-5, -5, -5, -5, -5, -5, -5], dtype=torch.int64)
    upper = torch.tensor([5, 5, 5, 5, 5, 5, 5], dtype=torch.int64)

    def fn(tensor, rhs, minimum, maximum):
        selected = torch.where(tensor < rhs, tensor, 0)
        clipped = torch.clamp(selected, min=minimum, max=maximum)
        return clipped.masked_fill(selected == 0, -2.9)

    inputs = (value, other, lower, upper)
    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == ["fused_elementwise"]

    actual = result.plan(*(_torch_to_mlx(tensor) for tensor in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("operation", ["clamp_min", "clamp_max"])
def test_integer_one_sided_clamp_uses_neutral_limit(dtype, operation) -> None:
    info = torch.iinfo(dtype)
    value = torch.tensor([info.min, -3, 0, 3, info.max], dtype=dtype)
    operation_fn = torch.clamp_min if operation == "clamp_min" else torch.clamp_max
    bound = 1 if operation == "clamp_min" else -1
    result = compile_callable(lambda tensor: operation_fn(tensor, bound), (value,))

    body = result.lowered_graph.ops[0].attrs.get("body")
    pointwise_op = body[0] if body is not None else result.lowered_graph.ops[0]
    output_type = result.lowered_graph.value(pointwise_op.outputs[0]).type
    environment = {"arg0": _torch_to_mlx(value)}
    actual = metal_pointwise._evaluate_pointwise_mlx(
        pointwise_op, environment, output_type
    )
    expected = _torch_to_mlx(operation_fn(value, bound))
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected)


def test_integer_constants_are_common_subexpressions() -> None:
    def fn():
        return (
            torch.full((4,), 7, dtype=torch.int64),
            torch.full((4,), 7, dtype=torch.int64),
        )

    result = compile_callable(fn, ())
    assert [op.op for op in result.input_graph.ops] == ["full", "full"]
    assert [op.op for op in result.lowered_graph.ops] == ["full"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["full"]


def test_arange_forms_generate_integer_constants() -> None:
    def fn():
        return (
            torch.arange(8),
            torch.arange(2, 8),
            torch.arange(2, 8, 2),
            torch.arange(8, 2, -2),
            torch.arange(8, dtype=torch.int32),
        )

    result = compile_callable(fn, ())
    assert [op.op for op in result.input_graph.ops] == ["arange"] * 5
    assert [op.op for op in result.lowered_graph.ops] == ["arange"] * 5
    assert [kernel.kind for kernel in result.plan.kernels] == ["arange"] * 5

    actual = result.plan()
    expected = tuple(_torch_to_mlx(output) for output in fn())
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == expected_value.dtype
        assert mx.array_equal(actual_value, expected_value)


def test_arange_large_int64_values_do_not_round_through_float() -> None:
    start = 2**53
    result = compile_callable(lambda: torch.arange(start, start + 5), ())

    actual = result.plan()[0]
    expected = _torch_to_mlx(torch.arange(start, start + 5))
    mx.eval(actual, expected)
    assert actual.dtype == mx.int64
    assert mx.array_equal(actual, expected)


def test_arange_float_forms_match_pytorch_exactly() -> None:
    """Integer bounds with a float dtype take the same values as PyTorch."""

    def fn():
        return (
            torch.arange(8, dtype=torch.float32),
            torch.arange(0, 10, 2, dtype=torch.float32),
            torch.arange(0, 1, 0.1),
            torch.arange(1, 0, -0.25),
            torch.arange(0, 1, 0.1, dtype=torch.float16),
            torch.arange(0, 4, 0.5, dtype=torch.bfloat16),
        )

    result = compile_callable(fn, ())
    assert [kernel.kind for kernel in result.plan.kernels] == ["arange"] * 6

    actual = result.plan()
    expected = tuple(_torch_to_mlx(output) for output in fn())
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == expected_value.dtype
        assert mx.array_equal(actual_value, expected_value)


def _float32_ulp(value: float) -> float:
    """Spacing of float32 values near ``value``."""

    if value == 0:
        return 2.0**-149
    _, exponent = math.frexp(abs(value))
    return math.ldexp(1.0, exponent - 24)


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [(0.0, 1.0, 0.017), (0.0, 1.0, 0.07), (-1.5, 1.5, 0.3), (0.0, 1.0, 0.001)],
)
def test_arange_fractional_steps_stay_within_one_ulp(
    start: float, end: float, step: float
) -> None:
    """Metal has no double, so the fma path is compared in ulps, not exactly.

    PyTorch's own CPU kernel computes these sequences with a different
    intermediate precision, which is why it does not agree with the exactly
    rounded value either.
    """

    result = compile_callable(lambda: torch.arange(start, end, step), ())
    actual = result.plan()[0]
    expected = _torch_to_mlx(torch.arange(start, end, step))
    mx.eval(actual, expected)
    assert actual.shape == expected.shape

    values = actual.tolist()
    expected_values = expected.tolist()
    scale = max((abs(value) for value in expected_values), default=1.0) or 1.0
    tolerance = _float32_ulp(scale)
    for got, want in zip(values, expected_values, strict=True):
        assert abs(got - want) <= tolerance


def test_arange_float_output_rejects_fractional_integer_bounds() -> None:
    output_type = TensorType((3,), DType.I64)
    graph = Graph(
        (),
        {"out": Value("out", output_type)},
        (
            Operation(
                "arange",
                (),
                ("out",),
                {"start": 0.0, "end": 1.5, "step": 0.5},
            ),
        ),
        ("out",),
    )

    with pytest.raises(NotImplementedError, match="requires integer start"):
        compile_graph(graph)


def test_arange_is_common_subexpression() -> None:
    def fn():
        return torch.arange(8), torch.arange(8)

    result = compile_callable(fn, ())
    assert [op.op for op in result.input_graph.ops] == ["arange", "arange"]
    assert [op.op for op in result.lowered_graph.ops] == ["arange"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["arange"]


def test_arange_rejects_zero_step() -> None:
    output_type = TensorType((3,), DType.I64)
    graph = Graph(
        (),
        {"out": Value("out", output_type)},
        (
            Operation(
                "arange",
                (),
                ("out",),
                {"start": 0, "end": 3, "step": 0},
            ),
        ),
        ("out",),
    )

    with pytest.raises(ValueError, match="arange.step must be nonzero"):
        compile_graph(graph)


def test_arange_rejects_non_rank_one_output() -> None:
    output_type = TensorType((2, 2), DType.I64)
    graph = Graph(
        (),
        {"out": Value("out", output_type)},
        (
            Operation(
                "arange",
                (),
                ("out",),
                {"start": 0, "end": 4, "step": 1},
            ),
        ),
        ("out",),
    )

    with pytest.raises(ValueError, match="rank-1 tensor"):
        compile_graph(graph)


def test_arange_rejects_output_length_mismatch() -> None:
    output_type = TensorType((4,), DType.I64)
    graph = Graph(
        (),
        {"out": Value("out", output_type)},
        (
            Operation(
                "arange",
                (),
                ("out",),
                {"start": 0, "end": 3, "step": 1},
            ),
        ),
        ("out",),
    )

    with pytest.raises(ValueError, match="expected length 3"):
        compile_graph(graph)


def test_integer_constant_rejects_i64_overflow() -> None:
    output_type = TensorType((2,), DType.I64)
    graph = Graph(
        (),
        {"out": Value("out", output_type)},
        (
            Operation(
                "full",
                (),
                ("out",),
                {"value": 2**63},
            ),
        ),
        ("out",),
    )

    with pytest.raises(ValueError, match="overflows int64"):
        compile_graph(graph)


def test_arange_selects_the_float_kernel() -> None:
    result = compile_callable(lambda: torch.arange(8, dtype=torch.float32), ())
    assert [kernel.kind for kernel in result.plan.kernels] == ["arange"]
    assert "fma(factor," in result.plan.kernels[0].source


def test_metal_matmul_uses_shared_memory_tiles() -> None:
    torch.manual_seed(23)
    inputs = (torch.randn(11, 7), torch.randn(7, 13))
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    kernel = result.plan.kernels[0]

    assert kernel.name == "metal_matmul_tiled_64x64x32_4x4_0"
    assert "threadgroup float tile_a[2048]" in kernel.source
    assert "threadgroup float tile_b[2048]" in kernel.source
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in kernel.source
    assert "fma(av_0, bv_0, acc_0_0)" in kernel.source
    assert "acc_3_3" in kernel.source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array((inputs[0] @ inputs[1]).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_large_k_matmul_uses_8x4_microtile() -> None:
    inputs = (torch.randn(3, 1024), torch.randn(1024, 5))
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    kernel = result.plan.kernels[0]

    assert kernel.name == "metal_matmul_tiled_64x64x32_8x4_0"
    assert "acc_7_3" in kernel.source
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array((inputs[0] @ inputs[1]).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_handles_reduction_tail() -> None:
    torch.manual_seed(27)
    inputs = (torch.randn(67, 35), torch.randn(35, 71))
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    kernel = result.plan.kernels[0]

    assert "k_base + local_k < 35u" in kernel.source
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array((inputs[0] @ inputs[1]).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_matmul_supports_low_precision(dtype) -> None:
    torch.manual_seed(29)
    inputs = (
        torch.randn(5, 7, dtype=dtype),
        torch.randn(7, 9, dtype=dtype),
    )
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(
        *(mx.array(value.float().numpy(), dtype=mlx_dtype) for value in inputs)
    )[0]
    expected = mx.array((inputs[0] @ inputs[1]).float().numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    ("dtype", "simdgroup_type"),
    [(torch.float16, "half"), (torch.bfloat16, "bfloat")],
)
def test_metal_matmul_uses_simdgroup_for_complete_blocks(dtype, simdgroup_type) -> None:
    torch.manual_seed(30)
    inputs = (torch.randn(64, 35, dtype=dtype), torch.randn(35, 64, dtype=dtype))
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    kernel = result.plan.kernels[0]

    assert kernel.name == "metal_matmul_simdgroup_64x64x32_0"
    assert f"simdgroup_{simdgroup_type}8x8 frag_a[2]" in kernel.source
    assert "simdgroup_float8x8 acc[8]" in kernel.source
    assert "simdgroup_multiply_accumulate" in kernel.source
    assert "k_base + local_k < 35u" in kernel.source
    assert "vec<T," not in kernel.source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(inputs[0] @ inputs[1])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_matmul_uses_tensor_core_tiles(dtype) -> None:
    _require_tensor_cores()
    torch.manual_seed(33)
    inputs = (
        torch.randn(64, 64, dtype=dtype),
        torch.randn(64, 64, dtype=dtype),
    )
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    kernel = result.plan.kernels[0]
    source = kernel.source

    assert kernel.name == "metal_matmul_tensorcore_nn_64x64x32_2x2_0"
    assert "mpp::tensor_ops::matmul2d_descriptor(" in source
    assert "mpp::tensor_ops::matmul2d<" in source
    assert "get_left_input_cooperative_tensor" in source
    assert "get_destination_cooperative_tensor" in source
    assert "MetalPerformancePrimitives" in kernel.header

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(inputs[0] @ inputs[1])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


def test_backend_falls_back_when_the_gpu_cannot_load_tensor_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU without MetalPerformancePrimitives still has to produce results.

    Tensor cores are the fast path, not the only path. Forcing the capability
    probe off must select the simdgroup kernels and stay numerically correct;
    otherwise every machine older than the tile tables has no working matmul.
    """

    monkeypatch.setattr(metal_target, "_mpp_supported", lambda: False)

    torch.manual_seed(35)
    matmul_inputs = (
        torch.randn(64, 64, dtype=torch.float16),
        torch.randn(64, 64, dtype=torch.float16),
    )
    matmul = compile_callable(
        lambda x, weight: x @ weight, matmul_inputs, codegen_matmul=True
    )
    assert "tensorcore" not in matmul.plan.kernels[0].name

    actual = matmul.plan(*(_torch_to_mlx(value) for value in matmul_inputs))[0]
    expected = _torch_to_mlx(matmul_inputs[0] @ matmul_inputs[1])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)

    def attention(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    torch.manual_seed(36)
    attention_inputs = (
        torch.randn(1, 2, 17, 64, dtype=torch.float16),
        torch.randn(1, 2, 40, 64, dtype=torch.float16),
        torch.randn(1, 2, 40, 64, dtype=torch.float16),
    )
    compiled = compile_callable(attention, attention_inputs, codegen_attention=True)
    assert "tensorcore" not in compiled.plan.kernels[0].name

    actual = compiled.plan(*(_torch_to_mlx(value) for value in attention_inputs))[0]
    expected = _torch_to_mlx(attention(*attention_inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_a_device_that_rejects_tensor_core_kernels_still_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap `__HAVE_TENSOR__` cannot see: tensor ops exist and do not load.

    A paravirtualised GPU — hosted CI, cloud Macs — defines the macro and then
    refuses the metallib with "unsupported deferred-static-alloca-size function
    body". `compile` has to notice that, rebuild without tensor cores, and say
    why in the selection, because the alternative is an opaque `RuntimeError`
    on a user's first matmul.
    """

    monkeypatch.setattr(metal_target, "_TENSOR_CORES_BLOCKED", False)
    monkeypatch.setattr(metal_target, "_TENSOR_CORE_BLOCK_ERROR", None)
    assert metal_target._mpp_supported(), "this test needs the hardware gate on"

    create_kernel = metal_msl._metal_kernel

    def refuse_tensor_core_kernels(name, input_names, output_names, source, header):
        kernel = create_kernel(name, input_names, output_names, source, header)
        if "matmul2d" not in source:
            return kernel

        def reject(*args, **kwargs):
            raise RuntimeError(
                f"[metal::Device] Unable to load kernel {name}\n"
                " error: unsupported deferred-static-alloca-size function body"
            )

        return reject

    monkeypatch.setattr(metal_msl, "_metal_kernel", refuse_tensor_core_kernels)

    torch.manual_seed(38)
    inputs = (
        torch.randn(64, 64, dtype=torch.float16),
        torch.randn(64, 64, dtype=torch.float16),
    )
    plan = compile_callable(
        lambda x, weight: x @ weight, inputs, codegen_matmul=True
    ).plan

    assert "tensorcore" not in plan.kernels[0].name
    assert metal_target.tensor_core_fallback_reason() is not None
    assert "tensor cores rejected" in plan.selections[0].reason

    actual = plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(inputs[0] @ inputs[1])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


def test_tensor_core_support_is_checked_per_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device can load FP16 tensor ops and reject bfloat16 ones.

    bfloat16 needs hardware support that FP16 does not, and the two share one
    kernel source with a different template type, so nothing in the emitted text
    separates them: only the probe's dtype does. Refusing by template type
    simulates a device with exactly that asymmetry, and a probe that ran once
    and answered for both would keep the bfloat16 kernel it cannot load.
    """

    _require_tensor_cores()
    monkeypatch.setattr(metal_target, "_TENSOR_CORES_BLOCKED", False)
    monkeypatch.setattr(metal_target, "_TENSOR_CORE_BLOCK_ERROR", None)

    create_kernel = metal_msl._metal_kernel

    def refuse_bfloat16_tiles(name, input_names, output_names, source, header):
        kernel = create_kernel(name, input_names, output_names, source, header)
        if "tensorcore" not in name:
            return kernel

        def reject(*args, **kwargs):
            template = kwargs.get("template") or ()
            if any(value == mx.bfloat16 for _, value in template):
                raise RuntimeError(f"[metal::Device] Unable to load kernel {name}")
            return kernel(*args, **kwargs)

        return reject

    monkeypatch.setattr(metal_msl, "_metal_kernel", refuse_bfloat16_tiles)

    torch.manual_seed(39)
    half = (torch.randn(64, 64, dtype=torch.float16),) * 2
    half_plan = compile_callable(
        lambda x, weight: x @ weight, half, codegen_matmul=True
    ).plan
    assert "tensorcore" in half_plan.kernels[0].name

    # The blocked flag is process-wide, so this is the last compile that can
    # still see tensor cores.
    torch.manual_seed(40)
    single = (torch.randn(64, 64, dtype=torch.bfloat16),) * 2
    float_plan = compile_callable(
        lambda x, weight: x @ weight, single, codegen_matmul=True
    ).plan
    assert "tensorcore" not in float_plan.kernels[0].name

    actual = float_plan(*(_torch_to_mlx(value) for value in single))[0]
    expected = _torch_to_mlx(single[0] @ single[1])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("shape", "transposed"),
    [((512, 2048, 512), False), ((256, 256, 256), True)],
)
def test_float32_matmul_keeps_float32_accuracy(shape, transposed) -> None:
    """f32 must not reach the tensor cores: they have no f32 mode.

    A `T = float` `matmul2d` runs at bfloat16 precision — 1.6e-1 off on the
    first shape here, the same distance MLX's own f32 matmul lands. Both are
    tensor-core eligible shapes, so the assertion is that f32 stays on the
    scalar kernel and matches PyTorch to f32 accuracy instead.
    """

    rows, reduction, columns = shape
    torch.manual_seed(41)
    left = torch.randn(rows, reduction, dtype=torch.float32)
    right = torch.randn(reduction, columns, dtype=torch.float32)
    fn = (
        (lambda x, weight: x @ weight.T)
        if transposed
        else (lambda x, weight: x @ weight)
    )

    result = compile_callable(fn, (left, right), codegen_matmul=True)
    assert "tensorcore" not in result.plan.kernels[0].name

    actual = result.plan(*(_torch_to_mlx(value) for value in (left, right)))[0]
    expected = _torch_to_mlx(fn(left, right))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("batch_shape", "batch_count"),
    [((2,), 2), ((2, 3), 6)],
)
def test_metal_batched_matmul_codegen_supports_rank_three_and_four(
    batch_shape, batch_count
) -> None:
    torch.manual_seed(43)
    inputs = (
        torch.randn(*batch_shape, 5, 7),
        torch.randn(*batch_shape, 7, 4),
    )
    result = compile_callable(
        lambda left, right: left @ right,
        inputs,
        codegen_matmul=True,
    )
    kernel = result.plan.kernels[0]

    assert kernel.name == f"metal_matmul_tiled_64x64x32_4x4_batch{batch_count}_0"
    assert "threadgroup_position_in_grid.y" in kernel.source
    assert "batch * 35u" in kernel.source
    assert "batch * 28u" in kernel.source
    assert "batch * 20u" in kernel.source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(torch.matmul(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_codegen_handles_singleton_batch_dimension() -> None:
    torch.manual_seed(54)
    inputs = (
        torch.randn(1, 5, 7),
        torch.randn(1, 7, 4),
    )
    result = compile_callable(
        lambda left, right: left @ right,
        inputs,
        codegen_matmul=True,
    )

    assert result.plan.kernels[0].name == "metal_matmul_tiled_64x64x32_4x4_0"
    source = result.plan.kernels[0].source
    assert "threadgroup_position_in_grid.y" not in source
    assert "batch *" not in source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(torch.matmul(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_batched_matmul_vectorizes_batch_inputs() -> None:
    _require_tensor_cores()
    torch.manual_seed(44)
    inputs = (
        torch.randn(2, 3, 64, 32, dtype=torch.float16),
        torch.randn(2, 3, 32, 64, dtype=torch.float16),
    )
    result = compile_callable(
        lambda left, right: left @ right,
        inputs,
        codegen_matmul=True,
    )
    kernel = result.plan.kernels[0]

    assert kernel.name == "metal_matmul_tensorcore_nn_64x64x32_2x2_batch6_0"
    assert "threadgroup_position_in_grid.z" in kernel.source
    assert "batch * 2048u" in kernel.source
    assert "batch * 4096u" in kernel.source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(torch.matmul(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


def test_metal_batched_matmul_combines_transposes_with_batch_strides() -> None:
    torch.manual_seed(45)
    inputs = (
        torch.randn(2, 8, 4),
        torch.randn(2, 5, 8),
    )
    result = compile_callable(
        lambda left, right: left.transpose(-2, -1) @ right.transpose(-2, -1),
        inputs,
        codegen_matmul=True,
    )

    assert result.lowered_graph.ops[0].attrs == {
        "transpose_a": True,
        "transpose_b": True,
    }
    source = result.plan.kernels[0].source
    assert "batch * 32u + (k_base + local_k) * 4u" in source
    assert "batch * 40u + (col_base + local_column) * 8u" in source
    assert "batch * 20u" in source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(
        torch.matmul(inputs[0].transpose(-2, -1), inputs[1].transpose(-2, -1)).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_codegen_broadcasts_rank_four_batch_dimensions() -> None:
    torch.manual_seed(46)
    inputs = (
        torch.randn(2, 1, 5, 7),
        torch.randn(1, 3, 7, 4),
    )
    result = compile_callable(
        lambda left, right: left @ right,
        inputs,
        codegen_matmul=True,
    )

    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]
    source = result.plan.kernels[0].source
    assert "((batch / 3u) % 2u) * 35u" in source
    assert "batch % 3u * 28u" in source
    assert "batch * 20u" in source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(torch.matmul(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_codegen_folds_single_use_batch_expands() -> None:
    torch.manual_seed(47)
    inputs = (
        torch.randn(2, 1, 5, 7),
        torch.randn(1, 3, 7, 4),
    )

    def fn(left, right):
        return left.expand(2, 3, 5, 7) @ right.expand(2, 3, 7, 4)

    result = compile_callable(fn, inputs, codegen_matmul=True)
    assert [op.op for op in result.input_graph.ops].count("expand") >= 2
    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_batch_expand_is_kept_when_it_is_a_graph_output() -> None:
    torch.manual_seed(48)
    inputs = (
        torch.randn(2, 1, 5, 7),
        torch.randn(1, 3, 7, 4),
    )

    def fn(left, right):
        expanded = left.expand(2, 3, 5, 7)
        return expanded @ right.expand(2, 3, 7, 4), expanded

    result = compile_callable(fn, inputs, codegen_matmul=True)
    assert "expand" in [op.op for op in result.lowered_graph.ops]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))
    expected = tuple(mx.array(value.numpy()) for value in fn(*inputs))
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert mx.allclose(actual_value, expected_value, atol=1e-5, rtol=1e-5)


def test_metal_matmul_batch_expand_is_kept_when_shared() -> None:
    torch.manual_seed(49)
    inputs = (
        torch.randn(2, 1, 5, 7),
        torch.randn(1, 3, 7, 4),
    )

    def fn(left, right):
        expanded = left.expand(2, 3, 5, 7)
        return expanded @ right.expand(2, 3, 7, 4), expanded.sum(dim=-1)

    result = compile_callable(fn, inputs, codegen_matmul=True)
    assert "expand" in [op.op for op in result.lowered_graph.ops]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))
    expected = tuple(mx.array(value.numpy()) for value in fn(*inputs))
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert mx.allclose(actual_value, expected_value, atol=1e-5, rtol=1e-5)


def test_metal_matmul_batch_expand_is_kept_when_shape_is_not_inferable() -> None:
    torch.manual_seed(50)
    inputs = (
        torch.randn(1, 2, 5, 7),
        torch.randn(1, 2, 7, 4),
    )

    def fn(left, right):
        return left.expand(3, 2, 5, 7) @ right.expand(3, 2, 7, 4)

    result = compile_callable(fn, inputs, codegen_matmul=True)
    assert [op.op for op in result.lowered_graph.ops].count("expand") == 2

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_broadcast_combines_with_transpose() -> None:
    torch.manual_seed(51)
    inputs = (
        torch.randn(2, 1, 5, 7),
        torch.randn(1, 3, 5, 4),
    )

    def fn(left, right):
        return left.transpose(-2, -1).expand(2, 3, 7, 5) @ right.expand(2, 3, 5, 4)

    result = compile_callable(fn, inputs, codegen_matmul=True)
    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]
    assert result.lowered_graph.ops[0].attrs == {"transpose_a": True}
    source = result.plan.kernels[0].source
    assert "((batch / 3u) % 2u) * 35u" in source
    assert "batch % 3u * 20u" in source
    assert "(k_base + local_k) * 7u + row_base + local_row" in source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_broadcasts_rank_two_against_rank_four() -> None:
    torch.manual_seed(52)
    inputs = (
        torch.randn(5, 7),
        torch.randn(2, 3, 7, 4),
    )
    result = compile_callable(
        lambda left, right: left @ right,
        inputs,
        codegen_matmul=True,
    )

    assert [op.op for op in result.lowered_graph.ops] == ["matmul"]
    source = result.plan.kernels[0].source
    assert "batch * 28u" in source
    assert "batch * 20u" in source

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(torch.matmul(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_metal_matmul_broadcasts_batch_inputs() -> None:
    _require_tensor_cores()
    torch.manual_seed(53)
    inputs = (
        torch.randn(2, 1, 64, 32, dtype=torch.float16),
        torch.randn(1, 3, 32, 64, dtype=torch.float16),
    )
    result = compile_callable(
        lambda left, right: left @ right,
        inputs,
        codegen_matmul=True,
    )

    assert result.plan.kernels[0].name == (
        "metal_matmul_tensorcore_nn_64x64x32_2x2_batch6_0"
    )
    source = result.plan.kernels[0].source
    assert "((batch / 3u) % 2u) * 2048u" in source
    assert "batch % 3u * 2048u" in source
    assert "batch * 4096u" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(torch.matmul(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


def test_metal_matmul_codegen_folds_transposed_operand() -> None:
    _require_tensor_cores()
    torch.manual_seed(34)
    inputs = (
        torch.randn(64, 64, dtype=torch.float16),
        torch.randn(64, 64, dtype=torch.float16),
    )
    result = compile_callable(
        lambda left, right: left @ right.T,
        inputs,
        codegen_matmul=True,
    )

    assert result.lowered_graph.ops[0].attrs == {"transpose_b": True}
    kernel = result.plan.kernels[0]
    source = kernel.source
    assert kernel.name == "metal_matmul_tensorcore_nt_64x64x32_2x2_0"
    assert "(col_base + 0u + fm) * 64u" in source
    assert "vec<T, 8>" not in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(inputs[0] @ inputs[1].T)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


def test_metal_addmm_codegen_folds_transposed_operand() -> None:
    torch.manual_seed(35)
    inputs = (
        torch.randn(5),
        torch.randn(4, 3),
        torch.randn(4, 5),
    )

    def addmm(bias, left, right):
        return torch.addmm(bias, left.T, right, alpha=0.75, beta=1.25)

    result = compile_callable(addmm, inputs, codegen_matmul=True)
    assert result.lowered_graph.ops[0].attrs["transpose_a"] is True

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(addmm(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_addmm_tensor_core_applies_broadcast_bias() -> None:
    _require_tensor_cores()
    torch.manual_seed(32)
    inputs = (
        torch.randn(1, 64, dtype=torch.float16),
        torch.randn(64, 32, dtype=torch.float16),
        torch.randn(32, 64, dtype=torch.float16),
    )

    def addmm(bias, left, right):
        return torch.addmm(bias, left, right, alpha=0.75, beta=1.25)

    result = compile_callable(addmm, inputs, codegen_matmul=True)
    kernel = result.plan.kernels[0]
    assert kernel.name == "metal_addmm_tensorcore_nn_64x64x32_2x2_0"
    assert "0.75" in kernel.source and "1.25" in kernel.source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(addmm(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("bias_shape", [(5,), (1, 5), (3, 1)])
def test_addmm_codegen_broadcasts_bias(bias_shape) -> None:
    torch.manual_seed(31)
    inputs = (
        torch.randn(3, 4),
        torch.randn(4, 5),
        torch.randn(bias_shape),
    )
    result = compile_callable(
        lambda x, weight, bias: torch.addmm(bias, x, weight, alpha=0.75, beta=1.25),
        inputs,
        codegen_matmul=True,
    )
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(
        torch.addmm(inputs[2], inputs[0], inputs[1], alpha=0.75, beta=1.25).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_exact_matmul_tiles_skip_boundary_predicates() -> None:
    inputs = (torch.randn(64, 32), torch.randn(32, 64))
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    source = result.plan.kernels[0].source

    assert "row_ok_0" not in source
    assert "col_ok_0" not in source
    assert "if (" not in source


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 5), (2, 300)])
def test_metal_softmax_last_dim(shape) -> None:
    torch.manual_seed(37)
    x = torch.randn(shape)
    result = compile_callable(
        lambda value: torch.nn.functional.softmax(value, dim=-1), (x,)
    )
    kernel = result.plan.kernels[0]

    assert kernel.kind == "softmax"
    assert kernel.tensor_type.shape == shape
    assert "threadgroup_barrier" in kernel.source
    assert result.input_graph.ops[0].attrs == {"dim": -1}

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.nn.functional.softmax(x, dim=-1).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_safe_softmax_returns_zero_for_fully_masked_rows() -> None:
    torch.manual_seed(40)
    x = torch.randn(2, 7)
    x[0] = -torch.inf

    result = compile_callable(
        lambda value: torch.ops.aten._safe_softmax.default(value, -1), (x,)
    )
    assert [op.op for op in result.lowered_graph.ops] == ["safe_softmax"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["safe_softmax"]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.ops.aten._safe_softmax.default(x, -1).numpy())
    mx.eval(actual, expected)
    assert mx.array_equal(actual[0], mx.zeros((7,)))
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metal_softmax_supports_low_precision(dtype) -> None:
    torch.manual_seed(41)
    x = torch.randn(2, 3, 7, dtype=dtype)
    result = compile_callable(
        lambda value: torch.nn.functional.softmax(value, dim=-1), (x,)
    )
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16

    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected = mx.array(torch.nn.functional.softmax(x.float(), dim=-1).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dim, stride", [(0, 12), (1, 4)])
def test_metal_softmax_supports_non_last_dimension(dim, stride) -> None:
    torch.manual_seed(39)
    x = torch.randn(2, 3, 4)
    result = compile_callable(
        lambda value: torch.nn.functional.softmax(value, dim=dim), (x,)
    )
    kernel = result.plan.kernels[0]
    assert f"col * {stride}u" in kernel.source

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.nn.functional.softmax(x, dim=dim).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_multi_axis_softmax_compiles_through_reshape() -> None:
    """`softmax` takes one `dim`; the multi-axis form is flatten + reshape."""

    def fn(value):
        flat = value.reshape(value.shape[0], -1)
        return torch.nn.functional.softmax(flat, dim=-1).reshape(value.shape)

    torch.manual_seed(40)
    x = torch.randn(2, 3, 4)
    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "reshape",
        "softmax",
        "reshape",
    ]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_softmax_and_fused_pointwise_execute_as_separate_kernels() -> None:
    torch.manual_seed(43)
    x = torch.randn(3, 7)

    def fn(value):
        return torch.nn.functional.softmax(value, dim=-1) * 2.0 + 1.0

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "softmax",
        "fused_elementwise",
    ]
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "softmax",
        "fused_elementwise",
    ]

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_layer_norm_with_affine_parameters() -> None:
    torch.manual_seed(47)
    x = torch.randn(2, 3, 300)
    weight = torch.randn(300)
    bias = torch.randn(300)

    def fn(value, scale, shift):
        return torch.nn.functional.layer_norm(value, (300,), scale, shift, eps=1e-5)

    result = compile_callable(fn, (x, weight, bias))
    assert [op.op for op in result.input_graph.ops] == ["layer_norm"]
    assert result.input_graph.ops[0].attrs == {
        "normalized_shape": (300,),
        "eps": 1e-5,
        "has_weight": True,
        "has_bias": True,
    }
    kernel = result.plan.kernels[0]
    assert kernel.name == "metal_layer_norm_256_0"
    assert "threadgroup_barrier" in kernel.source
    assert "in_1[col]" in kernel.source
    assert "in_2[col]" in kernel.source

    actual = result.plan(
        mx.array(x.numpy()), mx.array(weight.numpy()), mx.array(bias.numpy())
    )[0]
    expected = mx.array(fn(x, weight, bias).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_layer_norm_without_affine_and_multidimensional_shape() -> None:
    torch.manual_seed(53)
    x = torch.randn(2, 3, 4)
    result = compile_callable(
        lambda value: torch.nn.functional.layer_norm(value, (3, 4), eps=1e-3),
        (x,),
    )
    assert result.input_graph.ops[0].attrs["normalized_shape"] == (3, 4)
    assert result.input_graph.ops[0].attrs["has_weight"] is False
    assert result.input_graph.ops[0].attrs["has_bias"] is False
    assert "metal_layer_norm_16_0" == result.plan.kernels[0].name

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.nn.functional.layer_norm(x, (3, 4), eps=1e-3).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layer_norm_supports_low_precision(dtype) -> None:
    torch.manual_seed(59)
    x = torch.randn(2, 3, 7, dtype=dtype)
    weight = torch.randn(7, dtype=dtype)
    bias = torch.randn(7, dtype=dtype)
    result = compile_callable(
        lambda value, scale, shift: torch.nn.functional.layer_norm(
            value, (7,), scale, shift
        ),
        (x, weight, bias),
    )
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(
        *(
            mx.array(value.float().numpy(), dtype=mlx_dtype)
            for value in (x, weight, bias)
        )
    )[0]
    expected = mx.array(
        torch.nn.functional.layer_norm(
            x.float(), (7,), weight.float(), bias.float()
        ).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_layer_norm_exposes_the_mean_and_rstd_statistics() -> None:
    torch.manual_seed(60)
    x = torch.randn(2, 3, 7)

    def fn(value):
        _, mean, rstd = torch.native_layer_norm(value, (7,), None, None, 1e-5)
        return mean * rstd

    result = compile_callable(fn, (x,))
    op = result.lowered_graph.ops[0]
    assert op.op == "layer_norm"
    # The statistics are outputs of the normalization kernel, not a second pass
    # over the input, so the whole graph is one kernel and one pointwise region.
    assert len(op.outputs) == 3
    assert [kernel.name for kernel in result.plan.kernels] == [
        "metal_layer_norm_8_0",
        "mul_1",
    ]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (2, 3, 1)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("index", [1, 2])
def test_layer_norm_statistics_work_without_the_normalized_output(index) -> None:
    torch.manual_seed(62)
    x = torch.randn(4, 9)

    def fn(value):
        return torch.native_layer_norm(value, (9,), None, None, 1e-3)[index]

    result = compile_callable(fn, (x,))
    ops = result.lowered_graph.ops
    assert [op.op for op in ops] == ["layer_norm"]
    # Reading only a statistic still needs the normalization kernel, because
    # that kernel is what reduces the moments; it writes the normalized value
    # anyway and the graph leaves it unused.
    assert len(ops[0].outputs) == 3
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_layer_norm_statistics_compose_with_the_normalized_output() -> None:
    torch.manual_seed(63)
    x = torch.randn(2, 5, 6)
    weight = torch.randn(6)
    bias = torch.randn(6)

    def fn(value, scale, shift):
        out, mean, rstd = torch.native_layer_norm(value, (6,), scale, shift, 1e-5)
        return out + rstd * mean

    result = compile_callable(fn, (x, weight, bias))
    ops = [op.op for op in result.lowered_graph.ops]
    assert ops[0] == "layer_norm"
    actual = result.plan(
        mx.array(x.numpy()), mx.array(weight.numpy()), mx.array(bias.numpy())
    )[0]
    expected = mx.array(fn(x, weight, bias).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_layer_norm_statistics_survive_an_artifact_round_trip(tmp_path) -> None:
    torch.manual_seed(64)
    x = torch.randn(2, 3, 5)

    def fn(value):
        out, _, rstd = torch.native_layer_norm(value, (5,), None, None, 1e-5)
        return out * rstd

    result = compile_callable(fn, (x,))
    path = tmp_path / "layer_norm_stats.aoto"
    save_compiled(result, path)
    loaded = load_compiled_module(path)
    actual = loaded(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_layer_norm_composes_with_matmul_and_pointwise() -> None:
    torch.manual_seed(61)
    inputs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8), torch.randn(8))

    def fn(value, weight, scale, shift):
        projected = value @ weight
        normalized = torch.nn.functional.layer_norm(projected, (8,), scale, shift)
        return torch.nn.functional.silu(normalized)

    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == [
        "matmul",
        "layer_norm",
        "silu",
    ]
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "layer_norm",
        "silu",
    ]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize(
    ("weighted", "fn"),
    [
        (
            False,
            lambda value, weight: (
                value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6)
            ),
        ),
        (
            False,
            lambda value, weight: (
                value / torch.sqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6)
            ),
        ),
        (
            False,
            lambda value, weight: (
                value * torch.rsqrt((value * value).mean(-1, keepdim=True) + 1e-6)
            ),
        ),
        (
            False,
            lambda value, weight: (
                value
                * torch.rsqrt(torch.add(value.pow(2).mean(-1, keepdim=True), 1e-6))
            ),
        ),
        (
            True,
            lambda value, weight: (
                weight
                * (value / torch.sqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6))
            ),
        ),
        (
            True,
            lambda value, weight: (
                (value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6))
                * weight
            ),
        ),
    ],
)
def test_rms_norm_folds_the_common_decomposition_spellings(weighted, fn) -> None:
    torch.manual_seed(79)
    x = torch.randn(3, 4, 6)
    weight = torch.randn(6)

    result = compile_callable(fn, (x, weight))
    assert [op.op for op in result.lowered_graph.ops] == ["rms_norm"]
    attrs = result.lowered_graph.ops[0].attrs
    assert attrs["normalized_shape"] == (6,)
    assert attrs["eps"] == pytest.approx(1e-6)
    assert attrs["has_weight"] is weighted

    actual = result.plan(mx.array(x.numpy()), mx.array(weight.numpy()))[0]
    expected = torch.nn.functional.rms_norm(
        x, (6,), weight if weighted else None, eps=1e-6
    )
    mx.eval(actual)
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-6, rtol=1e-6)


def test_rms_norm_with_weight_and_multidimensional_shape() -> None:
    torch.manual_seed(67)
    x = torch.randn(2, 3, 4, 7)
    weight = torch.randn(4, 7)

    def fn(value, scale):
        return torch.nn.functional.rms_norm(value, (4, 7), scale, eps=1e-6)

    result = compile_callable(fn, (x, weight))
    assert [op.op for op in result.input_graph.ops] == ["rms_norm"]
    attrs = result.input_graph.ops[0].attrs
    assert attrs["normalized_shape"] == (4, 7)
    assert attrs["eps"] == pytest.approx(1e-6)
    assert attrs["has_weight"] is True
    kernel = result.plan.kernels[0]
    assert kernel.name == "metal_rms_norm_32_0"
    assert "local_square_sum" in kernel.source
    assert "mean_square" in kernel.source
    assert "mean_value" not in kernel.source
    assert "in_1[col]" in kernel.source

    actual = result.plan(mx.array(x.numpy()), mx.array(weight.numpy()))[0]
    expected = mx.array(fn(x, weight).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rms_norm_without_weight() -> None:
    torch.manual_seed(71)
    x = torch.randn(2, 3, 300)
    result = compile_callable(
        lambda value: torch.nn.functional.rms_norm(value, (300,), None, eps=1e-5),
        (x,),
    )
    assert result.input_graph.ops[0].attrs["has_weight"] is False
    assert result.plan.kernels[0].name == "metal_rms_norm_256_0"

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.nn.functional.rms_norm(x, (300,), None, eps=1e-5).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_supports_low_precision(dtype) -> None:
    torch.manual_seed(73)
    x = torch.randn(2, 3, 7, dtype=dtype)
    weight = torch.randn(7, dtype=dtype)
    result = compile_callable(
        lambda value, scale: torch.nn.functional.rms_norm(value, (7,), scale, eps=1e-6),
        (x, weight),
    )
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(
        *(mx.array(value.float().numpy(), dtype=mlx_dtype) for value in (x, weight))
    )[0]
    expected = mx.array(
        torch.nn.functional.rms_norm(x.float(), (7,), weight.float(), eps=1e-6).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_rms_norm_composes_with_matmul_and_pointwise() -> None:
    torch.manual_seed(79)
    inputs = (
        torch.randn(2, 4),
        torch.randn(4, 8),
        torch.randn(8),
    )

    def fn(value, weight, scale):
        projected = value @ weight
        normalized = torch.nn.functional.rms_norm(projected, (8,), scale, eps=1e-6)
        return torch.nn.functional.silu(normalized)

    result = compile_callable(fn, inputs)
    assert [op.op for op in result.lowered_graph.ops] == [
        "matmul",
        "rms_norm",
        "silu",
    ]
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "rms_norm",
        "silu",
    ]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-4, rtol=5e-4)


def test_rms_norm_pattern_does_not_match_non_tail_reduction() -> None:
    torch.manual_seed(84)
    x = torch.randn(3, 5)

    def fn(value):
        mean_square = value.square().mean(dim=0, keepdim=True)
        return value * torch.rsqrt(mean_square + 1e-6)

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "pow",
        "mean",
        "fused_elementwise",
    ]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_rms_norm_pattern_requires_single_use_intermediates() -> None:
    torch.manual_seed(85)
    x = torch.randn(2, 5)

    def fn(value):
        squared = value.square()
        normalized = value * torch.rsqrt(squared.mean(dim=-1, keepdim=True) + 1e-6)
        return normalized, squared

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [
        "pow",
        "mean",
        "fused_elementwise",
    ]
    actual = result.plan(mx.array(x.numpy()))
    expected = tuple(mx.array(value.numpy()) for value in fn(x))
    mx.eval(*actual, *expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert mx.allclose(actual_value, expected_value, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("dtype_override", [False, True])
def test_default_reductions_reduce_all_dimensions(reduction, dtype_override) -> None:
    torch.manual_seed(82)
    dtype = torch.float16 if dtype_override else torch.float32
    x = torch.randn(2, 3, 4, dtype=dtype)

    if reduction == "sum":
        if dtype_override:
            fn = lambda value: torch.sum(value, dtype=torch.float32)
        else:
            fn = torch.sum
    else:
        if dtype_override:
            fn = lambda value: torch.mean(value, dtype=torch.float32)
        else:
            fn = torch.mean

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.lowered_graph.ops] == [reduction]
    assert result.lowered_graph.ops[0].attrs == {
        "dims": (0, 1, 2),
        "keepdim": False,
        **({"dtype": DType.F32} if dtype_override else {}),
    }

    mlx_dtype = mx.float16 if dtype_override else mx.float32
    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected = mx.array(fn(x).float().numpy())
    mx.eval(actual, expected)
    assert actual.shape == ()
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_last_dim_supports_keepdim(keepdim) -> None:
    torch.manual_seed(83)
    x = torch.randn(2, 3, 300)

    def fn(value):
        return torch.sum(value, dim=-1, keepdim=keepdim)

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.input_graph.ops] == ["sum"]
    assert result.input_graph.ops[0].attrs == {
        "dims": (-1,),
        "keepdim": keepdim,
    }
    kernel = result.plan.kernels[0]
    assert kernel.name == "metal_sum_256_0"
    assert "threadgroup float scratch[256]" in kernel.source

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mean_reduces_multiple_tail_dimensions() -> None:
    torch.manual_seed(89)
    x = torch.randn(2, 3, 4, 5)

    def fn(value):
        return torch.mean(value, dim=(-2, -1), keepdim=True)

    result = compile_callable(fn, (x,))
    assert [op.op for op in result.input_graph.ops] == ["mean"]
    assert result.plan.kernels[0].name == "metal_mean_32_0"

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (2, 3, 1, 1)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_mean_reduces_contiguous_interior_dimensions() -> None:
    torch.manual_seed(93)
    x = torch.randn(2, 3, 4, 5)

    def fn(value):
        return torch.mean(value, dim=(1, 2), keepdim=True)

    result = compile_callable(fn, (x,))
    kernel = result.plan.kernels[0]
    assert kernel.name == "metal_mean_16_0"
    assert "col * 5u" in kernel.source

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (2, 1, 1, 5)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_sum_reduces_leading_dimension() -> None:
    torch.manual_seed(95)
    x = torch.randn(3, 4, 5)
    result = compile_callable(lambda value: torch.sum(value, dim=0), (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.sum(x, dim=0).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (4, 5)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_sum_can_reduce_all_dimensions() -> None:
    torch.manual_seed(96)
    x = torch.randn(2, 3, 4)
    result = compile_callable(lambda value: torch.sum(value, dim=(0, 1, 2)), (x,))
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.sum(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == ()
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mean_supports_low_precision(dtype) -> None:
    torch.manual_seed(97)
    x = torch.randn(2, 3, 300, dtype=dtype)

    def fn(value):
        return torch.mean(value, dim=-1)

    result = compile_callable(fn, (x,))
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected = mx.array(torch.mean(x.float(), dim=-1).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_mean_composes_with_matmul() -> None:
    torch.manual_seed(101)
    inputs = (torch.randn(2, 4), torch.randn(4, 8))

    def fn(value, weight):
        return torch.mean(value @ weight, dim=-1)

    result = compile_callable(fn, inputs)
    assert [op.op for op in result.input_graph.ops] == ["matmul", "mean"]
    assert [kernel.kind for kernel in result.plan.kernels] == ["mean"]

    mlx_inputs = tuple(mx.array(value.numpy()) for value in inputs)
    actual = result.plan(*mlx_inputs)[0]
    expected = mx.mean(mx.matmul(mlx_inputs[0], mlx_inputs[1]), axis=-1)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_mean_reduces_interleaved_dimensions() -> None:
    torch.manual_seed(103)
    x = torch.randn(2, 3, 4, 5)

    def fn(value):
        return torch.mean(value, dim=(0, 2), keepdim=True)

    result = compile_callable(fn, (x,))
    kernel = result.plan.kernels[0]
    assert kernel.name == "metal_mean_8_0"
    assert "col / 4u" in kernel.source
    assert "col % 4u" in kernel.source

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (1, 3, 1, 5)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_sum_reduces_interleaved_dimensions_without_keepdim() -> None:
    torch.manual_seed(105)
    x = torch.randn(2, 3, 4, 5)

    def fn(value):
        return torch.sum(value, dim=(1, 3))

    result = compile_callable(fn, (x,))
    kernel = result.plan.kernels[0]
    assert kernel.name == "metal_sum_16_0"
    assert "col / 5u" in kernel.source

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (2, 4)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduction_supports_float32_dtype_override(reduction, dtype) -> None:
    torch.manual_seed(107)
    x = torch.randn(2, 3, 300, dtype=dtype)
    reduce = torch.sum if reduction == "sum" else torch.mean

    def fn(value):
        return reduce(value, dim=-1, dtype=torch.float32)

    result = compile_callable(fn, (x,))
    assert result.input_graph.ops[0].attrs == {
        "dims": (-1,),
        "keepdim": False,
        "dtype": DType.F32,
    }
    kernel = result.plan.kernels[0]
    assert kernel.tensor_type.dtype == DType.F32
    assert "out[output_index] = (O)" in kernel.source

    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(mx.array(x.float().numpy(), dtype=mlx_dtype))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.float32
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_reduction_supports_narrower_output_dtype(reduction) -> None:
    torch.manual_seed(109)
    x = torch.randn(2, 3, 17)
    reduce = torch.sum if reduction == "sum" else torch.mean

    def fn(value):
        return reduce(value, dim=-1, dtype=torch.float16)

    result = compile_callable(fn, (x,))
    assert result.plan.kernels[0].tensor_type.dtype == DType.F16

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(fn(x).numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mx.float16
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


# ``eye``, ``ldexp`` and ``addbmm`` are the composites that joined the lowering
# table last: ATen leaves each of them to the backend, and the decomposition
# below is the one Torch's own reference implementation is written in. These
# tests pin both halves - the operators the decomposition emits, and the numbers.

EYE_CASES = [
    pytest.param(4, None, torch.float32, mx.float32, id="square"),
    pytest.param(3, 5, torch.float32, mx.float32, id="rectangular"),
    pytest.param(5, 2, torch.float32, mx.float32, id="tall"),
    pytest.param(2, 2, torch.float16, mx.float16, id="half"),
    pytest.param(1, 4, torch.float32, mx.float32, id="single_row"),
    pytest.param(3, 3, torch.int32, mx.int32, id="integer"),
]


@pytest.mark.parametrize(("rows", "columns", "dtype", "mlx_dtype"), EYE_CASES)
def test_eye_is_the_equality_of_two_ranges(rows, columns, dtype, mlx_dtype) -> None:
    def fn():
        if columns is None:
            return torch.eye(rows, dtype=dtype)
        return torch.eye(rows, columns, dtype=dtype)

    result = compile_callable(fn, ())

    # The equality is the diagonal mask; the cast is the dtype the caller asked
    # for. A square identity needs one range, a rectangular one needs both.
    ops = [op.op for op in result.lowered_graph.ops]
    assert ops[-2:] == ["eq", "cast"]
    assert ops.count("arange") == (1 if columns is None or columns == rows else 2)

    actual = result.plan()[0]
    expected = mx.array(fn().numpy())
    mx.eval(actual, expected)
    assert actual.dtype == mlx_dtype
    assert actual.shape == tuple(expected.shape)
    assert bool(mx.array_equal(actual, expected))


def test_eye_without_a_column_count_reuses_one_range() -> None:
    # ``eye(n)`` is the square case of ``eye(n, m)``, so the second range must
    # not survive as a second kernel when it is the same range.
    result = compile_callable(lambda: torch.eye(6), ())

    assert [op.op for op in result.lowered_graph.ops] == [
        "arange",
        "unsqueeze",
        "eq",
        "cast",
    ]
    assert [kernel.kind for kernel in result.plan.kernels] == ["arange", "eq"]


def test_eye_rejects_an_empty_side() -> None:
    with pytest.raises(ValueError, match="positive"):
        compile_callable(lambda: torch.eye(0), ())


def test_eye_composes_with_a_pointwise_operand() -> None:
    torch.manual_seed(41)
    x = torch.randn(3, 3)

    result = compile_callable(lambda v: torch.eye(3) + v, (x,))

    assert [op.op for op in result.lowered_graph.ops] == [
        "arange",
        "unsqueeze",
        "eq",
        "cast",
        "add",
    ]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array((torch.eye(3) + x).numpy())
    mx.eval(actual, expected)
    assert bool(mx.array_equal(actual, expected))


def test_empty_is_materialized_as_zeros() -> None:
    # Torch leaves the contents of ``empty`` unspecified, so any value satisfies
    # it; the plan picks zeros so a buffer nobody wrote stays deterministic.
    # ``empty_like`` reaches the same lowering even though Torch's own
    # decomposition rewrites it into ``empty_permuted``.
    torch.manual_seed(39)
    x = torch.randn(4, 6)

    literal = compile_callable(lambda v: torch.empty(2, 3), (x,))
    like = compile_callable(lambda v: torch.empty_like(v), (x,))

    assert [op.op for op in literal.lowered_graph.ops] == ["zeros"]
    assert [op.op for op in like.lowered_graph.ops] == ["zeros"]

    filled = literal.plan(mx.array(x.numpy()))[0]
    shaped = like.plan(mx.array(x.numpy()))[0]
    expected = mx.zeros((2, 3))
    mx.eval(filled, shaped)
    assert filled.shape == (2, 3) and shaped.shape == (4, 6)
    assert bool(mx.array_equal(filled, expected))
    assert bool(mx.array_equal(shaped, mx.zeros((4, 6))))


def test_a_filled_empty_buffer_folds_into_the_constant() -> None:
    # The usual spelling is allocate-then-fill, and the fill makes the
    # allocation dead: the whole pair is one constant.
    torch.manual_seed(40)
    x = torch.randn(3, 4)

    result = compile_callable(lambda v: torch.empty_like(v).fill_(2.5), (x,))

    assert [op.op for op in result.lowered_graph.ops] == ["full_like"]
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array((torch.empty_like(x).fill_(2.5)).numpy())
    mx.eval(actual, expected)
    assert bool(mx.array_equal(actual, expected))


@pytest.mark.parametrize(
    ("dtype", "mlx_dtype"),
    [
        pytest.param(torch.float32, mx.float32, id="float32"),
        pytest.param(torch.float16, mx.float16, id="float16"),
    ],
)
def test_ldexp_scales_by_a_power_of_two(dtype, mlx_dtype) -> None:
    torch.manual_seed(42)
    value = torch.randn(3, 4, dtype=dtype)
    exponent = torch.randint(-4, 5, (3, 4))

    def fn(mantissa, power):
        return torch.ldexp(mantissa, power)

    result = compile_callable(fn, (value, exponent))

    # The integer exponent is cast to the result dtype, then the scaling is the
    # operator itself: a separate ``exp2`` and multiply would round twice and
    # break the exactness ``ldexp`` promises.
    assert [op.op for op in result.lowered_graph.ops] == ["cast", "ldexp"]

    actual = result.plan(
        mx.array(value.float().numpy(), dtype=mlx_dtype),
        mx.array(exponent.numpy()),
    )[0]
    expected = mx.array(torch.ldexp(value, exponent).float().numpy(), dtype=mlx_dtype)
    mx.eval(actual, expected)
    assert actual.dtype == mlx_dtype
    # Scaling by a power of two only moves the exponent field, so this is the
    # same bit pattern Torch produces, on every GPU.
    assert bool(mx.array_equal(actual, expected))


def test_ldexp_keeps_a_zero_and_an_unsigned_zero() -> None:
    value = torch.tensor([0.0, -0.0, 1.5])
    exponent = torch.tensor([3, 3, -1])

    result = compile_callable(lambda a, b: torch.ldexp(a, b), (value, exponent))
    actual = result.plan(mx.array(value.numpy()), mx.array(exponent.numpy()))[0]
    expected = mx.array(torch.ldexp(value, exponent).numpy())
    mx.eval(actual, expected)
    assert bool(mx.array_equal(actual, expected))


@pytest.mark.parametrize("codegen_matmul", [False, True])
def test_addbmm_matches_pytorch(codegen_matmul) -> None:
    def fn(bias, left, right):
        return torch.addbmm(bias, left, right, beta=0.5, alpha=2.0)

    torch.manual_seed(43)
    inputs = (
        torch.randn(2, 5),
        torch.randn(3, 2, 4),
        torch.randn(3, 4, 5),
    )

    result = compile_callable(fn, inputs, codegen_matmul=codegen_matmul)

    # The batch product collapses into a single matrix before the scales fold
    # in, so the whole expression is three operators and one kernel.
    assert [op.op for op in result.lowered_graph.ops] == [
        "matmul",
        "sum",
        "fused_elementwise",
    ]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=5e-3)


@pytest.mark.parametrize(
    "bias_shape",
    [
        pytest.param((5,), id="row"),
        pytest.param((1, 5), id="leading_singleton"),
        pytest.param((), id="scalar"),
    ],
)
def test_addbmm_broadcasts_the_bias(bias_shape) -> None:
    torch.manual_seed(44)
    inputs = (
        torch.randn(bias_shape),
        torch.randn(2, 3, 4),
        torch.randn(2, 4, 5),
    )

    def fn(bias, left, right):
        return torch.addbmm(bias, left, right, beta=1.5, alpha=1.0)

    result = compile_callable(fn, inputs)
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (3, 5)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=5e-3)


def test_addbmm_drops_the_bias_at_beta_zero() -> None:
    # ATen documents that a zero beta ignores ``input``, so an infinite one must
    # not turn the result into NaN - which means the add cannot be emitted at
    # all, rather than emitted with a zero scale.
    torch.manual_seed(45)
    bias = torch.tensor([[float("inf"), float("nan"), 1.0, -2.0, 0.5]])
    left = torch.randn(2, 3, 4)
    right = torch.randn(2, 4, 5)

    def fn(input, batch1, batch2):
        return torch.addbmm(input, batch1, batch2, beta=0.0, alpha=2.0)

    result = compile_callable(fn, (bias, left, right))

    assert [op.op for op in result.lowered_graph.ops] == ["matmul", "sum", "mul"]
    actual = result.plan(
        mx.array(bias.numpy()), mx.array(left.numpy()), mx.array(right.numpy())
    )[0]
    expected = mx.array(fn(bias, left, right).numpy())
    mx.eval(actual, expected)
    assert not bool(mx.any(mx.isnan(actual)))
    assert mx.allclose(actual, expected, atol=2e-2, rtol=5e-3)


def test_addbmm_drops_the_products_at_alpha_zero() -> None:
    # The mirror case: a zero alpha ignores ``batch1``/``batch2``, so the NaN
    # they carry must not reach a result that is only ``beta * input``.
    torch.manual_seed(46)
    bias = torch.randn(3, 5)
    left = torch.full((2, 3, 4), float("nan"))
    right = torch.full((2, 4, 5), float("nan"))

    def fn(input, batch1, batch2):
        return torch.addbmm(input, batch1, batch2, beta=1.5, alpha=0.0)

    result = compile_callable(fn, (bias, left, right))

    actual = result.plan(
        mx.array(bias.numpy()), mx.array(left.numpy()), mx.array(right.numpy())
    )[0]
    expected = mx.array(fn(bias, left, right).numpy())
    mx.eval(actual, expected)
    assert not bool(mx.any(mx.isnan(actual)))
    assert bool(mx.array_equal(actual, expected))


def test_addbmm_at_zero_alpha_and_beta_answers_zeros() -> None:
    torch.manual_seed(47)
    inputs = (
        torch.full((3, 5), float("inf")),
        torch.full((2, 3, 4), float("nan")),
        torch.full((2, 4, 5), float("nan")),
    )

    def fn(input, batch1, batch2):
        return torch.addbmm(input, batch1, batch2, beta=0.0, alpha=0.0)

    result = compile_callable(fn, inputs)
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert bool(mx.array_equal(actual, expected))


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        pytest.param(1.0, 1.0, id="plain"),
        pytest.param(2.0, 0.5, id="scaled"),
        pytest.param(-1.5, 3.0, id="mixed_signs"),
    ],
)
def test_addmv_matches_pytorch(alpha, beta) -> None:
    def fn(input, matrix, vector):
        return torch.addmv(input, matrix, vector, beta=beta, alpha=alpha)

    torch.manual_seed(49)
    inputs = (torch.randn(3), torch.randn(3, 4), torch.randn(4))

    result = compile_callable(fn, inputs)

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (3,)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=5e-3)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((3,), id="vector"),
        pytest.param((), id="scalar"),
        pytest.param((1,), id="singleton"),
    ],
)
def test_addmv_broadcasts_the_addend(input_shape) -> None:
    def fn(input, matrix, vector):
        return torch.addmv(input, matrix, vector)

    torch.manual_seed(50)
    inputs = (torch.randn(input_shape), torch.randn(3, 4), torch.randn(4))

    result = compile_callable(fn, inputs)
    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (3,)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=5e-3)


def test_addmv_drops_the_addend_at_beta_zero() -> None:
    # Same rule as ``addbmm``: ATen ignores the addend when beta is zero, so an
    # infinite one must not turn the product into NaN.
    def fn(input, matrix, vector):
        return torch.addmv(input, matrix, vector, beta=0.0, alpha=2.0)

    torch.manual_seed(51)
    inputs = (
        torch.full((3,), float("inf")),
        torch.randn(3, 4),
        torch.randn(4),
    )

    result = compile_callable(fn, inputs)
    assert "add" not in [op.op for op in result.lowered_graph.ops]

    actual = result.plan(*(mx.array(value.numpy()) for value in inputs))[0]
    expected = mx.array(fn(*inputs).numpy())
    mx.eval(actual, expected)
    assert not bool(mx.any(mx.isnan(actual)))
    assert mx.allclose(actual, expected, atol=2e-2, rtol=5e-3)


def test_einsum_full_contraction_answers_a_scalar() -> None:
    # ``einsum("ij,ij->")`` contracts every axis, and Torch spells that as a
    # matrix product followed by a reshape to the rank-0 shape ``()``.
    torch.manual_seed(48)
    left = torch.randn(4, 5)
    right = torch.randn(4, 5)

    result = compile_callable(lambda a, b: torch.einsum("ij,ij->", a, b), (left, right))

    assert [op.op for op in result.lowered_graph.ops] == [
        "reshape",
        "reshape",
        "matmul",
        "reshape",
    ]
    assert result.lowered_graph.ops[-1].attrs == {"shape": ()}

    actual = result.plan(mx.array(left.numpy()), mx.array(right.numpy()))[0]
    expected = mx.array(torch.einsum("ij,ij->", left, right).numpy())
    mx.eval(actual, expected)
    assert actual.shape == ()
    assert mx.allclose(actual, expected, atol=1e-4, rtol=1e-4)
