from __future__ import annotations

import math

import mlx.core as mx
import pytest
import torch

from transformer_subgraph_compiler import (
    compile_callable,
    compile_graph,
    eliminate_common_subexpressions,
    fuse_pointwise,
)
from transformer_subgraph_compiler.backends import apple_metal as metal_backend
from transformer_subgraph_compiler.ir import DType, Graph, Operation, TensorType, Value


_MLX_DTYPES = {
    torch.float16: mx.float16,
    torch.bfloat16: mx.bfloat16,
    torch.float32: mx.float32,
    torch.bool: mx.bool_,
    torch.int32: mx.int32,
    torch.int64: mx.int64,
}


def _torch_to_mlx(value: torch.Tensor) -> mx.array:
    source = value.float().numpy() if value.is_floating_point() else value.numpy()
    return mx.array(source).astype(_MLX_DTYPES[value.dtype])


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
    metal_backend._metal_kernel.cache_clear()
    x = torch.randn(2, 8)
    first = compile_callable(lambda value: torch.relu(value), (x,))
    second = compile_callable(lambda value: torch.relu(value), (x,))

    cache_info = metal_backend._metal_kernel.cache_info()
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
        "fused_elementwise",
        "matmul",
    ]

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
    assert [op.op for op in result.lowered_graph.ops] == [
        "matmul",
        "matmul",
        "fused_elementwise",
    ]

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


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
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
    assert [
        op.op
        for op in result.input_graph.ops
        if op.op.startswith("bitwise_")
    ] == ["bitwise_and", "bitwise_or", "bitwise_xor"]
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


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
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
    projected = mx.matmul(
        mx.array(inputs[0].numpy()), mx.array(inputs[1].numpy())
    )
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
        return (
            x.reshape(2, 3)
            .transpose(0, 1)
            .transpose(0, 1)
            .reshape(3, 2)
            .clone()
        )

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


@pytest.mark.parametrize(
    ("dim", "index_dtype"), [(1, torch.int64), (-1, torch.int32)]
)
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

    actual = result.plan(
        source_mlx, _torch_to_mlx(indices), _torch_to_mlx(update)
    )[0]
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
            _torch_to_mlx(
                torch.tensor([invalid, [0, 1]], dtype=torch.int64)
            ),
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
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
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
    result = compile_callable(
        fn, inputs, codegen_attention=codegen_attention
    )

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
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
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
    result = compile_callable(
        fn, inputs, codegen_attention=codegen_attention
    )
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
    result = compile_callable(
        fn, inputs, codegen_attention=codegen_attention
    )

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
    result = compile_callable(
        fn, inputs, codegen_attention=codegen_attention
    )

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
        torch.randn(2, 2, 5, 64, dtype=dtype),
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
def test_native_attention_reuses_query_tile_across_simd_groups(
    dtype, tolerance
) -> None:
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

    assert "metal_attention_flash_16x32_512" in kernel.name
    assert "threadgroup T tile_q[1024]" in source
    assert "threadgroup float tile_scores[512]" in source
    assert "threadgroup T tile_probabilities_low[512]" in source
    assert "if (warp < 8u)" in source
    assert "uint query_fragment = warp / 4u" in source
    assert "uint key_fragment = warp % 4u" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(fn(*inputs))
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float16, 5e-3), (torch.bfloat16, 2e-2)],
)
def test_native_attention_uses_long_sequence_query_tile(dtype, tolerance) -> None:
    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )

    torch.manual_seed(67)
    inputs = tuple(
        torch.randn(1, 1, 769, 128, dtype=dtype) for _ in range(3)
    )
    result = compile_callable(fn, inputs, codegen_attention=True)
    kernel = result.plan.kernels[0]
    source = kernel.source

    assert "metal_attention_flash_24x64_768" in kernel.name
    assert "threadgroup T tile_q[3072]" in source
    assert "alignas(32) threadgroup T tile_q[3072]" in source
    assert "threadgroup float tile_workspace[2304]" in source
    assert "threadgroup T* tile_probabilities_low" in source
    assert "float score1 = -INFINITY" in source
    assert "value_chunk_base" in source
    assert "if (warp < 24u)" in source
    assert "vec<T, 16>" in source

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
        "query": Value("query", TensorType((2, 5, 16), dtype)),
        "key": Value("key", TensorType((2, 7, 16), dtype)),
        "value": Value("value", TensorType((2, 7, 16), dtype)),
        "mask": Value("mask", TensorType((5, 7), DType.BOOL)),
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
    assert "tile_probabilities_low" in result.plan.kernels[0].source

    torch.manual_seed(76)
    mask = torch.rand(5, 7) > 0.25
    mask[1, :] = False
    inputs = (
        torch.randn(2, 5, 16, dtype=torch.float16),
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
        key = key[:, :, None].expand(
            batch, key_value_heads, groups, sequence, head_dim
        ).reshape(batch, query_heads, sequence, head_dim)
        value = value[:, :, None].expand(
            batch, key_value_heads, groups, sequence, head_dim
        ).reshape(batch, query_heads, sequence, head_dim)
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
    query = mx.reshape(mx.transpose(query, (0, 2, 1, 3)), (batch * heads, sequence, head_dim))
    key = mx.reshape(mx.transpose(key, (0, 2, 1, 3)), (batch * heads, sequence, head_dim))
    value = mx.reshape(mx.transpose(value, (0, 2, 1, 3)), (batch * heads, sequence, head_dim))
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
        return torch.nn.functional.rms_norm(
            value, (hidden_size,), weight, eps=1e-6
        )

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
        key = key.reshape(
            batch_size, sequence_length, num_heads, head_dim
        ).transpose(1, 2)
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

    frequencies = torch.arange(
        sequence_length, dtype=torch.float32
    )[:, None] / torch.pow(
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
    assert lowered_ops.count("matmul") == 11
    assert lowered_ops.count("fused_elementwise") == 6
    assert "embedding" in lowered_ops
    assert "attention" in {kernel.kind for kernel in result.plan.kernels}
    assert "matmul" in {kernel.kind for kernel in result.plan.kernels}

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
    assert "cat" in lowered_ops
    assert "fused_elementwise" in lowered_ops
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "neg",
        "fused_elementwise",
    ]

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
        normalized = torch.nn.functional.rms_norm(
            x, (hidden,), norm_weight, eps=1e-6
        )
        query = (normalized @ query_weight).reshape(
            batch, sequence, heads, head_dim
        ).transpose(1, 2)
        key = (normalized @ key_weight).reshape(
            batch, sequence, heads, head_dim
        ).transpose(1, 2)
        value = (normalized @ value_weight).reshape(
            batch, sequence, heads, head_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (
            head_dim**-0.5
        ) + causal_mask
        probabilities = torch.softmax(scores, dim=-1)
        attention = torch.matmul(probabilities, value)
        attention = attention.transpose(1, 2).reshape(batch, sequence, hidden)
        x = x + attention @ output_weight

        normalized = torch.nn.functional.rms_norm(
            x, (hidden,), norm_weight, eps=1e-6
        )
        mlp = torch.nn.functional.silu(normalized @ gate_weight) * (
            normalized @ up_weight
        )
        x = x + mlp @ down_weight
        return torch.nn.functional.rms_norm(
            x, (hidden,), norm_weight, eps=1e-6
        )

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
    assert {"matmul", "rms_norm", "fused_elementwise"} <= set(lowered_ops)

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
        return value * mx.rsqrt(
            mx.mean(value * value, axis=-1, keepdims=True) + 1e-6
        ) * norm_weight

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
        mx.matmul(query, mx.swapaxes(key, -1, -2)) * (head_dim**-0.5)
        + causal_mask,
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
            lambda value: torch.nn.functional.softplus(
                value, beta=2.0, threshold=10.0
            ),
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


@pytest.mark.parametrize("keyword_arguments", [False, True], ids=["positional", "keyword"])
def test_threshold_matches_pytorch_arguments_and_boundaries(
    keyword_arguments,
) -> None:
    def fn(value):
        if keyword_arguments:
            return torch.ops.aten.threshold.default(
                value, threshold=0.25, value=-2.0
            )
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
        _numerical_pointwise_chain(
            *(value.float() for value in inputs)
        ).numpy(),
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
        "fused_elementwise",
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
        fn = lambda value, lower, upper: torch.clamp(
            value, min=lower, max=upper
        )
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
    result = compile_callable(
        lambda value: torch.clamp(value, min=-0.5, max=0.5), (x,)
    )
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
        assert math.copysign(1.0, actual_value) == math.copysign(
            1.0, expected_value
        )
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
    result = compile_callable(
        lambda: torch.scalar_tensor(value, dtype=torch.int64), ()
    )

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
    assert [kernel.kind for kernel in result.plan.kernels] == [
        "fused_elementwise"
    ]

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
def test_integer_floor_divide_and_remainder_match_pytorch(
    dtype, divisor: int
) -> None:
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
    actual = metal_backend._evaluate_pointwise_mlx(
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


def test_arange_rejects_floating_point_outputs() -> None:
    with pytest.raises(NotImplementedError, match="int32 and int64"):
        compile_callable(lambda: torch.arange(8, dtype=torch.float32), ())


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
def test_metal_matmul_uses_simdgroup_for_complete_blocks(
    dtype, simdgroup_type
) -> None:
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
def test_metal_matmul_simdgroup_vectorizes_complete_tiles(dtype) -> None:
    torch.manual_seed(33)
    inputs = (
        torch.randn(64, 64, dtype=dtype),
        torch.randn(64, 64, dtype=dtype),
    )
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    source = result.plan.kernels[0].source

    assert "alignas(16) threadgroup T tile_a[2048]" in source
    assert "alignas(16) threadgroup T tile_b[2048]" in source
    assert "vec<T, 8>" in source

    actual = result.plan(*(_torch_to_mlx(value) for value in inputs))[0]
    expected = _torch_to_mlx(inputs[0] @ inputs[1])
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=5e-2, rtol=5e-2)


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


def test_metal_batched_simdgroup_matmul_vectorizes_batch_inputs() -> None:
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

    assert kernel.name == "metal_matmul_simdgroup_64x64x32_batch6_0"
    assert "vec<T, 8>" in kernel.source
    assert "in_0 + batch * 2048u" in kernel.source
    assert "in_1 + batch * 2048u" in kernel.source
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
        return left.transpose(-2, -1).expand(2, 3, 7, 5) @ right.expand(
            2, 3, 5, 4
        )

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


def test_metal_simdgroup_matmul_broadcasts_batch_inputs() -> None:
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
        "metal_matmul_simdgroup_64x64x32_batch6_0"
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
    source = result.plan.kernels[0].source
    assert "(col_base + local_column) * 64u + k_base + local_k" in source
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


def test_simdgroup_addmm_applies_broadcast_bias() -> None:
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
    assert kernel.name == "metal_addmm_simdgroup_64x64x32_0"

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
    expected = mx.array(
        torch.ops.aten._safe_softmax.default(x, -1).numpy()
    )
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
    expected = mx.array(
        torch.nn.functional.softmax(x, dim=dim).numpy()
    )
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
        return torch.nn.functional.layer_norm(
            value, (300,), scale, shift, eps=1e-5
        )

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
    expected = mx.array(
        torch.nn.functional.layer_norm(x, (3, 4), eps=1e-3).numpy()
    )
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
        torch.nn.functional.layer_norm(x.float(), (7,), weight.float(), bias.float()).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_layer_norm_rejects_auxiliary_outputs() -> None:
    x = torch.randn(2, 3, 7)

    def fn(value):
        _, mean, _ = torch.native_layer_norm(value, (7,), None, None, 1e-5)
        return mean

    with pytest.raises(NotImplementedError, match="mean/rstd"):
        compile_callable(fn, (x,))


def test_layer_norm_composes_with_matmul_and_pointwise() -> None:
    torch.manual_seed(61)
    inputs = (torch.randn(2, 4), torch.randn(4, 8), torch.randn(8), torch.randn(8))

    def fn(value, weight, scale, shift):
        projected = value @ weight
        normalized = torch.nn.functional.layer_norm(
            projected, (8,), scale, shift
        )
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


def test_rms_norm_with_weight_and_multidimensional_shape() -> None:
    torch.manual_seed(67)
    x = torch.randn(2, 3, 4, 7)
    weight = torch.randn(4, 7)

    def fn(value, scale):
        return torch.nn.functional.rms_norm(
            value, (4, 7), scale, eps=1e-6
        )

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
        lambda value: torch.nn.functional.rms_norm(
            value, (300,), None, eps=1e-5
        ),
        (x,),
    )
    assert result.input_graph.ops[0].attrs["has_weight"] is False
    assert result.plan.kernels[0].name == "metal_rms_norm_256_0"

    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(
        torch.nn.functional.rms_norm(x, (300,), None, eps=1e-5).numpy()
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_supports_low_precision(dtype) -> None:
    torch.manual_seed(73)
    x = torch.randn(2, 3, 7, dtype=dtype)
    weight = torch.randn(7, dtype=dtype)
    result = compile_callable(
        lambda value, scale: torch.nn.functional.rms_norm(
            value, (7,), scale, eps=1e-6
        ),
        (x, weight),
    )
    mlx_dtype = mx.bfloat16 if dtype == torch.bfloat16 else mx.float16
    actual = result.plan(
        *(
            mx.array(value.float().numpy(), dtype=mlx_dtype)
            for value in (x, weight)
        )
    )[0]
    expected = mx.array(
        torch.nn.functional.rms_norm(
            x.float(), (7,), weight.float(), eps=1e-6
        ).numpy()
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
        normalized = torch.nn.functional.rms_norm(
            projected, (8,), scale, eps=1e-6
        )
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
        normalized = value * torch.rsqrt(
            squared.mean(dim=-1, keepdim=True) + 1e-6
        )
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
def test_default_reductions_reduce_all_dimensions(
    reduction, dtype_override
) -> None:
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
    actual = result.plan(
        mx.array(x.float().numpy(), dtype=mlx_dtype)
    )[0]
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
    result = compile_callable(
        lambda value: torch.sum(value, dim=0), (x,)
    )
    actual = result.plan(mx.array(x.numpy()))[0]
    expected = mx.array(torch.sum(x, dim=0).numpy())
    mx.eval(actual, expected)
    assert actual.shape == (4, 5)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_sum_can_reduce_all_dimensions() -> None:
    torch.manual_seed(96)
    x = torch.randn(2, 3, 4)
    result = compile_callable(
        lambda value: torch.sum(value, dim=(0, 1, 2)), (x,)
    )
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
    actual = result.plan(
        mx.array(x.float().numpy(), dtype=mlx_dtype)
    )[0]
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
    actual = result.plan(
        mx.array(x.float().numpy(), dtype=mlx_dtype)
    )[0]
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
