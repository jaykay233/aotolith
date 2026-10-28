from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from transformer_subgraph_compiler import compile_callable, compile_graph, fuse_pointwise
from transformer_subgraph_compiler.ir import DType, Graph, Operation, TensorType, Value


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
