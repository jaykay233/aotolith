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


def test_metal_matmul_uses_register_microtile() -> None:
    torch.manual_seed(23)
    inputs = (torch.randn(11, 7), torch.randn(7, 13))
    result = compile_callable(lambda x, weight: x @ weight, inputs, codegen_matmul=True)
    kernel = result.plan.kernels[0]

    assert kernel.name == "metal_matmul_4x4_0"
    assert "threadgroup float" not in kernel.source
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

    assert kernel.name == "metal_matmul_8x4_0"
    assert "acc_7_3" in kernel.source
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
    inputs = (torch.randn(8, 8), torch.randn(8, 8))
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


def test_metal_softmax_rejects_non_last_dimension() -> None:
    x = torch.randn(2, 3, 4)
    with pytest.raises(NotImplementedError, match="last dimension"):
        compile_callable(lambda value: torch.nn.functional.softmax(value, dim=1), (x,))


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
    x = torch.randn(3, 5)

    def fn(value):
        mean_square = value.square().mean(dim=0, keepdim=True)
        return value * torch.rsqrt(mean_square + 1e-6)

    with pytest.raises(
        ValueError, match="unsupported FX target: aten.pow.Tensor_Scalar"
    ):
        compile_callable(fn, (x,))


def test_rms_norm_pattern_requires_single_use_intermediates() -> None:
    x = torch.randn(2, 5)

    def fn(value):
        squared = value.square()
        normalized = value * torch.rsqrt(
            squared.mean(dim=-1, keepdim=True) + 1e-6
        )
        return normalized, squared

    with pytest.raises(
        ValueError, match="unsupported FX target: aten.pow.Tensor_Scalar"
    ):
        compile_callable(fn, (x,))
