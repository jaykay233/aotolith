"""Int4 weights through the compiler: the fold, the kernel, and the fallback.

There are two compiled shapes and they take different paths, so each gets its
own numeric check. A single-row matmul - the decode case, one token against a
whole weight matrix - is folded into the matmul, which then streams the packed
words and dequantizes them in registers, so no unpacked weight ever exists in
memory. A wider matmul - prefill - keeps the explicit unpack and runs it as one
streaming kernel, because with many rows the weight stream is the cheap part of
the multiply and the int4 decode would only add work.

The reference is Torch: the same ``matmul(x, dequantize_i4(...))`` evaluated
eagerly against the same packed words. Checking against the *original* weights
would only measure quantization error, which the format tests already cover.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from aotolith import compile_callable
from aotolith.backends.apple import AppleMetalBackend
from aotolith.capture import capture_fx
from aotolith.models.quantized import dequantize_i4, pack_weight
from aotolith.passes import rewrite
from aotolith.quantize import dequantize_int4, quantize_int4


GROUP = 32


def _case(rows: int, inner: int, batch: int, seed: int):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((rows, inner)).astype(np.float32)
    activation = rng.standard_normal((batch, inner)).astype(np.float32)
    packed, scales = quantize_int4(weight, GROUP)
    return weight, activation, packed, scales


def _torch_reference(activation, packed, scales):
    return torch.matmul(
        torch.from_numpy(activation),
        dequantize_i4(torch.from_numpy(packed), torch.from_numpy(scales), GROUP).t(),
    ).numpy()


def _compile(rows: int, inner: int, batch: int, seed: int):
    _, activation, packed, scales = _case(rows, inner, batch, seed)
    weight = torch.from_numpy(packed)
    scale = torch.from_numpy(scales)

    def call(activation, packed, scales):
        return torch.matmul(activation, dequantize_i4(packed, scales, GROUP).t())

    compiled = compile_callable(call, (torch.from_numpy(activation), weight, scale))
    actual = compiled.plan(mx.array(activation), mx.array(packed), mx.array(scales))
    mx.eval(actual)
    return np.asarray(actual), _torch_reference(activation, packed, scales)


@pytest.mark.parametrize("rows,inner", [(64, 128), (32, 256), (128, 64)])
def test_decode_matmul_matches_torch(rows: int, inner: int) -> None:
    actual, expected = _compile(rows, inner, 1, seed=rows + inner)
    scale = float(np.abs(expected).max()) or 1.0
    assert float(np.abs(actual - expected).max()) <= 3e-2 * scale


@pytest.mark.parametrize("batch", [2, 4, 7])
def test_prefill_matmul_matches_torch(batch: int) -> None:
    actual, expected = _compile(48, 128, batch, seed=batch)
    scale = float(np.abs(expected).max()) or 1.0
    assert float(np.abs(actual - expected).max()) <= 3e-2 * scale


def test_a_single_row_matmul_keeps_no_unpacked_weight() -> None:
    # The fold exists so that the weight never materializes; if the graph still
    # had an unpack node, the dequantized tensor would be the biggest live
    # allocation in the plan and the memory argument for int4 would be gone.
    def call(x, packed, scales):
        return torch.matmul(x, dequantize_i4(packed, scales, GROUP).t())

    weight = torch.from_numpy(_case(64, 128, 1, seed=3)[2])
    scale = torch.from_numpy(_case(64, 128, 1, seed=3)[3])
    graph = rewrite(capture_fx(call, (torch.zeros(1, 128), weight, scale)))
    assert [op.op for op in graph.ops if op.op == "dequantize_i4"] == []


def test_a_wide_matmul_keeps_the_unpack() -> None:
    def call(x, packed, scales):
        return torch.matmul(x, dequantize_i4(packed, scales, GROUP).t())

    weight = torch.from_numpy(_case(64, 128, 4, seed=3)[2])
    scale = torch.from_numpy(_case(64, 128, 4, seed=3)[3])
    graph = rewrite(capture_fx(call, (torch.zeros(4, 128), weight, scale)))
    assert [op.op for op in graph.ops].count("dequantize_i4") == 1


def test_unpacked_fallback_agrees_with_the_native_kernel() -> None:
    # The MLX unpack and the Metal kernel decode the same words; a mismatch here
    # would mean the fallback path and the fast path disagree about the format.
    _, activation, packed, scales = _case(32, 64, 1, seed=11)
    reference = dequantize_int4(packed, scales, GROUP).T
    actual = _compile(32, 64, 1, seed=11)[1]
    expected = np.matmul(activation, reference)
    assert np.allclose(actual, expected, atol=1e-4 * (np.abs(expected).max() + 1e-6))


def test_group_size_survives_into_the_folded_matmul() -> None:
    def call(x, packed, scales):
        return torch.matmul(x, dequantize_i4(packed, scales, GROUP).t())

    packed = torch.from_numpy(_case(16, 64, 1, seed=5)[2])
    scales = torch.from_numpy(_case(16, 64, 1, seed=5)[3])
    graph = rewrite(capture_fx(call, (torch.zeros(1, 64), packed, scales)))
    matmuls = [op for op in graph.ops if op.op == "matmul"]
    assert len(matmuls) == 1
    # The kernel reads the group size off the op to pick a scale per word, so it
    # must travel with the fold rather than being inferred from shapes.
    assert matmuls[0].attrs["weight_group_size"] == GROUP


def test_pack_weight_is_the_inverse_the_tests_check_against() -> None:
    rng = np.random.default_rng(17)
    weight = torch.from_numpy(rng.standard_normal((8, 64)).astype(np.float32))
    packed, scales = pack_weight(weight, GROUP)
    np.testing.assert_allclose(
        dequantize_i4(packed, scales, GROUP).numpy(),
        dequantize_int4(packed.numpy(), scales.numpy(), GROUP),
        atol=1e-6,
    )


def _lowered(batch: int):
    def call(x, packed, scales):
        return torch.matmul(x, dequantize_i4(packed, scales, GROUP).t())

    _, activation, packed, scales = _case(32, 64, batch, seed=batch)
    graph = rewrite(
        capture_fx(
            call,
            (
                torch.zeros(batch, 64),
                torch.from_numpy(packed),
                torch.from_numpy(scales),
            ),
        )
    )
    return AppleMetalBackend().compile(graph)


def test_the_decode_matmul_runs_natively_without_dense_codegen() -> None:
    # Dense matmul codegen is off by default because MLX's own kernel wins. A
    # packed weight is the exception: falling back means materializing the dense
    # weight, which is exactly the allocation int4 removes, so the streaming
    # kernel has to be chosen even with codegen off.
    result = _lowered(1)
    selections = {selection.output_id: selection for selection in result.selections}
    assert len(selections) == 1
    assert all(selection.is_native for selection in selections.values())


def test_the_prefill_unpack_runs_as_one_kernel() -> None:
    result = _lowered(4)
    unpack = [
        selection for selection in result.selections if selection.output_id == "v0"
    ]
    assert unpack and unpack[0].is_native


def test_a_graph_that_returns_the_unpacked_matrix_keeps_the_unpack() -> None:
    # Dropping the unpack would leave a graph output with no producer, so the
    # fold has to decline even though the matmul has a single-row left operand.
    def call(x, packed, scales):
        weight = dequantize_i4(packed, scales, GROUP)
        return torch.matmul(x, weight.t()), weight

    _, _, packed, scales = _case(32, 64, 1, seed=13)
    graph = rewrite(
        capture_fx(
            call,
            (torch.zeros(1, 64), torch.from_numpy(packed), torch.from_numpy(scales)),
        )
    )
    assert [op.op for op in graph.ops].count("dequantize_i4") == 1


def test_a_residual_folds_into_one_packed_addmm() -> None:
    # ``linear(x, w) + residual`` becomes a single addmm, and with a packed
    # weight that addmm carries four operands: residual, activation, words and
    # scales. The arity and the epilogue fusion both have to allow it.
    def call(x, packed, scales, residual):
        return residual + torch.matmul(x, dequantize_i4(packed, scales, GROUP).t())

    _, activation, packed, scales = _case(32, 64, 1, seed=19)
    residual = np.random.default_rng(19).standard_normal((1, 32)).astype(np.float32)
    graph = rewrite(
        capture_fx(
            call,
            (
                torch.from_numpy(activation),
                torch.from_numpy(packed),
                torch.from_numpy(scales),
                torch.from_numpy(residual),
            ),
        )
    )
    addmms = [op for op in graph.ops if op.op == "addmm"]
    assert len(addmms) == 1
    assert len(addmms[0].inputs) == 4
    assert addmms[0].attrs["weight_group_size"] == GROUP

    expected = residual + np.matmul(
        activation, dequantize_int4(packed, scales, GROUP).T
    )
    compiled = compile_callable(
        call,
        (
            torch.from_numpy(activation),
            torch.from_numpy(packed),
            torch.from_numpy(scales),
            torch.from_numpy(residual),
        ),
    )
    actual = compiled.plan(
        mx.array(activation),
        mx.array(packed),
        mx.array(scales),
        mx.array(residual),
    )
    mx.eval(actual)
    scale = float(np.abs(expected).max())
    assert float(np.abs(np.asarray(actual) - expected).max()) <= 3e-2 * scale
