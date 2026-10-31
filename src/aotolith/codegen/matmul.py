"""Matmul kernels: three shapes of the same operation.

Which one to emit depends on the product shape and on what the device supports:
a matvec becomes one threadgroup per row with a strided dot product, a small
tile becomes a simdgroup matrix multiply, and a large one becomes a tensor-core
tile with a ladded of candidate tile sizes. `_select_matmul_microtile` and
`_supports_*` decide, the `_emit_*` functions write, and the epilogue helpers
apply the addmm bias or the fused norm on the way out.

The tile ladders themselves live in ``target.apple`` because they are calibrated
against that device's threadgroup memory.
"""

from __future__ import annotations

import mlx.core as mx
from math import ceil, prod
from typing import Any
from ..ir import DType
from ..ir import Graph
from ..ir import Operation
from ..ir import TensorType
from ..shapes import can_broadcast as _can_broadcast
from .dtypes import _MLX_DTYPES
from .expr import _offset_expression
from . import msl
from .msl import KernelArtifact, _MPP_HEADER
from .norm import _emit_thread_rms_norm_prologue, _emit_warp_rms_norm_prologue
from ..target import apple as target
from ..target.apple import _select_mpp_gemm_config, _supports_mpp_matmul
from ..ir.ops.kernels import _matmul_weight_group_size

_MATMUL_MICROTILE = (4, 4)


_MATMUL_LARGE_K_MICROTILE = (8, 4)


_MATMUL_LARGE_K_THRESHOLD = 1024


_MATMUL_BLOCK_ROWS = 64


_MATMUL_BLOCK_COLUMNS = 64


_MATMUL_BLOCK_REDUCTION = 32


_MATMUL_GEMV_THREADS = 256


_MATMUL_GEMV_NARROW_COLUMNS = 8192


_MATMUL_GEMV_THREADS_NARROW = 1024


_MATMUL_GEMV_THREADS_WIDE = 128


#: Four-bit weights come eight to an int32 word, which is the load width the
#: packed GEMV walks the reduction in.
_QUANTIZED_NIBBLES = 8


def _gemv_threads(columns: int, transpose_right: bool) -> int:
    """Pick the GEMV threadgroup size for one output width.

    The transposed path runs one warp per output column, so the block size
    trades block count against per-block streaming width and the best value
    flips once the output is wide enough to fill the GPU on its own.
    """

    if transpose_right:
        if columns <= _MATMUL_GEMV_NARROW_COLUMNS:
            return _MATMUL_GEMV_THREADS_NARROW
        return _MATMUL_GEMV_THREADS_WIDE
    return _MATMUL_GEMV_THREADS


def _select_matmul_microtile(reduction: int) -> tuple[int, int]:
    if reduction >= _MATMUL_LARGE_K_THRESHOLD:
        return _MATMUL_LARGE_K_MICROTILE
    return _MATMUL_MICROTILE


def _supports_simdgroup_matmul(rows: int, columns: int, dtype: DType) -> bool:
    return (
        dtype in {DType.F16, DType.BF16}
        and rows % _MATMUL_BLOCK_ROWS == 0
        and columns % _MATMUL_BLOCK_COLUMNS == 0
    )


def _supports_gemv_matmul(
    rows: int,
    dtype: DType,
    transpose_left: bool,
    transpose_right: bool,
) -> bool:
    """Return whether the one-row matmul can use the bandwidth-oriented GEMV."""

    return (
        rows == 1 and dtype in {DType.F16, DType.BF16, DType.F32} and not transpose_left
    )


def _emit_gemv_matmul_source(
    op: Operation,
    reduction: int,
    columns: int,
    graph: Graph,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
    transpose_right: bool,
) -> str:
    """Emit a GEMV with coalesced weight loads for either right-hand layout."""

    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    threads = _gemv_threads(columns, False)
    block_columns = threads
    exact_columns = columns % block_columns == 0
    left_batch_prefix = f"{left_batch_offset} + " if left_batch_offset else ""
    right_batch_prefix = f"{right_batch_offset} + " if right_batch_offset else ""
    if transpose_right:
        return _emit_transposed_gemv_matmul_source(
            op,
            graph,
            reduction,
            columns,
            bias_id,
            bias_type,
            output_shape,
            alpha,
            beta,
            left_position,
            right_position,
            batch_count,
            left_batch_prefix,
            right_batch_prefix,
        )

    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    norm_slot = _matmul_norm_position(op)
    if norm_slot is not None:
        lines.extend(
            _emit_thread_rms_norm_prologue(
                op, norm_slot, left_position, reduction, left_batch_prefix
            )
        )
    lines.extend(
        [
            f"uint column = threadgroup_index * {block_columns}u + tid;",
            "float acc0 = 0.0f;",
            "float acc1 = 0.0f;",
            "float acc2 = 0.0f;",
            "float acc3 = 0.0f;",
            "uint k = 0u;",
            f"for (; k + 3u < {reduction}u; k += 4u) {{",
        ]
    )
    if norm_slot is None:
        lines.extend(
            [
                f"    float x0 = float(in_{left_position}[{left_batch_prefix}k]);",
                f"    float x1 = float(in_{left_position}[{left_batch_prefix}k + 1u]);",
                f"    float x2 = float(in_{left_position}[{left_batch_prefix}k + 2u]);",
                f"    float x3 = float(in_{left_position}[{left_batch_prefix}k + 3u]);",
            ]
        )
    else:
        for offset in range(4):
            lines.append(
                f"    float x{offset} = float((T)(norm_scale * "
                f"float(in_{left_position}[{left_batch_prefix}k + {offset}u]) * "
                f"float(in_{norm_slot}[k + {offset}u])));"
            )
    if not exact_columns:
        lines.append(f"    if (column < {columns}u) {{")
    indent = "        " if not exact_columns else "    "
    lines.extend(
        [
            f"{indent}acc0 = fma(x0, float(in_{right_position}["
            f"{right_batch_prefix}k * {columns}u + column]), acc0);",
            f"{indent}acc1 = fma(x1, float(in_{right_position}["
            f"{right_batch_prefix}(k + 1u) * {columns}u + column]), acc1);",
            f"{indent}acc2 = fma(x2, float(in_{right_position}["
            f"{right_batch_prefix}(k + 2u) * {columns}u + column]), acc2);",
            f"{indent}acc3 = fma(x3, float(in_{right_position}["
            f"{right_batch_prefix}(k + 3u) * {columns}u + column]), acc3);",
        ]
    )
    if not exact_columns:
        lines.append("    }")
    lines.extend(
        [
            "}",
            f"for (; k < {reduction}u; ++k) {{",
        ]
    )
    if norm_slot is None:
        lines.append(f"    float x = float(in_{left_position}[{left_batch_prefix}k]);")
    else:
        lines.append(
            f"    float x = float((T)(norm_scale * "
            f"float(in_{left_position}[{left_batch_prefix}k]) * "
            f"float(in_{norm_slot}[k])));"
        )
    if not exact_columns:
        lines.append(f"    if (column < {columns}u) {{")
    indent = "        " if not exact_columns else "    "
    lines.append(
        f"{indent}acc0 = fma(x, float(in_{right_position}["
        f"{right_batch_prefix}k * {columns}u + column]), acc0);"
    )
    if not exact_columns:
        lines.append("    }")
    lines.append("}")
    if not exact_columns:
        lines.append(f"if (column < {columns}u) {{")
    indent = "    " if not exact_columns else ""
    output_batch_offset = f"batch * {columns}u + " if batch_count > 1 else ""
    output_linear = f"{output_batch_offset}column"
    value = "((acc0 + acc1) + (acc2 + acc3))"
    if bias_id is not None:
        assert bias_type is not None
        bias_offset = _offset_expression(
            bias_type.shape,
            output_shape,
            output_linear,
        )
        value = f"{beta!r} * float(in_0[{bias_offset}]) + {alpha!r} * {value}"
    value = _matmul_epilogue_value(value, op, graph, output_shape, output_linear)
    lines.append(f"{indent}out[{output_linear}] = (T)({value});")
    if not exact_columns:
        lines.append("}")
    return "\n".join(f"        {line}" for line in lines)


def _emit_transposed_gemv_matmul_source(
    op: Operation,
    graph: Graph,
    reduction: int,
    columns: int,
    bias_id: str | None,
    bias_type: TensorType | None,
    output_shape: tuple[int, ...],
    alpha: float,
    beta: float,
    left_position: int,
    right_position: int,
    batch_count: int,
    left_batch_prefix: str,
    right_batch_prefix: str,
) -> str:
    """Emit warp-per-column GEMV for a physical ``[columns, reduction]`` weight."""

    threads = _gemv_threads(columns, True)
    warps = threads // 32
    exact_columns = columns % warps == 0
    output_batch_offset = f"batch * {columns}u + " if batch_count > 1 else ""
    output_linear = f"{output_batch_offset}column"
    paired_position = _matmul_paired_weight_position(op, right_position)
    if paired_position is not None and reduction % 4 != 0:
        raise ValueError(
            "the paired gate/up epilogue needs a reduction that is a multiple of 4"
        )
    value = "total"
    if bias_id is not None:
        assert bias_type is not None
        bias_offset = _offset_expression(
            bias_type.shape,
            output_shape,
            output_linear,
        )
        value = f"{beta!r} * float(in_0[{bias_offset}]) + {alpha!r} * total"
    if paired_position is None:
        value = _matmul_epilogue_value(value, op, graph, output_shape, output_linear)
    else:
        # ``silu`` of the gate row scaled by the paired up row, both accumulated
        # from the same activation in one pass over the two weight rows. The gate
        # is rounded to the stored element type first, which is exactly the value
        # the separate projection would have written, so the result is unchanged.
        value = "(total_up) * (paired_gate / (1.0f + metal::exp(-paired_gate)))"

    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint lane = tid % 32u;",
        "uint warp = tid / 32u;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    lines.extend(
        [
            f"uint column = threadgroup_index * {warps}u + warp;",
        ]
    )
    norm_slot = _matmul_norm_position(op)
    if norm_slot is not None:
        lines.extend(
            _emit_warp_rms_norm_prologue(
                op, norm_slot, left_position, reduction, left_batch_prefix
            )
        )
    if not exact_columns:
        lines.append(f"if (column < {columns}u) {{")
    indent = "    " if not exact_columns else ""
    # Vectorized fast path: one lane owns four consecutive weights of its row,
    # so a warp moves 256 bytes per load instruction instead of 64.  Pure
    # scalar loads cap the streaming rate well below what the GPU can sustain.
    vectorized = reduction % 4 == 0
    lines.extend(
        [
            f"{indent}float acc0 = 0.0f;",
            f"{indent}float acc1 = 0.0f;",
            f"{indent}float acc2 = 0.0f;",
            f"{indent}float acc3 = 0.0f;",
        ]
    )
    if paired_position is not None:
        lines.extend(
            [
                f"{indent}float acc4 = 0.0f;",
                f"{indent}float acc5 = 0.0f;",
                f"{indent}float acc6 = 0.0f;",
                f"{indent}float acc7 = 0.0f;",
            ]
        )
    if vectorized:
        lines.extend(
            [
                f"{indent}const device vec<T, 4>* row = "
                "reinterpret_cast<const device vec<T, 4>*>("
                f"in_{right_position} + {right_batch_prefix}"
                f"column * {reduction}u);",
                f"{indent}constexpr uint vectors = {reduction // 4}u;",
                f"{indent}for (uint index = lane; index < vectors; index += 32u) {{",
                f"{indent}    vec<T, 4> weight = row[index];",
                f"{indent}    uint base = index * 4u;",
            ]
        )
        if paired_position is not None:
            lines.extend(
                [
                    f"{indent}    const device vec<T, 4>* paired_row = "
                    "reinterpret_cast<const device vec<T, 4>*>("
                    f"in_{paired_position} + {right_batch_prefix}"
                    f"column * {reduction}u);",
                    f"{indent}    vec<T, 4> paired_weight = paired_row[index];",
                ]
            )
        if norm_slot is None:
            for offset in range(4):
                lines.append(
                    f"{indent}    float x{offset} = float(in_{left_position}"
                    f"[{left_batch_prefix}base + {offset}u]);"
                )
        else:
            lines.append(
                f"{indent}    vec<T, 4> norm_source_value = norm_source[index];"
            )
            lines.append(
                f"{indent}    vec<T, 4> norm_weight_value = norm_weight_values[index];"
            )
            for offset in range(4):
                lines.append(
                    f"{indent}    float x{offset} = float((T)(norm_scale * "
                    f"float(norm_source_value[{offset}]) * "
                    f"float(norm_weight_value[{offset}])));"
                )
        for offset in range(4):
            lines.append(
                f"{indent}    acc{offset} = fma(x{offset}, float(weight[{offset}]), "
                f"acc{offset});"
            )
        if paired_position is not None:
            for offset in range(4):
                lines.append(
                    f"{indent}    acc{offset + 4} = fma(x{offset}, "
                    f"float(paired_weight[{offset}]), acc{offset + 4});"
                )
        lines.append(f"{indent}}}")
    else:

        def left_term(index: str) -> str:
            if norm_slot is None:
                return f"float(in_{left_position}[{left_batch_prefix}{index}])"
            return (
                f"float((T)(norm_scale * float(in_{left_position}"
                f"[{left_batch_prefix}{index}]) * float(in_{norm_slot}[{index}])))"
            )

        lines.extend(
            [
                f"{indent}uint k = lane;",
                f"{indent}for (; k + 96u < {reduction}u; k += 128u) {{",
                f"{indent}    acc0 = fma({left_term('k')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k]), acc0);",
                f"{indent}    acc1 = fma({left_term('k + 32u')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k + 32u]), acc1);",
                f"{indent}    acc2 = fma({left_term('k + 64u')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k + 64u]), acc2);",
                f"{indent}    acc3 = fma({left_term('k + 96u')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k + 96u]), acc3);",
                f"{indent}}}",
                f"{indent}for (; k < {reduction}u; k += 32u) {{",
                f"{indent}    acc0 = fma({left_term('k')}, float(in_{right_position}["
                f"{right_batch_prefix}column * {reduction}u + k]), acc0);",
                f"{indent}}}",
            ]
        )
    lines.append(f"{indent}float total = simd_sum((acc0 + acc1) + (acc2 + acc3));")
    if paired_position is not None:
        lines.append(
            f"{indent}float total_up = simd_sum((acc4 + acc5) + (acc6 + acc7));"
        )
        lines.append(f"{indent}float paired_gate = float((T)total);")
    lines.extend(
        [
            f"{indent}if (lane == 0u) {{",
            f"{indent}    out[{output_linear}] = (T)({value});",
            f"{indent}}}",
        ]
    )
    if not exact_columns:
        lines.append("}")
    return "\n".join(f"        {line}" for line in lines)


def _packed_matmul_supported(graph: Graph, op: Operation) -> bool:
    """Whether a folded packed-weight matmul has a native kernel.

    The backend asks this before emitting a candidate, exactly as it asks the
    dense shape rules elsewhere: a shape this build cannot stream should fall
    back to MLX rather than fail the compile. The scale-free checks (batch
    broadcast, dtype agreement) stay in the lowering, where a disagreement is a
    malformed graph rather than an unsupported shape.
    """

    group_size = _matmul_weight_group_size(op)
    if group_size is None:
        return False
    right_position = 2 if op.op == "addmm" else 1
    if len(op.inputs) <= right_position + 1:
        return False
    if op.attrs.get("transpose_b") is not None:
        return False
    packed_type = graph.value(op.inputs[right_position]).type
    left_position = 1 if op.op == "addmm" else 0
    left_type = graph.value(op.inputs[left_position]).type
    output_type = graph.value(op.outputs[0]).type
    if len(packed_type.shape) < 2 or len(left_type.shape) < 2:
        return False
    columns, words = packed_type.shape[-2:]
    reduction = words * _QUANTIZED_NIBBLES
    if left_type.shape[-1] != reduction:
        return False
    return _supports_quantized_gemv_matmul(
        left_type.shape[-2],
        columns,
        reduction,
        group_size,
        output_type.dtype,
    )


def _supports_quantized_gemv_matmul(
    rows: int,
    columns: int,
    reduction: int,
    group_size: int,
    dtype: DType,
) -> bool:
    """Return whether a packed 4-bit weight can use the streaming GEMV.

    The kernel walks the reduction in whole packed words, so the extent has to
    be a multiple of eight, and it reads one scale per word group, so the group
    size has to be too. Those are the only structural conditions; the rest is
    the same one-row shape the dense GEMV needs.
    """

    return (
        rows == 1
        and dtype in {DType.F16, DType.BF16, DType.F32}
        and reduction % _QUANTIZED_NIBBLES == 0
        and group_size % _QUANTIZED_NIBBLES == 0
        and reduction % group_size == 0
        and columns > 0
    )


def _emit_quantized_gemv_matmul_source(
    op: Operation,
    graph: Graph,
    reduction: int,
    columns: int,
    group_size: int,
    batch_count: int,
    left_batch_offset: str,
    weight_batch_offset: str,
    scales_batch_offset: str,
) -> str:
    """Emit a GEMV over a packed 4-bit weight, one warp per output column.

    The body is the dense transposed GEMV with the weight load replaced by a
    word load: one ``int`` carries eight weights, so the streaming loop moves a
    quarter of the bytes for the same number of multiply-adds. Each lane takes
    one word per step, decodes its eight nibbles, multiplies each by the
    activations at the same positions and by the group's scale, and accumulates.

    The scale multiply happens before the accumulate rather than after the
    reduction, because a group's scale applies to its own slice of the reduction
    and folding it out would need every group boundary to be a warp boundary.
    """

    threads = _gemv_threads(columns, True)
    warps = threads // 32
    exact_columns = columns % warps == 0
    output_batch_offset = f"batch * {columns}u + " if batch_count > 1 else ""
    output_linear = f"{output_batch_offset}column"
    if op.op == "addmm":
        left_position = 1
        weight_position = 2
        bias_position: int | None = 0
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
    else:
        left_position = 0
        weight_position = 1
        bias_position = None
        alpha = 1.0
        beta = 1.0
    scales_position = _matmul_quantized_scales_position(op, weight_position)
    assert scales_position is not None
    output_shape = graph.value(op.outputs[0]).type.shape

    value = "total"
    if bias_position is not None:
        bias_type = graph.value(op.inputs[bias_position]).type
        bias_offset = _offset_expression(bias_type.shape, output_shape, output_linear)
        value = f"{beta!r} * float(in_{bias_position}[{bias_offset}]) + {alpha!r} * total"
    value = _matmul_epilogue_value(value, op, graph, output_shape, output_linear)

    words = reduction // _QUANTIZED_NIBBLES
    words_per_group = group_size // _QUANTIZED_NIBBLES
    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint lane = tid % 32u;",
        "uint warp = tid / 32u;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    lines.extend(
        [
            f"uint column = threadgroup_index * {warps}u + warp;",
            f"const device int* weight_row = in_{weight_position} + "
            f"{weight_batch_offset}column * {words}u;",
            f"const device T* scale_row = in_{scales_position} + "
            f"{scales_batch_offset}column * {words // words_per_group}u;",
        ]
    )
    norm_slot = _matmul_norm_position(op)
    if norm_slot is not None:
        lines.extend(
            _emit_warp_rms_norm_prologue(
                op, norm_slot, left_position, reduction, left_batch_offset
            )
        )
    if not exact_columns:
        lines.append(f"if (column < {columns}u) {{")
    indent = "    " if not exact_columns else ""
    lines.extend(f"{indent}float acc{offset} = 0.0f;" for offset in range(8))
    lines.extend(
        [
            f"{indent}for (uint index = lane; index < {words}u; index += 32u) {{",
            f"{indent}    uint word = (uint)weight_row[index];",
            f"{indent}    uint base = index * {_QUANTIZED_NIBBLES}u;",
            f"{indent}    float scale = float(scale_row[index / {words_per_group}u]);",
        ]
    )
    if norm_slot is None:
        lines.extend(
            [
                f"{indent}    const device vec<T, 4>* activations = "
                "reinterpret_cast<const device vec<T, 4>*>("
                f"in_{left_position} + {left_batch_offset}base);",
                f"{indent}    vec<T, 4> low = activations[0];",
                f"{indent}    vec<T, 4> high = activations[1];",
            ]
        )
        for offset in range(8):
            source = "low" if offset < 4 else "high"
            index_in_vector = offset % 4
            lines.append(
                f"{indent}    float x{offset} = float({source}[{index_in_vector}]);"
            )
    else:
        lines.extend(
            [
                f"{indent}    vec<T, 4> norm_low = norm_source[index * 2u];",
                f"{indent}    vec<T, 4> norm_high = norm_source[index * 2u + 1u];",
                f"{indent}    vec<T, 4> weight_low = norm_weight_values[index * 2u];",
                f"{indent}    vec<T, 4> weight_high = "
                "norm_weight_values[index * 2u + 1u];",
            ]
        )
        for offset in range(8):
            source = "norm_low" if offset < 4 else "norm_high"
            weight = "weight_low" if offset < 4 else "weight_high"
            index_in_vector = offset % 4
            lines.append(
                f"{indent}    float x{offset} = float((T)(norm_scale * "
                f"float({source}[{index_in_vector}]) * "
                f"float({weight}[{index_in_vector}])));"
            )
    for offset in range(8):
        lines.extend(
            [
                f"{indent}    int code{offset} = int((word >> "
                f"{4 * offset}u) & 0xFu);",
                f"{indent}    code{offset} = code{offset} >= 8 ? "
                f"code{offset} - 16 : code{offset};",
                f"{indent}    acc{offset} = fma(float(code{offset}), "
                f"x{offset} * scale, acc{offset});",
            ]
        )
    lines.extend(
        [
            f"{indent}}}",
            f"{indent}float total = simd_sum(((acc0 + acc1) + (acc2 + acc3)) + "
            "((acc4 + acc5) + (acc6 + acc7)));",
            f"{indent}if (lane == 0u) {{",
            f"{indent}    out[{output_linear}] = (T)({value});",
            f"{indent}}}",
        ]
    )
    if not exact_columns:
        lines.append("}")
    return "\n".join(f"        {line}" for line in lines)


def _matmul_batch_shape(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    if len(left_shape) < 2 or len(right_shape) < 2:
        return None

    left_batch = left_shape[:-2]
    right_batch = right_shape[:-2]
    batch_rank = max(len(left_batch), len(right_batch))
    batch_shape: list[int] = []
    for axis in range(batch_rank):
        left_dim = (
            left_batch[axis - (batch_rank - len(left_batch))]
            if axis >= batch_rank - len(left_batch)
            else 1
        )
        right_dim = (
            right_batch[axis - (batch_rank - len(right_batch))]
            if axis >= batch_rank - len(right_batch)
            else 1
        )
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            return None
        batch_shape.append(max(left_dim, right_dim))
    return tuple(batch_shape)


def _broadcast_batch_offset_expression(
    input_batch_shape: tuple[int, ...],
    output_batch_shape: tuple[int, ...],
    matrix_elements: int,
) -> str:
    """Map a flat output batch index to the source operand's matrix base."""

    if not output_batch_shape or prod(output_batch_shape) == 1:
        return ""
    if input_batch_shape == output_batch_shape:
        return f"batch * {matrix_elements}u"

    batch_offset = len(output_batch_shape) - len(input_batch_shape)
    terms: list[str] = []
    for input_axis, input_dim in enumerate(input_batch_shape):
        output_axis = input_axis + batch_offset
        if input_dim == 1:
            continue
        output_stride = prod(output_batch_shape[output_axis + 1 :])
        input_stride = prod(input_batch_shape[input_axis + 1 :]) * matrix_elements
        coordinate = (
            f"batch % {output_batch_shape[output_axis]}u"
            if output_stride == 1
            else f"((batch / {output_stride}u) % {output_batch_shape[output_axis]}u)"
        )
        terms.append(f"{coordinate} * {input_stride}u")
    return " + ".join(terms)


def _matmul_transpose_attribute(op: Operation, name: str) -> bool:
    value = op.attrs.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"matmul {name} must be a boolean")
    return value


def _emit_simdgroup_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    graph: Graph,
    dtype: DType,
    transpose_left: bool,
    transpose_right: bool,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
) -> str:
    if rows % _MATMUL_BLOCK_ROWS != 0 or columns % _MATMUL_BLOCK_COLUMNS != 0:
        raise ValueError("simdgroup matmul requires complete 64x64 output blocks")

    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    block_rows = _MATMUL_BLOCK_ROWS
    block_columns = _MATMUL_BLOCK_COLUMNS
    block_reduction = _MATMUL_BLOCK_REDUCTION
    threads = 256
    column_blocks = columns // block_columns
    exact_reduction = reduction % block_reduction == 0
    left_batch_prefix = f"{left_batch_offset} + " if left_batch_offset else ""
    right_batch_prefix = f"{right_batch_offset} + " if right_batch_offset else ""
    output_batch_offset = f"batch * {rows * columns}u + " if batch_count > 1 else ""
    load_vector_width = 8
    vector_loads = (
        exact_reduction
        and reduction % load_vector_width == 0
        and columns % load_vector_width == 0
        and not transpose_left
        and not transpose_right
    )
    tile_alignment = "alignas(16) " if vector_loads else ""
    simdgroup_type = {
        DType.F16: "half",
        DType.BF16: "bfloat",
    }[dtype]
    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint warp = tid / 32u;",
        "uint warp_row = warp / 2u;",
        "uint warp_column = warp % 2u;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    lines.extend(
        [
            f"uint block_row = threadgroup_index / {column_blocks}u;",
            f"uint block_column = threadgroup_index % {column_blocks}u;",
            f"uint row_base = block_row * {block_rows}u;",
            f"uint col_base = block_column * {block_columns}u;",
            f"{tile_alignment}threadgroup T tile_a[{block_rows * block_reduction}];",
            f"{tile_alignment}threadgroup T tile_b[{block_reduction * block_columns}];",
            f"threadgroup float tile_c[{block_rows * block_columns}];",
            "simdgroup_float8x8 acc[8];",
            f"simdgroup_{simdgroup_type}8x8 frag_a[2];",
            f"simdgroup_{simdgroup_type}8x8 frag_b[4];",
            "for (uint i = 0u; i < 8u; ++i) {",
            "    acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);",
            "}",
            f"for (uint k_base = 0u; k_base < {reduction}u; "
            f"k_base += {block_reduction}u) {{",
        ]
    )
    if vector_loads:
        lines.extend(
            [
                f"    for (uint vector_index = tid; "
                f"vector_index < {block_rows * block_reduction // load_vector_width}u; "
                f"vector_index += {threads}u) {{",
                f"        uint index = vector_index * {load_vector_width}u;",
                f"        uint local_row = index / {block_reduction}u;",
                f"        uint local_k = index % {block_reduction}u;",
                f"        threadgroup vec<T, {load_vector_width}>* destination =",
                f"            reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(",
                "                tile_a + index);",
                f"        const device vec<T, {load_vector_width}>* source =",
                f"            reinterpret_cast<const device vec<T, {load_vector_width}>*>(",
                f"                in_{left_position} + {left_batch_prefix}"
                f"(row_base + local_row) * {reduction}u",
                "                    + k_base + local_k);",
                "        *destination = *source;",
                "    }",
                f"    for (uint vector_index = tid; "
                f"vector_index < {block_reduction * block_columns // load_vector_width}u; "
                f"vector_index += {threads}u) {{",
                f"        uint index = vector_index * {load_vector_width}u;",
                f"        uint local_k = index / {block_columns}u;",
                f"        uint local_column = index % {block_columns}u;",
                f"        threadgroup vec<T, {load_vector_width}>* destination =",
                f"            reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(",
                "                tile_b + index);",
                f"        const device vec<T, {load_vector_width}>* source =",
                f"            reinterpret_cast<const device vec<T, {load_vector_width}>*>(",
                f"                in_{right_position} + {right_batch_prefix}"
                f"(k_base + local_k) * {columns}u",
                "                    + col_base + local_column);",
                "        *destination = *source;",
                "    }",
            ]
        )
    else:
        lines.extend(
            [
                f"    for (uint index = tid; "
                f"index < {block_rows * block_reduction}u; "
                f"index += {threads}u) {{",
                f"        uint local_row = index / {block_reduction}u;",
                f"        uint local_k = index % {block_reduction}u;",
            ]
        )
        if transpose_left:
            left_index = f"(k_base + local_k) * {rows}u + row_base + local_row"
        else:
            left_index = f"(row_base + local_row) * {reduction}u + k_base + local_k"
        if left_batch_offset:
            left_index = f"{left_batch_offset} + {left_index}"
        load_a = f"in_{left_position}[{left_index}]"
        if not exact_reduction:
            load_a = f"k_base + local_k < {reduction}u ? {load_a} : T(0.0)"
        lines.extend(
            [
                f"        tile_a[index] = {load_a};",
                "    }",
                f"    for (uint index = tid; "
                f"index < {block_reduction * block_columns}u; "
                f"index += {threads}u) {{",
                f"        uint local_k = index / {block_columns}u;",
                f"        uint local_column = index % {block_columns}u;",
            ]
        )
        if transpose_right:
            right_index = f"(col_base + local_column) * {reduction}u + k_base + local_k"
        else:
            right_index = f"(k_base + local_k) * {columns}u + col_base + local_column"
        if right_batch_offset:
            right_index = f"{right_batch_offset} + {right_index}"
        load_b = f"in_{right_position}[{right_index}]"
        if not exact_reduction:
            load_b = f"k_base + local_k < {reduction}u ? {load_b} : T(0.0)"
        lines.extend(
            [
                f"        tile_b[index] = {load_b};",
                "    }",
            ]
        )
    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")

    for k_offset in range(0, block_reduction, 8):
        lines.extend(
            [
                f"    simdgroup_load(frag_a[0], tile_a + "
                f"(warp_row * 16u) * {block_reduction}u + {k_offset}u, "
                f"{block_reduction}, 0, false);",
                f"    simdgroup_load(frag_a[1], tile_a + "
                f"(warp_row * 16u + 8u) * {block_reduction}u + {k_offset}u, "
                f"{block_reduction}, 0, false);",
            ]
        )
        for column_fragment in range(4):
            lines.append(
                f"    simdgroup_load(frag_b[{column_fragment}], tile_b + "
                f"{k_offset}u * {block_columns}u + warp_column * 32u + "
                f"{column_fragment * 8}u, {block_columns}, 0, false);"
            )
        for row_fragment in range(2):
            for column_fragment in range(4):
                accumulator = row_fragment * 4 + column_fragment
                lines.append(
                    f"    simdgroup_multiply_accumulate(acc[{accumulator}], "
                    f"frag_a[{row_fragment}], frag_b[{column_fragment}], "
                    f"acc[{accumulator}]);"
                )

    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append("}")

    for row_fragment in range(2):
        row_offset = row_fragment * 8
        for column_fragment in range(4):
            accumulator = row_fragment * 4 + column_fragment
            column_offset = column_fragment * 8
            lines.append(
                f"simdgroup_store(acc[{accumulator}], tile_c + "
                f"(warp_row * 16u + {row_offset}u) * {block_columns}u + "
                f"warp_column * 32u + {column_offset}u, "
                f"{block_columns}, 0, false);"
            )
    lines.extend(
        [
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"for (uint index = tid; index < {block_rows * block_columns}u; "
            f"index += {threads}u) {{",
            f"    uint local_row = index / {block_columns}u;",
            f"    uint local_column = index % {block_columns}u;",
            f"    uint linear = {output_batch_offset}(row_base + local_row) * {columns}u "
            "+ col_base + local_column;",
        ]
    )
    value = "tile_c[index]"
    if bias_id is not None:
        assert bias_type is not None
        bias_offset = _offset_expression(bias_type.shape, output_shape, "linear")
        value = f"{beta!r} * float(in_0[{bias_offset}]) + {alpha!r} * {value}"
    value = _matmul_epilogue_value(value, op, graph, output_shape, "linear")
    lines.extend(
        [
            f"    out[linear] = (T)({value});",
            "}",
        ]
    )
    return "\n".join(f"        {line}" for line in lines)


def _emit_mpp_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    graph: Graph,
    dtype: DType,
    transpose_left: bool,
    transpose_right: bool,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
    config: tuple[int, int, int, int, int],
) -> str:
    """Emit a ``mpp::tensor_ops::matmul2d`` GEMM for Apple tensor cores."""

    block_rows, block_columns, warp_rows, warp_columns, step = config
    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    warp_block_rows = block_rows // warp_rows
    warp_block_columns = block_columns // warp_columns
    row_fragments = warp_block_rows // 16
    column_pairs = warp_block_columns // 32
    step_fragments = step // 16
    if row_fragments < 1 or column_pairs < 1:
        raise ValueError("MPP matmul requires at least one 16x32 warp tile")

    full_steps, tail_fragments = divmod(reduction, step)
    tail_fragments //= 16
    guard_rows = rows % block_rows != 0
    guard_columns = columns % block_columns != 0
    left_batch_prefix = f"{left_batch_offset} + " if left_batch_offset else ""
    right_batch_prefix = f"{right_batch_offset} + " if right_batch_offset else ""
    output_batch_prefix = f"batch * {rows * columns}u + " if batch_count > 1 else ""

    lines: list[str] = []
    append = lines.append
    append("uint tid = thread_position_in_threadgroup.x;")
    append("uint lane = tid % 32u;")
    append("uint sg = tid / 32u;")
    append("uint qid = lane >> 2u;")
    append("uint fm = (qid & 4u) | ((lane >> 1u) & 3u);")
    append("uint fn = ((qid & 2u) | (lane & 1u)) * 4u;")
    if batch_count > 1:
        append("uint batch = threadgroup_position_in_grid.z;")
    append(
        f"uint row_base = threadgroup_position_in_grid.y * {block_rows}u"
        f" + (sg / {warp_columns}u) * {warp_block_rows}u;"
    )
    append(
        f"uint col_base = threadgroup_position_in_grid.x * {block_columns}u"
        f" + (sg % {warp_columns}u) * {warp_block_columns}u;"
    )
    append("constexpr auto tsc_gemm_descriptor = mpp::tensor_ops::matmul2d_descriptor(")
    append("    16, 32, 16, false, true, true,")
    append("    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);")
    append(
        "mpp::tensor_ops::matmul2d<tsc_gemm_descriptor, "
        "metal::execution_simdgroup> tsc_gemm;"
    )
    append(
        "using tsc_at = decltype("
        "tsc_gemm.get_left_input_cooperative_tensor<T, T, float>());"
    )
    append(
        "using tsc_bt = decltype("
        "tsc_gemm.get_right_input_cooperative_tensor<T, T, float>());"
    )
    append(
        "using tsc_ct = decltype(tsc_gemm.get_destination_cooperative_tensor<"
        "metal::remove_addrspace_t<tsc_at>, "
        "metal::remove_addrspace_t<tsc_bt>, float>());"
    )
    for row_fragment in range(row_fragments):
        for column_pair in range(column_pairs):
            name = f"tsc_acc_{row_fragment}_{column_pair}"
            append(f"tsc_ct {name} = tsc_gemm.get_destination_cooperative_tensor<")
            append(
                "    metal::remove_addrspace_t<tsc_at>, "
                "metal::remove_addrspace_t<tsc_bt>, float>();"
            )
            append(f"for (uint e = 0u; e < 16u; ++e) {name}[e] = 0.0f;")

    def row_expression(row_fragment: int, row_offset: int) -> str:
        expression = f"row_base + {row_fragment * 16 + row_offset}u + fm"
        if guard_rows:
            return f"metal::min({expression}, {rows - 1}u)"
        return expression

    def column_load_expression(column_fragment: int, column_offset: int) -> str:
        # The right cooperative tensor indexes its ``n`` rows with ``fm``.
        expression = f"col_base + {column_fragment * 16 + column_offset}u + fm"
        if guard_columns:
            return f"metal::min({expression}, {columns - 1}u)"
        return expression

    def column_store_expression(column_fragment: int, column_offset: int) -> str:
        # The destination cooperative tensor indexes its columns with ``fn``.
        expression = f"col_base + {column_fragment * 16 + column_offset}u + fn"
        return expression

    def k_expression(origin: str | int, offset: int) -> str:
        # Every lane owns ``fn`` of the 16-wide reduction fragment.
        if isinstance(origin, int):
            return f"{origin + offset}u + fn"
        return f"{origin} + {offset}u + fn"

    def emit_declarations(fragments: int, prefix: str, indent: str) -> list[str]:
        emitted: list[str] = []
        for row_fragment in range(row_fragments):
            for fragment in range(fragments):
                emitted.append(
                    f"{indent}tsc_at {prefix}ca_{row_fragment}_{fragment} ="
                    " tsc_gemm.get_left_input_cooperative_tensor<T, T, float>();"
                )
        for column_pair in range(column_pairs):
            for fragment in range(fragments):
                emitted.append(
                    f"{indent}tsc_bt {prefix}cb_{fragment}_{column_pair} ="
                    " tsc_gemm.get_right_input_cooperative_tensor<T, T, float>();"
                )
        return emitted

    def emit_body(
        origin: str | int, fragments: int, prefix: str, indent: str
    ) -> list[str]:
        emitted: list[str] = []
        for row_fragment in range(row_fragments):
            for fragment in range(fragments):
                for half in range(2):
                    for element in range(4):
                        k = k_expression(origin, fragment * 16 + element)
                        target = f"{prefix}ca_{row_fragment}_{fragment}"
                        if transpose_left:
                            index = (
                                f"{left_batch_prefix}({k})"
                                f" * {rows}u + ({row_expression(row_fragment, half * 8)})"
                            )
                        else:
                            index = (
                                f"{left_batch_prefix}"
                                f"({row_expression(row_fragment, half * 8)})"
                                f" * {reduction}u + ({k})"
                            )
                        emitted.append(
                            f"{indent}{target}[{half * 4 + element}] ="
                            f" in_{left_position}[{index}];"
                        )
        for column_pair in range(column_pairs):
            for fragment in range(fragments):
                for half in range(2):
                    for nf in range(2):
                        column_fragment = column_pair * 2 + nf
                        for element in range(4):
                            k = k_expression(origin, fragment * 16 + element)
                            target = f"{prefix}cb_{fragment}_{column_pair}"
                            if transpose_right:
                                index = (
                                    f"{right_batch_prefix}"
                                    f"({column_load_expression(column_fragment, half * 8)})"
                                    f" * {reduction}u + ({k})"
                                )
                            else:
                                index = (
                                    f"{right_batch_prefix}({k})"
                                    f" * {columns}u"
                                    f" + ({column_load_expression(column_fragment, half * 8)})"
                                )
                            emitted.append(
                                f"{indent}{target}[{nf * 8 + half * 4 + element}] ="
                                f" in_{right_position}[{index}];"
                            )
        for row_fragment in range(row_fragments):
            for column_pair in range(column_pairs):
                for fragment in range(fragments):
                    emitted.append(
                        f"{indent}tsc_gemm.run({prefix}ca_{row_fragment}_{fragment},"
                        f" {prefix}cb_{fragment}_{column_pair},"
                        f" tsc_acc_{row_fragment}_{column_pair});"
                    )
        return emitted

    if full_steps > 0:
        lines.extend(emit_declarations(step_fragments, "loop_", ""))
        append(f"for (uint k0 = 0u; k0 < {full_steps * step}u; k0 += {step}u) {{")
        lines.extend(emit_body("k0", step_fragments, "loop_", "    "))
        append("}")
    if tail_fragments > 0:
        lines.extend(emit_declarations(tail_fragments, "tail_", ""))
        lines.extend(emit_body(full_steps * step, tail_fragments, "tail_", ""))

    for row_fragment in range(row_fragments):
        for column_pair in range(column_pairs):
            for nf in range(2):
                column_fragment = column_pair * 2 + nf
                for half in range(2):
                    for element in range(4):
                        row = row_expression(row_fragment, half * 8)
                        column = column_store_expression(column_fragment, element)
                        linear = f"({row}) * {columns}u + ({column})"
                        value = f"float(tsc_acc_{row_fragment}_{column_pair}[{nf * 8 + half * 4 + element}])"
                        if bias_id is not None:
                            assert bias_type is not None
                            bias_index = _offset_expression(
                                bias_type.shape, output_shape, linear
                            )
                            value = (
                                f"{beta!r} * float(in_0[{bias_index}])"
                                f" + {alpha!r} * {value}"
                            )
                        elif alpha != 1.0:
                            value = f"{alpha!r} * {value}"
                        value = _matmul_epilogue_value(
                            value, op, graph, output_shape, linear
                        )
                        conditions = []
                        if guard_rows:
                            conditions.append(
                                f"row_base + {row_fragment * 16 + half * 8}u"
                                f" + fm < {rows}u"
                            )
                        if guard_columns:
                            conditions.append(
                                f"col_base + {column_fragment * 16 + element}u"
                                f" + fn < {columns}u"
                            )
                        guard = f"if ({' && '.join(conditions)}) " if conditions else ""
                        append(
                            f"{guard}out[{output_batch_prefix}{linear}] = (T)({value});"
                        )
    return "\n".join(lines)


def _emit_tiled_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    micro_rows: int,
    micro_columns: int,
    graph: Graph,
    transpose_left: bool,
    transpose_right: bool,
    batch_count: int,
    left_batch_offset: str,
    right_batch_offset: str,
) -> str:
    if op.op == "addmm":
        bias_id = op.inputs[0]
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    block_rows = _MATMUL_BLOCK_ROWS
    block_columns = _MATMUL_BLOCK_COLUMNS
    block_reduction = _MATMUL_BLOCK_REDUCTION
    micro_rows_per_block = block_rows // micro_rows
    micro_columns_per_block = block_columns // micro_columns
    threads = micro_rows_per_block * micro_columns_per_block
    column_blocks = ceil(columns / block_columns)
    exact_rows = rows % block_rows == 0
    exact_columns = columns % block_columns == 0
    exact_reduction = reduction % block_reduction == 0
    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint threadgroup_index = threadgroup_position_in_grid.x;",
    ]
    if batch_count > 1:
        lines.append("uint batch = threadgroup_position_in_grid.y;")
    lines.extend(
        [
            f"uint block_row = threadgroup_index / {column_blocks}u;",
            f"uint block_column = threadgroup_index % {column_blocks}u;",
            f"uint row_base = block_row * {block_rows}u;",
            f"uint col_base = block_column * {block_columns}u;",
            f"uint micro_row = tid / {micro_columns_per_block}u;",
            f"uint micro_column = tid % {micro_columns_per_block}u;",
            f"threadgroup float tile_a[{block_rows * block_reduction}];",
            f"threadgroup float tile_b[{block_reduction * block_columns}];",
        ]
    )
    for row in range(micro_rows):
        for column in range(micro_columns):
            lines.append(f"float acc_{row}_{column} = 0.0f;")
    lines.append(
        f"for (uint k_base = 0u; k_base < {reduction}u; "
        f"k_base += {block_reduction}u) {{"
    )

    lines.append(
        f"    for (uint index = tid; index < {block_rows * block_reduction}u; "
        f"index += {threads}u) {{"
    )
    lines.append(f"        uint local_row = index / {block_reduction}u;")
    lines.append(f"        uint local_k = index % {block_reduction}u;")
    if transpose_left:
        left_index = f"(k_base + local_k) * {rows}u + row_base + local_row"
    else:
        left_index = f"(row_base + local_row) * {reduction}u + k_base + local_k"
    if left_batch_offset:
        left_index = f"{left_batch_offset} + {left_index}"
    load_a = f"float(in_{left_position}[{left_index}])"
    a_guards = []
    if not exact_rows:
        a_guards.append(f"row_base + local_row < {rows}u")
    if not exact_reduction:
        a_guards.append(f"k_base + local_k < {reduction}u")
    if a_guards:
        load_a = f"{' && '.join(a_guards)} ? {load_a} : 0.0f"
    lines.append(f"        tile_a[index] = {load_a};")
    lines.append("    }")

    lines.append(
        f"    for (uint index = tid; index < {block_reduction * block_columns}u; "
        f"index += {threads}u) {{"
    )
    lines.append(f"        uint local_k = index / {block_columns}u;")
    lines.append(f"        uint local_column = index % {block_columns}u;")
    if transpose_right:
        right_index = f"(col_base + local_column) * {reduction}u + k_base + local_k"
    else:
        right_index = f"(k_base + local_k) * {columns}u + col_base + local_column"
    if right_batch_offset:
        right_index = f"{right_batch_offset} + {right_index}"
    load_b = f"float(in_{right_position}[{right_index}])"
    b_guards = []
    if not exact_columns:
        b_guards.append(f"col_base + local_column < {columns}u")
    if not exact_reduction:
        b_guards.append(f"k_base + local_k < {reduction}u")
    if b_guards:
        load_b = f"{' && '.join(b_guards)} ? {load_b} : 0.0f"
    lines.append(f"        tile_b[index] = {load_b};")
    lines.append("    }")
    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")

    lines.append(
        f"    for (uint local_k = 0u; local_k < {block_reduction}u; ++local_k) {{"
    )
    for row in range(micro_rows):
        lines.append(
            f"        float av_{row} = tile_a[(micro_row * {micro_rows}u + {row}u) "
            f"* {block_reduction}u + local_k];"
        )
    for column in range(micro_columns):
        lines.append(
            f"        float bv_{column} = tile_b[local_k * {block_columns}u "
            f"+ micro_column * {micro_columns}u + {column}u];"
        )
    for row in range(micro_rows):
        for column in range(micro_columns):
            lines.append(
                f"        acc_{row}_{column} = fma("
                f"av_{row}, bv_{column}, acc_{row}_{column});"
            )
    lines.append("    }")
    lines.append("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append("}")

    exact_tiles = exact_rows and exact_columns
    if not exact_tiles:
        for row in range(micro_rows):
            lines.append(
                f"bool row_ok_{row} = "
                f"row_base + micro_row * {micro_rows}u + {row}u < {rows}u;"
            )
        for column in range(micro_columns):
            lines.append(
                f"bool col_ok_{column} = "
                f"col_base + micro_column * {micro_columns}u + {column}u < {columns}u;"
            )

    for row in range(micro_rows):
        for column in range(micro_columns):
            linear = f"linear_{row}_{column}"
            if not exact_tiles:
                lines.append(f"if (row_ok_{row} && col_ok_{column}) {{")
            indent = "    " if not exact_tiles else ""
            batch_offset = f"batch * {rows * columns}u + " if batch_count > 1 else ""
            lines.append(
                f"{indent}uint {linear} = "
                f"{batch_offset}"
                f"(row_base + micro_row * {micro_rows}u + {row}u) * {columns}u "
                f"+ col_base + micro_column * {micro_columns}u + {column}u;"
            )
            accumulator = f"acc_{row}_{column}"
            if bias_id is not None:
                assert bias_type is not None
                bias_offset = _offset_expression(bias_type.shape, output_shape, linear)
                value = (
                    f"{beta!r} * float(in_0[{bias_offset}]) + {alpha!r} * {accumulator}"
                )
            else:
                value = accumulator
            value = _matmul_epilogue_value(value, op, graph, output_shape, linear)
            lines.append(f"{indent}out[{linear}] = (T)({value});")
            if not exact_tiles:
                lines.append("}")

    return "\n".join(f"        {line}" for line in lines)


def _matmul_quantized_scales_position(op: Operation, right_position: int) -> int | None:
    """Return the input slot holding the scales beside a packed weight."""

    if _matmul_weight_group_size(op) is None:
        return None
    return right_position + 1


def _matmul_epilogue_kind(op: Operation) -> str | None:
    """Return the fused activation epilogue attached to a matrix multiply."""

    kind = op.attrs.get("epilogue")
    if kind is None:
        return None
    if kind not in {"silu_mul", "silu_mul_paired"}:
        raise ValueError(f"unsupported matmul epilogue {kind!r}")
    return kind


def _matmul_paired_weight_position(op: Operation, right_position: int) -> int | None:
    """Return the input slot holding the second projection of a paired epilogue."""

    if _matmul_epilogue_kind(op) != "silu_mul_paired":
        return None
    return right_position + 1


def _matmul_norm_position(op: Operation) -> int | None:
    """Return the input slot holding a fused RMS-norm weight, if any."""

    if op.attrs.get("rms_norm_eps") is None:
        return None
    return len(op.inputs) - 1


def _matmul_epilogue_position(op: Operation) -> int | None:
    """Return the input slot holding the fused epilogue operand, if any."""

    kind = _matmul_epilogue_kind(op)
    if kind is None:
        return None
    if kind == "silu_mul_paired":
        raise ValueError(
            "the paired gate/up epilogue is only lowered by the transposed GEMV"
        )
    return len(op.inputs) - 1 - int(_matmul_norm_position(op) is not None)


def _matmul_epilogue_value(
    value: str,
    op: Operation,
    graph: Graph,
    output_shape: tuple[int, ...],
    linear: str,
) -> str:
    """Apply the fused activation epilogue to one accumulated output element.

    ``silu_mul`` multiplies the accumulation by ``silu`` of the gate tensor that
    the activation would have consumed, so the multiply that followed the
    projection disappears into this kernel.
    """

    position = _matmul_epilogue_position(op)
    if position is None:
        return value
    gate_id = op.inputs[position]
    gate_type = graph.value(gate_id).type
    gate_offset = _offset_expression(gate_type.shape, output_shape, linear)
    gate = f"float(in_{position}[{gate_offset}])"
    return f"({value}) * ({gate} / (1.0f + metal::exp(-{gate})))"


class MatmulLowering:
    def _compile_matmul(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        right_position = 2 if op.op == "addmm" else 1
        quantized = _matmul_quantized_scales_position(op, right_position) is not None
        expected_inputs = 3 if op.op == "addmm" else 2
        if quantized:
            # The packed weight and its scales ride in one extra slot.
            expected_inputs += 1
        epilogue = _matmul_epilogue_kind(op)
        if epilogue is not None:
            # Either a gate value to scale by, or a second weight matrix.
            expected_inputs += 1
        norm_position = _matmul_norm_position(op)
        if norm_position is not None:
            expected_inputs += 1
        if len(op.inputs) != expected_inputs:
            raise ValueError(f"{op.op} expects {expected_inputs} inputs")

        if norm_position is not None:
            norm_type = graph.value(op.inputs[norm_position]).type
            left_position = 1 if op.op == "addmm" else 0
            left_type = graph.value(op.inputs[left_position]).type
            if norm_type.shape != (left_type.shape[-1],):
                raise ValueError(
                    f"fused rms_norm weight shape {norm_type.shape} does not match "
                    f"the reduction dimension {left_type.shape[-1]}"
                )

        if op.op == "addmm":
            bias_id, left_id, right_id = op.inputs[:3]
        else:
            bias_id = None
            left_id, right_id = op.inputs[:2]

        if quantized:
            return self._compile_quantized_matmul(
                graph, op, index, right_position=right_position
            )

        left_type = graph.value(left_id).type
        right_type = graph.value(right_id).type
        output_type = graph.value(op.outputs[0]).type
        if len(left_type.shape) < 2 or len(right_type.shape) < 2:
            raise NotImplementedError("Metal matmul requires tensors with rank >= 2")
        batch_shape = _matmul_batch_shape(left_type.shape, right_type.shape)
        if batch_shape is None:
            raise NotImplementedError(
                "Metal matmul batch dimensions are not broadcast-compatible"
            )

        transpose_left = _matmul_transpose_attribute(op, "transpose_a")
        transpose_right = _matmul_transpose_attribute(op, "transpose_b")
        logical_left = left_type.shape
        logical_right = right_type.shape
        if transpose_left:
            logical_left = logical_left[:-2] + logical_left[-2:][::-1]
        if transpose_right:
            logical_right = logical_right[:-2] + logical_right[-2:][::-1]
        rows, reduction = logical_left[-2:]
        right_reduction, columns = logical_right[-2:]
        if reduction != right_reduction:
            raise ValueError(
                f"matmul reduction dimensions differ: {reduction} and {right_reduction}"
            )
        expected_output = batch_shape + (rows, columns)
        if output_type.shape != expected_output:
            raise ValueError(
                f"matmul output shape {output_type.shape} != {expected_output}"
            )
        batch_count = prod(batch_shape)
        left_batch_offset = _broadcast_batch_offset_expression(
            left_type.shape[:-2], batch_shape, rows * reduction
        )
        right_batch_offset = _broadcast_batch_offset_expression(
            right_type.shape[:-2], batch_shape, reduction * columns
        )

        dtypes = {left_type.dtype, right_type.dtype, output_type.dtype}
        if bias_id is not None:
            bias_type = graph.value(bias_id).type
            dtypes.add(bias_type.dtype)
            if not _can_broadcast(bias_type.shape, output_type.shape):
                raise NotImplementedError(
                    f"cannot broadcast addmm bias {bias_type.shape} "
                    f"to {output_type.shape}"
                )
        if len(dtypes) != 1:
            raise NotImplementedError(
                "Metal matmul requires identical input/output dtypes"
            )

        row_blocks = ceil(rows / _MATMUL_BLOCK_ROWS)
        column_blocks = ceil(columns / _MATMUL_BLOCK_COLUMNS)
        header = ""
        mpp_grid: tuple[int, int, int] | None = None
        if norm_position is not None and not _supports_gemv_matmul(
            rows,
            output_type.dtype,
            transpose_left,
            transpose_right,
        ):
            raise NotImplementedError(
                "fused rms_norm requires the Metal GEMV matmul path"
            )
        if _supports_gemv_matmul(
            rows,
            output_type.dtype,
            transpose_left,
            transpose_right,
        ):
            source = _emit_gemv_matmul_source(
                op,
                reduction,
                columns,
                graph,
                batch_count,
                left_batch_offset,
                right_batch_offset,
                transpose_right,
            )
            threads = _gemv_threads(columns, transpose_right)
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            layout = "transposed" if transpose_right else "row_major"
            name = f"metal_{op.op}_gemv_{layout}_{threads}{batch_suffix}_{index}"
        elif (
            _supports_mpp_matmul(rows, columns, reduction, output_type.dtype)
            and target._mpp_supported()
        ):
            mpp_config = _select_mpp_gemm_config(
                rows, columns, reduction, transpose_right
            )
            (
                mpp_block_rows,
                mpp_block_columns,
                mpp_warp_rows,
                mpp_warp_columns,
                mpp_step,
            ) = mpp_config
            source = _emit_mpp_matmul_source(
                op,
                rows,
                reduction,
                columns,
                graph,
                output_type.dtype,
                transpose_left,
                transpose_right,
                batch_count,
                left_batch_offset,
                right_batch_offset,
                mpp_config,
            )
            header = _MPP_HEADER
            threads = mpp_warp_rows * mpp_warp_columns * 32
            mpp_grid = (
                ceil(columns / mpp_block_columns) * threads,
                ceil(rows / mpp_block_rows),
                batch_count,
            )
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            orientation = ("t" if transpose_left else "n") + (
                "t" if transpose_right else "n"
            )
            name = (
                f"metal_{op.op}_tensorcore_{orientation}_"
                f"{mpp_block_rows}x{mpp_block_columns}x{mpp_step}"
                f"_{mpp_warp_rows}x{mpp_warp_columns}"
                f"{batch_suffix}_{index}"
            )
        elif _supports_simdgroup_matmul(rows, columns, output_type.dtype):
            source = _emit_simdgroup_matmul_source(
                op,
                rows,
                reduction,
                columns,
                graph,
                output_type.dtype,
                transpose_left,
                transpose_right,
                batch_count,
                left_batch_offset,
                right_batch_offset,
            )
            threads = 256
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            name = (
                f"metal_{op.op}_simdgroup_"
                f"{_MATMUL_BLOCK_ROWS}x{_MATMUL_BLOCK_COLUMNS}x"
                f"{_MATMUL_BLOCK_REDUCTION}{batch_suffix}_{index}"
            )
        else:
            micro_rows, micro_columns = _select_matmul_microtile(reduction)
            source = _emit_tiled_matmul_source(
                op,
                rows,
                reduction,
                columns,
                micro_rows,
                micro_columns,
                graph,
                transpose_left,
                transpose_right,
                batch_count,
                left_batch_offset,
                right_batch_offset,
            )
            threads = (_MATMUL_BLOCK_ROWS // micro_rows) * (
                _MATMUL_BLOCK_COLUMNS // micro_columns
            )
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            name = (
                f"metal_{op.op}_tiled_"
                f"{_MATMUL_BLOCK_ROWS}x{_MATMUL_BLOCK_COLUMNS}x"
                f"{_MATMUL_BLOCK_REDUCTION}_{micro_rows}x{micro_columns}"
                f"{batch_suffix}_{index}"
            )
        kernel = msl._metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            header,
        )
        if mpp_grid is not None:
            grid = mpp_grid
        elif _supports_gemv_matmul(
            rows,
            output_type.dtype,
            transpose_left,
            transpose_right,
        ):
            columns_per_group = threads // 32 if transpose_right else threads
            grid = (
                ceil(columns / columns_per_group) * threads,
                batch_count,
                1,
            )
        else:
            grid = (
                row_blocks * column_blocks * threads,
                batch_count,
                1,
            )
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=header,
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind=op.op,
        )
        return artifact, runtime

    def _compile_quantized_matmul(
        self,
        graph: Graph,
        op: Operation,
        index: int,
        *,
        right_position: int,
    ) -> tuple[KernelArtifact, Any]:
        """Emit the streaming GEMV over a packed 4-bit weight.

        Only the one-row shape is lowered natively. A wider left operand would
        have to reuse each decoded weight over several rows, which is a tiled
        kernel rather than this one, and the codegen declines so the MLX
        fallback unpacks and runs - correct, and only paid for by prefill.
        """

        group_size = _matmul_weight_group_size(op)
        assert group_size is not None
        scales_position = _matmul_quantized_scales_position(op, right_position)
        assert scales_position is not None
        left_position = 1 if op.op == "addmm" else 0

        packed_type = graph.value(op.inputs[right_position]).type
        scales_type = graph.value(op.inputs[scales_position]).type
        left_type = graph.value(op.inputs[left_position]).type
        output_type = graph.value(op.outputs[0]).type
        if len(packed_type.shape) < 2:
            raise NotImplementedError("packed matmul weights must be rank >= 2")
        columns, words = packed_type.shape[-2:]
        rows = left_type.shape[-2] if len(left_type.shape) >= 2 else 0
        reduction = words * _QUANTIZED_NIBBLES
        if left_type.shape[-1] != reduction:
            raise ValueError(
                f"packed weight reduction {reduction} does not match the left "
                f"operand's {left_type.shape[-1]}"
            )
        if scales_type.shape[-2:] != (columns, reduction // group_size):
            raise ValueError(
                f"packed weight scales shape {scales_type.shape} does not cover "
                f"{columns} columns in groups of {group_size}"
            )
        if not _supports_quantized_gemv_matmul(
            rows, columns, reduction, group_size, output_type.dtype
        ):
            raise NotImplementedError(
                "the packed 4-bit matmul is only lowered natively for a single "
                f"row with a reduction in whole words, got {rows} rows and "
                f"{reduction} x {columns} in groups of {group_size}"
            )
        batch_shape = _matmul_batch_shape(
            left_type.shape,
            packed_type.shape,
        )
        if batch_shape is None:
            raise NotImplementedError(
                "packed matmul batch dimensions are not broadcast-compatible"
            )
        if _matmul_transpose_attribute(op, "transpose_a"):
            raise NotImplementedError("packed matmul does not support a transposed left")
        if _matmul_transpose_attribute(op, "transpose_b"):
            raise ValueError(
                "a packed weight already carries the [out, in] orientation, so "
                "transpose_b must not be set"
            )
        if left_type.dtype != output_type.dtype or scales_type.dtype != output_type.dtype:
            raise NotImplementedError(
                "the packed matmul requires the activation, scales and output to "
                "share one dtype"
            )
        batch_count = prod(batch_shape)
        left_batch_offset = _broadcast_batch_offset_expression(
            left_type.shape[:-2], batch_shape, rows * reduction
        )
        weight_batch_offset = _broadcast_batch_offset_expression(
            packed_type.shape[:-2], batch_shape, columns * words
        )
        scales_batch_offset = _broadcast_batch_offset_expression(
            scales_type.shape[:-2], batch_shape, columns * (reduction // group_size)
        )

        source = _emit_quantized_gemv_matmul_source(
            op,
            graph,
            reduction,
            columns,
            group_size,
            batch_count,
            left_batch_offset,
            weight_batch_offset,
            scales_batch_offset,
        )
        threads = _gemv_threads(columns, True)
        warps = threads // 32
        batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
        name = (
            f"metal_{op.op}_gemv_int4_g{group_size}_{threads}"
            f"{batch_suffix}_{index}"
        )
        kernel = msl._metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            "",
        )
        grid = (ceil(columns / warps) * threads, batch_count, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=(threads, 1, 1),
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header="",
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind=op.op,
        )
        return artifact, runtime
