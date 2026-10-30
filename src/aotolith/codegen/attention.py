"""Attention kernels: online softmax over a tile of keys, or one pass of decode.

Two kernels, chosen by shape. Prefill attention tiles the keys and keeps a
running softmax denominator, which is what makes a long sequence fit in
threadgroup memory; decode attention has a single query, so it needs no
rescaling and one threadgroup per head is enough.

The mask is applied by offsetting the score rather than branching, so the helpers
that build that offset (``_attention_mask_offset_expression`` and the clamp
predicate) are part of this family. ``_zero_fully_masked_rows`` is the one place
the fully-masked case is repaired, because a row with no visible key would
otherwise divide by a zero denominator.
"""

from __future__ import annotations

import mlx.core as mx
from dataclasses import dataclass
from math import ceil, prod
from typing import Any
from ..ir import DType
from ..ir import Graph
from ..ir import Operation
from .dtypes import _MLX_DTYPES
from . import msl
from .msl import KernelArtifact, _MPP_HEADER, _MSL_HEADER
from ..target import apple as target
from ..target.apple import (
    _ATTENTION_MAX_SHARED_BYTES,
    _DECODE_ATTENTION_DTYPES,
    _DECODE_ATTENTION_MAX_KEYS,
    _DECODE_ATTENTION_THREADS,
    _MPP_ATTENTION_KEY_TILE,
    _attention_shared_memory_bytes,
    _native_attention_mask_supported,
    _select_mpp_attention_warps,
    _supports_mpp_attention,
)

_ATTENTION_LONG_SEQUENCE_THRESHOLD = 768


def _attention_key_rows_need_clamp(
    key_length: int,
    key_tile: int,
    query_tile: int,
    causal: bool,
) -> bool:
    """Say whether the last key tile can run past ``key_length``.

    The tile loops read ``key_base + [0, key_tile)``, so every access stays in
    bounds exactly when the loop bound is a multiple of the tile. Non-causal
    attention uses ``key_length`` itself; causal attention stops at
    ``min(key_length, query_block * query_tile + query_tile)``, which stays
    aligned only while the query tile is a multiple of the key tile. When the
    bound is aligned the per-element ``metal::min`` clamp is provably the
    identity and is dropped, which measured 1.02x-1.13x faster than emitting it
    unconditionally.
    """

    if key_length % key_tile != 0:
        return True
    if causal and query_tile % key_tile != 0:
        return True
    return False


def _emit_mpp_attention_source(
    *,
    query_length: int,
    key_length: int,
    query_heads: int,
    key_value_heads: int,
    query_head_dim: int,
    value_head_dim: int,
    warps: int,
    scale: float,
    has_mask: bool,
    mask_shape: tuple[int, ...] | None,
    mask_is_boolean: bool,
    causal: bool,
    rank: int,
) -> str:
    """Emit a register-resident FlashAttention pass on Apple tensor cores.

    Each simdgroup owns a 16-row query fragment and the full value width, so the
    online-softmax accumulators never leave registers. ``Q`` is staged once in
    threadgroup memory; ``K``/``V`` stream straight from device memory.
    """

    query_tile = warps * 16
    key_tile = _MPP_ATTENTION_KEY_TILE
    dimension_fragments = query_head_dim // 16
    value_pairs = value_head_dim // 32
    query_blocks = ceil(query_length / query_tile)
    group_size = query_heads // key_value_heads
    threads = warps * 32
    scores_per_half = 4

    if rank == 4:
        query_batch_head = f"(batch * {query_heads}u + head)"
        key_value_batch_head = f"(batch * {key_value_heads}u + head / {group_size}u)"
        output_batch_head = query_batch_head
    else:
        query_batch_head = "batch"
        key_value_batch_head = "batch"
        output_batch_head = "batch"

    lines: list[str] = []
    append = lines.append
    append("uint tid = thread_position_in_threadgroup.x;")
    append("uint lane = tid % 32u;")
    append("uint sg = tid / 32u;")
    append("uint qid = lane >> 2u;")
    append("uint fm = (qid & 4u) | ((lane >> 1u) & 3u);")
    append("uint fn = ((qid & 2u) | (lane & 1u)) * 4u;")
    append("uint query_index = threadgroup_position_in_grid.x;")
    append(f"uint batch = query_index / {query_heads * query_blocks}u;")
    append(f"uint batch_index = query_index % {query_heads * query_blocks}u;")
    append(f"uint head = batch_index / {query_blocks}u;")
    append(f"uint query_block = batch_index % {query_blocks}u;")
    append(f"threadgroup T tile_q[{query_tile * query_head_dim}];")
    append(
        f"for (uint index = tid; index < {query_tile * query_head_dim}u;"
        f" index += {threads}u) {{"
    )
    append(f"    uint row = index / {query_head_dim}u;")
    append(f"    uint dimension = index % {query_head_dim}u;")
    append(
        f"    uint position = metal::min(query_block * {query_tile}u + row,"
        f" {query_length - 1}u);"
    )
    append(
        f"    tile_q[index] = in_0[({query_batch_head} * {query_length}u"
        f" + position) * {query_head_dim}u + dimension];"
    )
    append("}")
    append("threadgroup_barrier(mem_flags::mem_threadgroup);")

    # The descriptor always declares the transposed right operand; the physical
    # row-major layout is absorbed by the per-operand index expressions.
    for tag in ("qk", "pv"):
        append(
            f"constexpr auto tsc_{tag}_descriptor = "
            "mpp::tensor_ops::matmul2d_descriptor("
        )
        append("    16, 32, 16, false, true, true,")
        append("    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);")
        append(
            f"mpp::tensor_ops::matmul2d<tsc_{tag}_descriptor,"
            " metal::execution_simdgroup> tsc_"
            f"{tag};"
        )
        append(
            f"using tsc_{tag}_at = decltype("
            f"tsc_{tag}.get_left_input_cooperative_tensor<T, T, float>());"
        )
        append(
            f"using tsc_{tag}_bt = decltype("
            f"tsc_{tag}.get_right_input_cooperative_tensor<T, T, float>());"
        )
        append(
            f"using tsc_{tag}_ct = decltype(tsc_{tag}."
            f"get_destination_cooperative_tensor<metal::remove_addrspace_t<tsc_{tag}_at>,"
            f" metal::remove_addrspace_t<tsc_{tag}_bt>, float>());"
        )

    for pair in range(value_pairs):
        append(
            f"tsc_pv_ct tsc_acc_{pair} = tsc_pv."
            "get_destination_cooperative_tensor<"
            "metal::remove_addrspace_t<tsc_pv_at>,"
            " metal::remove_addrspace_t<tsc_pv_bt>, float>();"
        )
        append(f"for (uint e = 0u; e < 16u; ++e) {{ tsc_acc_{pair}[e] = 0.0f; }}")

    for half in range(2):
        append(f"float running_max_{half} = -INFINITY;")
        append(f"float running_sum_{half} = 0.0f;")

    if causal:
        append(
            f"uint key_limit = metal::min({key_length}u,"
            f" query_block * {query_tile}u + {query_tile}u);"
        )
    else:
        append(f"uint key_limit = {key_length}u;")

    append(
        f"for (uint key_base = 0u; key_base < key_limit; key_base += {key_tile}u) {{"
    )
    append(
        f"    const device T* key_head = in_1 + {key_value_batch_head}"
        f" * {key_length * query_head_dim}u;"
    )
    append(
        f"    const device T* value_head = in_2 + {key_value_batch_head}"
        f" * {key_length * value_head_dim}u;"
    )

    # --- Q @ K^T ---------------------------------------------------------
    append(
        "    tsc_qk_ct tsc_scores = tsc_qk."
        "get_destination_cooperative_tensor<"
        "metal::remove_addrspace_t<tsc_qk_at>,"
        " metal::remove_addrspace_t<tsc_qk_bt>, float>();"
    )
    append("    for (uint e = 0u; e < 16u; ++e) { tsc_scores[e] = 0.0f; }")
    for half in range(2):
        append(f"    uint query_row_{half} = sg * 16u + {half * 8}u + fm;")
        append(
            f"    uint query_pos_{half} = query_block * {query_tile}u"
            f" + query_row_{half};"
        )
    clamp_key_rows = _attention_key_rows_need_clamp(
        key_length, key_tile, query_tile, causal
    )
    for row in range(4):
        expression = f"key_base + {row * 8}u + fm"
        if clamp_key_rows:
            expression = f"metal::min(key_base + {row * 8}u + fm, {key_length - 1}u)"
        append(f"    uint key_row_{row} = {expression};")
    append("    vec<T, 4> tsc_vector;")
    for fragment in range(dimension_fragments):
        append("    {")
        append("        tsc_qk_at tsc_q; tsc_qk_bt tsc_k;")
        for half in range(2):
            append(
                "        tsc_vector = *reinterpret_cast<"
                f"const threadgroup vec<T, 4>*>(tile_q + query_row_{half}"
                f" * {query_head_dim}u + {fragment * 16}u + fn);"
            )
            for element in range(4):
                append(f"        tsc_q[{half * 4 + element}] = tsc_vector[{element}];")
        for nf in range(2):
            for half in range(2):
                row = nf * 2 + half
                append(
                    "        tsc_vector = *reinterpret_cast<"
                    f"const device vec<T, 4>*>(key_head + key_row_{row}"
                    f" * {query_head_dim}u + {fragment * 16}u + fn);"
                )
                for element in range(4):
                    append(
                        f"        tsc_k[{nf * 8 + half * 4 + element}]"
                        f" = tsc_vector[{element}];"
                    )
        append("        tsc_qk.run(tsc_q, tsc_k, tsc_scores);")
        append("    }")

    # --- scale, mask, causal --------------------------------------------
    mask_offset_expression = (
        _attention_mask_offset_expression(rank, mask_shape)
        if has_mask and mask_shape is not None
        else ""
    )
    append("    float scores[16];")
    for half in range(2):
        if has_mask and mask_shape is not None:
            append(
                f"    uint mask_row_{half} = metal::min(query_pos_{half},"
                f" {query_length - 1}u);"
            )
    for element in range(16):
        nf = element >> 3
        half = (element >> 2) & 1
        offset = element & 3
        append(f"    scores[{element}] = tsc_scores[{element}] * {float(scale)!r};")
        if has_mask and mask_shape is not None:
            mask_offset = mask_offset_expression.replace(
                "query_pos", f"mask_row_{half}"
            )
            append("    {")
            append(f"        uint key = key_base + {nf * 16 + offset}u + fn;")
            if mask_is_boolean:
                append(
                    f"        if (!in_3[{mask_offset}])"
                    f" {{ scores[{element}] = -INFINITY; }}"
                )
            else:
                append(f"        scores[{element}] += float(in_3[{mask_offset}]);")
            append("    }")
    if key_length % key_tile != 0:
        append(f"    if (key_base + {key_tile}u > {key_length}u) {{")
        for element in range(16):
            nf = element >> 3
            offset = element & 3
            append(
                f"        if (key_base + {nf * 16 + offset}u + fn"
                f" >= {key_length}u) {{ scores[{element}] = -INFINITY; }}"
            )
        append("    }")
    if causal:
        append(
            f"    if (key_base + {key_tile}u > query_block * {query_tile}u"
            f" + sg * 16u) {{"
        )
        for element in range(16):
            nf = element >> 3
            half = (element >> 2) & 1
            offset = element & 3
            append(
                f"        if (key_base + {nf * 16 + offset}u + fn"
                f" > query_pos_{half}) {{ scores[{element}] = -INFINITY; }}"
            )
        append("    }")

    # --- online softmax --------------------------------------------------
    for half in range(2):
        elements = [
            nf * 8 + half * scores_per_half + offset
            for nf in range(2)
            for offset in range(scores_per_half)
        ]
        local_max = f"scores[{elements[0]}]"
        for element in elements[1:]:
            local_max = f"metal::max({local_max}, scores[{element}])"
        append(f"    float block_max_{half} = {local_max};")
        append(
            f"    block_max_{half} = metal::max(block_max_{half},"
            f" simd_shuffle(block_max_{half}, (uint)(lane ^ 1u)));"
        )
        append(
            f"    block_max_{half} = metal::max(block_max_{half},"
            f" simd_shuffle(block_max_{half}, (uint)(lane ^ 8u)));"
        )
        append(
            f"    float new_max_{half} ="
            f" metal::max(running_max_{half}, block_max_{half});"
        )
        append(
            f"    float alpha_{half} = running_max_{half} == -INFINITY"
            f" ? 0.0f : metal::fast::exp(running_max_{half} - new_max_{half});"
        )
        append(f"    float block_sum_{half} = 0.0f;")
        for element in elements:
            append(
                f"    {{ float p = scores[{element}] == -INFINITY ? 0.0f"
                f" : metal::fast::exp(scores[{element}] - new_max_{half});"
                f" scores[{element}] = p; block_sum_{half} += p; }}"
            )
        append(
            f"    block_sum_{half} += simd_shuffle(block_sum_{half},"
            f" (uint)(lane ^ 1u));"
        )
        append(
            f"    block_sum_{half} += simd_shuffle(block_sum_{half},"
            f" (uint)(lane ^ 8u));"
        )
        append(
            f"    running_sum_{half} = running_sum_{half} * alpha_{half}"
            f" + block_sum_{half};"
        )
        append(f"    running_max_{half} = new_max_{half};")

    append("    if (alpha_0 != 1.0f || alpha_1 != 1.0f) {")
    for pair in range(value_pairs):
        for element in range(16):
            half = (element >> 2) & 1
            append(f"        tsc_acc_{pair}[{element}] *= alpha_{half};")
    append("    }")

    # --- P @ V -----------------------------------------------------------
    for fragment in range(2):
        append("    {")
        append("        tsc_pv_at tsc_p;")
        for half in range(2):
            for element in range(4):
                append(
                    f"        tsc_p[{half * 4 + element}] = (T)scores["
                    f"{fragment * 8 + half * 4 + element}];"
                )
        for pair in range(value_pairs):
            append("        {")
            append("            tsc_pv_bt tsc_v;")
            for half in range(2):
                for nf in range(2):
                    for element in range(4):
                        row = f"{fragment * 16 + element}u + fn"
                        if clamp_key_rows:
                            row = f"metal::min(key_base + {row}, {key_length - 1}u)"
                        else:
                            row = f"(key_base + {row})"
                        append(
                            f"            tsc_v[{nf * 8 + half * 4 + element}]"
                            f" = value_head[{row}"
                            f" * {value_head_dim}u"
                            f" + {pair * 32 + nf * 16 + half * 8}u + fm];"
                        )
            append(f"            tsc_pv.run(tsc_p, tsc_v, tsc_acc_{pair});")
            append("        }")
        append("    }")

    append("}")

    # --- normalize and store --------------------------------------------
    for half in range(2):
        append("if (")
        append(
            f"    query_block * {query_tile}u + sg * 16u + {half * 8}u + fm"
            f" < {query_length}u) {{"
        )
        append(
            f"    float inverse_sum = running_sum_{half} == 0.0f"
            " ? 0.0f : 1.0f / running_sum_"
            f"{half};"
        )
        append(
            f"    uint output_row = (({output_batch_head} * {query_length}u"
            f" + query_block * {query_tile}u + sg * 16u + {half * 8}u + fm)"
            f" * {value_head_dim}u);"
        )
        for pair in range(value_pairs):
            for nf in range(2):
                for element in range(4):
                    index = nf * 8 + half * 4 + element
                    append(
                        f"    out[output_row + {pair * 32 + nf * 16 + element}u"
                        f" + fn] = (T)(tsc_acc_{pair}[{index}] * inverse_sum);"
                    )
        append("}")
    return "\n".join(lines)


def _attention_mask_offset_expression(
    query_rank: int, mask_shape: tuple[int, ...]
) -> str:
    if query_rank == 4:
        coordinates = ("batch", "head", "query_pos", "key")
    else:
        coordinates = ("batch", "query_pos", "key")
    offset = len(coordinates) - len(mask_shape)
    if offset < 0:
        raise ValueError(f"attention mask rank exceeds query rank: {mask_shape}")

    terms: list[str] = []
    for axis, dimension in enumerate(mask_shape):
        if dimension == 1:
            continue
        coordinate = coordinates[axis + offset]
        stride = prod(mask_shape[axis + 1 :])
        terms.append(coordinate if stride == 1 else f"{coordinate} * {stride}u")
    return " + ".join(terms) if terms else "0u"


@dataclass(frozen=True, slots=True)
class _DecodeAttentionConfig:
    """Static description of one single-query attention launch."""

    threads: int
    batch: int
    query_heads: int
    key_value_heads: int
    group_size: int
    key_length: int
    query_head_dim: int
    value_head_dim: int
    scale: float
    mask_shape: tuple[int, ...] | None
    mask_is_boolean: bool
    dtype: DType


def _decode_attention_config(
    graph: Graph, op: Operation
) -> _DecodeAttentionConfig | None:
    """Return a launch plan when a one-query attention can use one kernel.

    Prefill attention needs the tiled online-softmax path. A decode step asks
    every head for a single output row, so the whole head fits in one
    threadgroup: scores live in threadgroup memory, the softmax is a two-level
    reduction, and the value accumulation has no online rescaling. That removes
    the per-layer gather/spread work the generic kernels need for a one-row
    query.
    """

    query_type = graph.value(op.inputs[0]).type
    key_type = graph.value(op.inputs[1]).type
    value_type = graph.value(op.inputs[2]).type
    output_type = graph.value(op.outputs[0]).type
    if len(query_type.shape) != 4 or output_type.dtype not in _DECODE_ATTENTION_DTYPES:
        return None
    batch, query_heads, query_length, query_head_dim = query_type.shape
    key_value_heads, key_length = key_type.shape[1], key_type.shape[2]
    value_head_dim = value_type.shape[-1]
    if query_length != 1 or op.attrs.get("causal"):
        return None
    if key_length < 1 or key_length > _DECODE_ATTENTION_MAX_KEYS:
        return None
    if query_head_dim > 256 or value_head_dim > 256:
        return None
    if batch < 1 or query_heads < 1 or key_value_heads < 1:
        return None
    if query_heads % key_value_heads != 0:
        return None
    has_mask = bool(op.attrs.get("has_mask"))
    mask_shape: tuple[int, ...] | None = None
    mask_is_boolean = False
    if has_mask:
        mask_type = graph.value(op.inputs[3]).type
        mask_shape = mask_type.shape
        mask_is_boolean = mask_type.dtype == DType.BOOL
        if not mask_is_boolean and mask_type.dtype != query_type.dtype:
            return None
        if len(mask_shape) > 4:
            return None
    threads = _DECODE_ATTENTION_THREADS
    shared_bytes = 4 * key_length + 4 * threads + 4 * query_head_dim
    if shared_bytes > _ATTENTION_MAX_SHARED_BYTES:
        return None
    return _DecodeAttentionConfig(
        threads=threads,
        batch=batch,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        group_size=query_heads // key_value_heads,
        key_length=key_length,
        query_head_dim=query_head_dim,
        value_head_dim=value_head_dim,
        scale=float(op.attrs["scale"]),
        mask_shape=mask_shape,
        mask_is_boolean=mask_is_boolean,
        dtype=output_type.dtype,
    )


def _emit_decode_attention_source(config: _DecodeAttentionConfig) -> str:
    """Emit the one-query attention kernel body for a static head shape."""

    threads = config.threads
    key_length = config.key_length
    query_dim = config.query_head_dim
    value_dim = config.value_head_dim
    query_heads = config.query_heads
    key_value_heads = config.key_value_heads
    splits = threads // value_dim if threads % value_dim == 0 else 1
    indent = "    "
    lines = [
        "uint tid = thread_position_in_threadgroup.x;",
        "uint head = threadgroup_position_in_grid.x;",
        "uint batch = threadgroup_position_in_grid.y;",
        "uint query_pos = 0u;",
        f"uint key_value_head = head / {config.group_size}u;",
        f"const device T* query = in_0 "
        f"+ (batch * {query_heads}u + head) * {query_dim}u;",
        f"const device T* key_base = in_1 "
        f"+ (batch * {key_value_heads}u + key_value_head) "
        f"* {key_length * query_dim}u;",
        f"const device T* value_base = in_2 "
        f"+ (batch * {key_value_heads}u + key_value_head) "
        f"* {key_length * value_dim}u;",
        f"threadgroup float scores[{key_length}];",
        f"threadgroup float scratch[{threads}];",
        f"threadgroup float query_shared[{query_dim}];",
        f"for (uint dimension = tid; dimension < {query_dim}u; "
        f"dimension += {threads}u) {{",
        f"{indent}query_shared[dimension] = float(query[dimension]);",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
        f"{indent}const device T* row = key_base + key * {query_dim}u;",
        f"{indent}float dot = 0.0f;",
        f"{indent}for (uint dimension = 0u; dimension < {query_dim}u; ++dimension) {{",
        f"{indent * 2}dot = fma(query_shared[dimension], float(row[dimension]), dot);",
        f"{indent}}}",
        f"{indent}float score = dot * float({config.scale!r});",
    ]
    if config.mask_shape is not None:
        mask_offset = _attention_mask_offset_expression(4, config.mask_shape)
        if config.mask_is_boolean:
            lines.append(f"{indent}score = in_3[{mask_offset}] ? score : -INFINITY;")
        else:
            lines.append(f"{indent}score += float(in_3[{mask_offset}]);")
    lines.extend(
        [
            f"{indent}scores[key] = score;",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "float local_max = -INFINITY;",
            f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
            f"{indent}local_max = metal::max(local_max, scores[key]);",
            "}",
            "scratch[tid] = local_max;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"for (uint stride = {threads // 2}u; stride > 0u; stride >>= 1) {{",
            f"{indent}if (tid < stride) {{",
            f"{indent * 2}scratch[tid] = metal::max(scratch[tid], "
            "scratch[tid + stride]);",
            f"{indent}}}",
            f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
            # A fully masked head has no finite maximum; shifting by zero keeps
            # every exponential underflowing to zero instead of producing NaN.
            "float row_max = metal::isinf(scratch[0]) ? 0.0f : scratch[0];",
            # Every thread must read the reduction result before the scratch
            # buffer is reused for the softmax sum, otherwise a leading thread
            # can overwrite it while its peers are still loading the maximum.
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "float local_sum = 0.0f;",
            f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
            f"{indent}local_sum += metal::exp(scores[key] - row_max);",
            "}",
            "scratch[tid] = local_sum;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"for (uint stride = {threads // 2}u; stride > 0u; stride >>= 1) {{",
            f"{indent}if (tid < stride) {{",
            f"{indent * 2}scratch[tid] += scratch[tid + stride];",
            f"{indent}}}",
            f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
            "float inv_total = scratch[0] > 0.0f ? 1.0f / scratch[0] : 0.0f;",
            f"for (uint key = tid; key < {key_length}u; key += {threads}u) {{",
            f"{indent}scores[key] = metal::exp(scores[key] - row_max) * inv_total;",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]
    )
    if splits > 1:
        lines.extend(
            [
                f"uint part = tid / {value_dim}u;",
                f"uint dimension = tid - part * {value_dim}u;",
                f"uint begin = ({key_length}u * part) / {splits}u;",
                f"uint end = ({key_length}u * (part + 1u)) / {splits}u;",
                "float accumulator = 0.0f;",
                "for (uint key = begin; key < end; ++key) {",
                f"{indent}accumulator = fma(scores[key], float(value_base[key * "
                f"{value_dim}u + dimension]), accumulator);",
                "}",
                "scratch[tid] = accumulator;",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                f"if (tid < {value_dim}u) {{",
                f"{indent}float total = 0.0f;",
                f"{indent}for (uint split = 0u; split < {splits}u; ++split) {{",
                f"{indent * 2}total += scratch[split * {value_dim}u + tid];",
                f"{indent}}}",
                f"{indent}out[(batch * {query_heads}u + head) * {value_dim}u "
                "+ tid] = (T)total;",
                "}",
            ]
        )
    else:
        lines.extend(
            [
                f"for (uint dimension = tid; dimension < {value_dim}u; "
                f"dimension += {threads}u) {{",
                f"{indent}float accumulator = 0.0f;",
                f"{indent}for (uint key = 0u; key < {key_length}u; ++key) {{",
                f"{indent * 2}accumulator = fma(scores[key], float(value_base[key "
                f"* {value_dim}u + dimension]), accumulator);",
                f"{indent}}}",
                f"{indent}out[(batch * {query_heads}u + head) * {value_dim}u "
                "+ dimension] = (T)accumulator;",
                "}",
            ]
        )
    return "\n".join(f"        {line}" for line in lines)


def _zero_fully_masked_rows(mask: mx.array, output: mx.array) -> mx.array:
    score_shape = output.shape[:-1] + (mask.shape[-1],)
    broadcast_mask = mx.broadcast_to(mask, score_shape)
    if broadcast_mask.dtype == mx.bool_:
        all_masked = mx.all(mx.logical_not(broadcast_mask), axis=-1, keepdims=True)
    else:
        all_masked = mx.all(broadcast_mask == -mx.inf, axis=-1, keepdims=True)
    return mx.where(all_masked, mx.array(0, dtype=output.dtype), output)


class AttentionLowering:
    def _compile_mpp_attention(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        query_type = graph.value(op.inputs[0]).type
        key_type = graph.value(op.inputs[1]).type
        value_type = graph.value(op.inputs[2]).type
        output_type = graph.value(op.outputs[0]).type
        has_mask = bool(op.attrs.get("has_mask"))
        mask_type = graph.value(op.inputs[3]).type if has_mask else None
        mask_shape = mask_type.shape if mask_type is not None else None
        if not _supports_mpp_attention(
            query_type, key_type, value_type, output_type, mask_shape
        ):
            return None
        if not target._mpp_supported():
            return None

        rank = len(query_type.shape)
        query_length = query_type.shape[-2]
        key_length = key_type.shape[-2]
        query_heads = query_type.shape[-3] if rank == 4 else 1
        key_value_heads = key_type.shape[-3] if rank == 4 else 1
        query_head_dim = query_type.shape[-1]
        value_head_dim = value_type.shape[-1]
        warps = _select_mpp_attention_warps(query_length)
        query_tile = warps * 16
        query_blocks = ceil(query_length / query_tile)
        batch = query_type.shape[0]
        threads = warps * 32
        mask_is_boolean = mask_type is not None and mask_type.dtype == DType.BOOL

        source = _emit_mpp_attention_source(
            query_length=query_length,
            key_length=key_length,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            query_head_dim=query_head_dim,
            value_head_dim=value_head_dim,
            warps=warps,
            scale=float(op.attrs.get("scale", 1.0)),
            has_mask=has_mask,
            mask_shape=mask_shape,
            mask_is_boolean=mask_is_boolean,
            causal=bool(op.attrs.get("causal")),
            rank=rank,
        )

        input_names = (
            ("in_0", "in_1", "in_2", "in_3") if has_mask else ("in_0", "in_1", "in_2")
        )
        name = (
            f"metal_attention_tensorcore_{query_tile}x{_MPP_ATTENTION_KEY_TILE}_"
            f"{threads}_{index}"
        )
        kernel = msl._metal_kernel(
            name,
            input_names,
            ("out",),
            source,
            _MPP_HEADER,
        )
        query_count = batch * query_heads * query_blocks
        grid = (query_count * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]
        template = [("T", dtype)]
        output_shapes = [output_type.shape]
        output_dtypes = [dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            return kernel(
                inputs=inputs,
                template=template,
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=_MPP_HEADER,
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind="attention",
        )
        return artifact, runtime

    def _compile_decode_attention(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        """Emit the single-query attention kernel when the shape allows it."""

        config = _decode_attention_config(graph, op)
        if config is None:
            return None
        output_type = graph.value(op.outputs[0]).type
        source = _emit_decode_attention_source(config)
        name = (
            f"metal_decode_attention_{config.query_heads}"
            f"_{config.key_value_heads}_{config.key_length}"
            f"_{config.query_head_dim}_{config.value_head_dim}_{index}"
        )
        input_names = tuple(f"in_{position}" for position in range(len(op.inputs)))
        kernel = msl._metal_kernel(
            name,
            input_names,
            ("out",),
            source,
            _MSL_HEADER,
        )
        dtype = _MLX_DTYPES[config.dtype]
        # One threadgroup per (head, batch) pair: grid.y is the batch count, not
        # batch * threads, otherwise the runtime launches threads duplicate
        # threadgroups that write past the end of the output buffer.
        grid = (
            config.query_heads * config.threads,
            config.batch,
            1,
        )
        threadgroup = (config.threads, 1, 1)

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
            header=_MSL_HEADER,
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind="attention",
        )
        return artifact, runtime

    def _compile_attention(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        """Emit a FlashAttention-style tiled online-softmax kernel."""

        decode = self._compile_decode_attention(graph, op, index)
        if decode is not None:
            return decode

        tensor_core = self._compile_mpp_attention(graph, op, index)
        if tensor_core is not None:
            return tensor_core

        query_type = graph.value(op.inputs[0]).type
        key_type = graph.value(op.inputs[1]).type
        value_type = graph.value(op.inputs[2]).type
        output_type = graph.value(op.outputs[0]).type
        query_shape = query_type.shape
        key_shape = key_type.shape
        value_shape = value_type.shape
        rank = len(query_shape)
        query_head_dim = query_shape[-1]
        value_head_dim = value_shape[-1]
        if query_head_dim > 256 or value_head_dim > 256:
            return None

        has_mask = bool(op.attrs.get("has_mask"))
        mask_type = graph.value(op.inputs[3]).type if has_mask else None
        mask_shape = mask_type.shape if mask_type is not None else None
        mask_is_boolean = mask_type is not None and mask_type.dtype == DType.BOOL
        if has_mask and not _native_attention_mask_supported(rank, mask_shape):
            return None

        batch, query_length, key_length = query_shape[0], query_shape[-2], key_shape[-2]
        query_heads = query_shape[-3] if rank == 4 else 1
        key_value_heads = key_shape[-3] if rank == 4 else 1
        group_size = query_heads // key_value_heads
        key_tile = 32
        low_precision_attention = output_type.dtype in {DType.F16, DType.BF16}
        use_simdgroup_qk = (
            low_precision_attention and query_head_dim % 8 == 0 and query_length >= 8
        )
        use_simdgroup_pv = (
            low_precision_attention and value_head_dim % 8 == 0 and query_length >= 8
        )
        query_tile = 1 if query_length == 1 else 8
        if use_simdgroup_qk and use_simdgroup_pv and query_length > 8:
            candidates = ((16, 32),)
            if query_length >= _ATTENTION_LONG_SEQUENCE_THRESHOLD:
                candidates = ((24, 64), (16, 64), (24, 32), (16, 32))
            for candidate_query_tile, candidate_key_tile in candidates:
                candidate_pv_value_tile = (
                    min(value_head_dim, 64)
                    if candidate_key_tile == 64
                    else value_head_dim
                )
                shared_bytes = _attention_shared_memory_bytes(
                    candidate_query_tile,
                    candidate_key_tile,
                    query_head_dim,
                    value_head_dim,
                    candidate_pv_value_tile,
                )
                if shared_bytes <= _ATTENTION_MAX_SHARED_BYTES:
                    query_tile = candidate_query_tile
                    key_tile = candidate_key_tile
                    break
        pv_value_tile = min(value_head_dim, 64) if key_tile == 64 else value_head_dim
        threads = query_tile * 32
        query_blocks = ceil(query_length / query_tile)
        query_count = batch * query_heads * query_blocks

        if rank == 4:
            query_batch_head = f"(batch * {query_heads}u + head)"
            key_value_batch_head = (
                f"(batch * {key_value_heads}u + head / {group_size}u)"
            )
        else:
            query_batch_head = "batch"
            key_value_batch_head = "batch"

        if use_simdgroup_qk and use_simdgroup_pv and key_tile == 64:
            tile_alignment = "alignas(32) "
            load_vector_width = (
                16 if query_head_dim % 16 == 0 and value_head_dim % 16 == 0 else 8
            )
            query_load_source = f"""
            for (uint vector_index = tid;
                 vector_index < {query_tile * query_head_dim // load_vector_width}u;
                 vector_index += {threads}u) {{
                uint index = vector_index * {load_vector_width}u;
                uint query_row = index / {query_head_dim}u;
                uint dimension = index % {query_head_dim}u;
                uint position = query_block * {query_tile}u + query_row;
                threadgroup vec<T, {load_vector_width}>* destination =
                    reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(
                        tile_q + index);
                if (position < {query_length}u) {{
                    const device vec<T, {load_vector_width}>* source =
                        reinterpret_cast<const device vec<T, {load_vector_width}>*>(
                            in_0 + ({query_batch_head} * {query_length}u + position)
                                * {query_head_dim}u + dimension);
                    *destination = *source;
                }} else {{
                    *destination = vec<T, {load_vector_width}>(T(0));
                }}
            }}
    """
            key_load_source = f"""
                for (uint vector_index = tid;
                     vector_index < {key_tile * query_head_dim // load_vector_width}u;
                     vector_index += {threads}u) {{
                    uint index = vector_index * {load_vector_width}u;
                    uint key_row = index / {query_head_dim}u;
                    uint dimension = index % {query_head_dim}u;
                    uint key = key_base + key_row;
                    threadgroup vec<T, {load_vector_width}>* destination =
                        reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(
                            tile_k + index);
                    if (key < {key_length}u) {{
                        const device vec<T, {load_vector_width}>* source =
                            reinterpret_cast<const device vec<T, {load_vector_width}>*>(
                                in_1 + ({key_value_batch_head} * {key_length}u + key)
                                    * {query_head_dim}u + dimension);
                        *destination = *source;
                    }} else {{
                        *destination = vec<T, {load_vector_width}>(T(0));
                    }}
                }}
    """
            value_load_source = f"""
                for (uint vector_index = tid;
                     vector_index < {key_tile * value_head_dim // load_vector_width}u;
                     vector_index += {threads}u) {{
                    uint index = vector_index * {load_vector_width}u;
                    uint key_row = index / {value_head_dim}u;
                    uint dimension = index % {value_head_dim}u;
                    uint global_key = key_base + key_row;
                    threadgroup vec<T, {load_vector_width}>* destination =
                        reinterpret_cast<threadgroup vec<T, {load_vector_width}>*>(
                            tile_k + index);
                    if (global_key < {key_length}u) {{
                        const device vec<T, {load_vector_width}>* source =
                            reinterpret_cast<const device vec<T, {load_vector_width}>*>(
                                in_2 + ({key_value_batch_head} * {key_length}u
                                    + global_key) * {value_head_dim}u + dimension);
                        *destination = *source;
                    }} else {{
                        *destination = vec<T, {load_vector_width}>(T(0));
                    }}
                }}
    """
        else:
            tile_alignment = ""
            query_load_source = f"""
            for (uint index = tid; index < {query_tile * query_head_dim}u;
                 index += {threads}u) {{
                uint query_row = index / {query_head_dim}u;
                uint dimension = index % {query_head_dim}u;
                uint position = query_block * {query_tile}u + query_row;
                tile_q[index] = position < {query_length}u
                    ? in_0[({query_batch_head} * {query_length}u + position)
                        * {query_head_dim}u + dimension]
                    : T(0);
            }}
    """
            key_load_source = f"""
                for (uint index = tid; index < {key_tile * query_head_dim}u;
                     index += {threads}u) {{
                    uint key_row = index / {query_head_dim}u;
                    uint dimension = index % {query_head_dim}u;
                    uint key = key_base + key_row;
                    tile_k[index] = key < {key_length}u
                        ? in_1[({key_value_batch_head} * {key_length}u + key)
                            * {query_head_dim}u + dimension]
                        : T(0);
                }}
    """
            value_load_source = f"""
                for (uint index = tid; index < {key_tile * value_head_dim}u;
                     index += {threads}u) {{
                    uint key_row = index / {value_head_dim}u;
                    uint dimension = index % {value_head_dim}u;
                    uint global_key = key_base + key_row;
                    tile_k[index] = global_key < {key_length}u
                        ? in_2[({key_value_batch_head} * {key_length}u + global_key)
                            * {value_head_dim}u + dimension]
                        : T(0);
                }}
    """

        score_expression = f"dot * {float(op.attrs['scale'])!r}"
        if has_mask:
            mask_offset = _attention_mask_offset_expression(rank, mask_shape)
            if mask_is_boolean:
                score_expression = (
                    f"in_3[{mask_offset}] ? ({score_expression}) : -INFINITY"
                )
            else:
                score_expression += f" + float(in_3[{mask_offset}])"
        if op.attrs.get("causal"):
            score_expression = f"key > query_pos ? -INFINITY : ({score_expression})"

        if use_simdgroup_qk:
            simdgroup_type = {
                DType.F16: "half",
                DType.BF16: "bfloat",
            }[output_type.dtype]
            query_fragments = query_tile // 8
            key_fragments = key_tile // 8
            qk_group_count = query_fragments * key_fragments
            qk_steps = "\n".join(
                f"""                simdgroup_load(qk_frag_q, tile_q
                        + query_fragment * 8u * {query_head_dim}u + {dimension}u,
                        {query_head_dim}, 0, false);
                    simdgroup_load(qk_frag_k, tile_k
                        + key_fragment * 8u * {query_head_dim}u + {dimension}u,
                        {query_head_dim}, 0, true);
                    simdgroup_multiply_accumulate(
                        qk_accumulator, qk_frag_q, qk_frag_k, qk_accumulator);"""
                for dimension in range(0, query_head_dim, 8)
            )
            if key_tile == 32:
                qk_score_read_source = f"""
                uint key = key_base + lane;
                float score = -INFINITY;
                if (query_pos < {query_length}u && key < {key_length}u) {{
                    float dot = float(tile_scores[warp * {key_tile}u + lane]);
                    score = {score_expression};
                }}
    """
            else:
                qk_score_read_source = f"""
                uint key0 = key_base + lane;
                float score0 = -INFINITY;
                if (query_pos < {query_length}u && key0 < {key_length}u) {{
                    float dot = float(tile_scores[warp * {key_tile}u + lane]);
                    uint key = key0;
                    score0 = {score_expression};
                }}

                uint key1 = key0 + 32u;
                float score1 = -INFINITY;
                if (query_pos < {query_length}u && key1 < {key_length}u) {{
                    float dot = float(tile_scores[warp * {key_tile}u + lane + 32u]);
                    uint key = key1;
                    score1 = {score_expression};
                }}
    """
            qk_score_source = f"""
                if (warp < {qk_group_count}u) {{
                    uint query_fragment = warp / {key_fragments}u;
                    uint key_fragment = warp % {key_fragments}u;
                    simdgroup_float8x8 qk_accumulator =
                        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    simdgroup_{simdgroup_type}8x8 qk_frag_q;
                    simdgroup_{simdgroup_type}8x8 qk_frag_k;
    {qk_steps}
                    simdgroup_store(qk_accumulator,
                        tile_scores + query_fragment * 8u * {key_tile}u
                            + key_fragment * 8u,
                        {key_tile}, 0, false);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
    {qk_score_read_source}
    """
        else:
            qk_score_source = f"""
                uint key = key_base + lane;
                float score = -INFINITY;
                if (query_pos < {query_length}u && key < {key_length}u) {{
                    float dot = 0.0f;
                    for (uint dimension = 0u; dimension < {query_head_dim}u;
                         ++dimension) {{
                        dot = fma(float(tile_q[warp * {query_head_dim}u + dimension]),
                            float(tile_k[lane * {query_head_dim}u + dimension]), dot);
                    }}
                    score = {score_expression};
                }}
    """
        if use_simdgroup_pv:
            simdgroup_type = {
                DType.F16: "half",
                DType.BF16: "bfloat",
            }[output_type.dtype]
            query_fragments = query_tile // 8
            value_fragments = value_head_dim // 8
            pv_probability_declaration = ""
            if key_tile == 64:
                probability_storage = "tile_probabilities_low"
                pv_shared_declaration = ""
                pv_probability_store = f"""
                tile_probabilities_low[warp * {key_tile}u + lane] =
                    (T)probability0;
                tile_probabilities_low[warp * {key_tile}u + lane + 32u] =
                    (T)probability1;"""
            else:
                probability_storage = "tile_probabilities_low"
                pv_shared_declaration = f"""
            threadgroup T tile_probabilities_low[{query_tile * key_tile}];
            threadgroup float tile_pv[{query_tile * pv_value_tile}];"""
                pv_probability_store = f"""
                tile_probabilities_low[warp * {key_tile}u + lane] = (T)probability;"""
            pv_preload_source = value_load_source
            if key_tile == 64:
                value_fragments_per_chunk = pv_value_tile // 8
                pv_steps = "\n".join(
                    f"""                    simdgroup_load(pv_frag_probability,
                            {probability_storage} + output_row_fragment * 8u
                                * {key_tile}u + {tile_key}u,
                            {key_tile}, 0, false);
                        simdgroup_load(pv_frag_value, tile_k
                            + {tile_key * value_head_dim}u
                            + value_chunk_base
                            + output_column_fragment * 8u,
                            {value_head_dim}, 0, false);
                        simdgroup_multiply_accumulate(
                            pv_accumulator, pv_frag_probability, pv_frag_value,
                            pv_accumulator);"""
                    for tile_key in range(0, key_tile, 8)
                )
                pv_accumulation_source = f"""
                for (uint value_chunk_base = 0u;
                     value_chunk_base < {value_head_dim}u;
                     value_chunk_base += {pv_value_tile}u) {{
                    uint value_chunk_fragments = metal::min(
                        {value_fragments_per_chunk}u,
                        ({value_head_dim}u - value_chunk_base) / 8u);
                    for (uint output_fragment = warp;
                         output_fragment <
                             {query_fragments}u * value_chunk_fragments;
                         output_fragment += {threads // 32}u) {{
                        uint output_row_fragment =
                            output_fragment / value_chunk_fragments;
                        uint output_column_fragment =
                            output_fragment % value_chunk_fragments;
                        simdgroup_float8x8 pv_accumulator =
                            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                        simdgroup_{simdgroup_type}8x8 pv_frag_probability;
                        simdgroup_{simdgroup_type}8x8 pv_frag_value;
    {pv_steps}
                        simdgroup_store(pv_accumulator,
                            tile_pv + output_row_fragment * 8u * {pv_value_tile}u
                                + output_column_fragment * 8u,
                            {pv_value_tile}, 0, false);
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint component = 0u;
                         component < {ceil(value_head_dim / 32)}u; ++component) {{
                        uint dimension = lane + component * 32u;
                        if (query_pos < {query_length}u
                            && dimension >= value_chunk_base
                            && dimension < value_chunk_base + {pv_value_tile}u
                            && dimension < {value_head_dim}u) {{
                            accumulator[component] = accumulator[component] * alpha
                                + block_scale * tile_pv[warp * {pv_value_tile}u
                                    + dimension - value_chunk_base];
                        }}
                    }}
                    if (value_chunk_base + {pv_value_tile}u < {value_head_dim}u) {{
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }}
                }}
    """
            else:
                pv_fragment_count = query_fragments * value_fragments
                pv_steps = "\n".join(
                    f"""                simdgroup_load(pv_frag_probability,
                        {probability_storage} + output_row_fragment * 8u
                            * {key_tile}u + {tile_key}u, {key_tile}, 0, false);
                    simdgroup_load(pv_frag_value, tile_k
                        + {tile_key * value_head_dim}u
                        + output_column_fragment * 8u,
                        {value_head_dim}, 0, false);
                    simdgroup_multiply_accumulate(
                        pv_accumulator, pv_frag_probability, pv_frag_value,
                        pv_accumulator);"""
                    for tile_key in range(0, key_tile, 8)
                )
                pv_accumulation_source = f"""
                for (uint output_fragment = warp;
                     output_fragment < {pv_fragment_count}u;
                     output_fragment += {query_tile}u) {{
                    uint output_row_fragment = output_fragment / {value_fragments}u;
                    uint output_column_fragment = output_fragment % {value_fragments}u;
                    simdgroup_float8x8 pv_accumulator =
                        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    simdgroup_{simdgroup_type}8x8 pv_frag_probability;
                    simdgroup_{simdgroup_type}8x8 pv_frag_value;
    {pv_steps}
                    simdgroup_store(pv_accumulator,
                        tile_pv + output_row_fragment * 8u * {pv_value_tile}u
                            + output_column_fragment * 8u,
                        {pv_value_tile}, 0, false);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint component = 0u;
                     component < {ceil(value_head_dim / 32)}u; ++component) {{
                    uint dimension = lane + component * 32u;
                    if (query_pos < {query_length}u && dimension < {value_head_dim}u) {{
                        accumulator[component] = accumulator[component] * alpha
                            + block_scale
                                * tile_pv[warp * {pv_value_tile}u + dimension];
                    }}
                }}
    """
        else:
            pv_probability_declaration = f"""
            threadgroup float tile_probabilities[{query_tile * key_tile}];"""
            pv_shared_declaration = ""
            pv_probability_store = f"""
                tile_probabilities[warp * {key_tile}u + lane] = probability;"""
            pv_preload_source = ""
            pv_accumulation_source = f"""
                for (uint component = 0u;
                     component < {ceil(value_head_dim / 32)}u; ++component) {{
                    uint dimension = lane + component * 32u;
                    float tile_accumulator = 0.0f;
                    if (query_pos < {query_length}u && dimension < {value_head_dim}u) {{
                        for (uint tile_key = 0u; tile_key < {key_tile}u; ++tile_key) {{
                            uint global_key = key_base + tile_key;
                            if (global_key < {key_length}u) {{
                                tile_accumulator = fma(
                                    tile_probabilities[warp * {key_tile}u + tile_key],
                                    float(in_2[({key_value_batch_head} * {key_length}u
                                        + global_key) * {value_head_dim}u + dimension]),
                                    tile_accumulator);
                            }}
                        }}
                        accumulator[component] = accumulator[component] * alpha
                            + block_scale * tile_accumulator;
                    }}
                }}
    """
        if key_tile == 64:
            probability_float_offset = query_tile * key_tile // 2
            workspace_floats = max(
                query_tile * key_tile,
                probability_float_offset + query_tile * pv_value_tile,
            )
            score_storage_declaration = f"""
            threadgroup float tile_workspace[{workspace_floats}];
            threadgroup float* tile_scores = tile_workspace;
            threadgroup T* tile_probabilities_low =
                (threadgroup T*)tile_workspace;
            threadgroup float* tile_pv =
                tile_workspace + {probability_float_offset}u;"""
        else:
            score_storage_declaration = (
                f"threadgroup float tile_scores[{query_tile * key_tile}];"
            )
        score_read_barrier = (
            "\n            threadgroup_barrier(mem_flags::mem_threadgroup);"
            if key_tile == 64
            else ""
        )
        if key_tile == 32:
            softmax_source = """
                float block_max = simd_max(score);
                float new_max = metal::max(running_max, block_max);
                float alpha = running_max == -INFINITY
                    ? 0.0f
                    : metal::exp(running_max - new_max);
                float block_scale = block_max == -INFINITY
                    ? 0.0f
                    : metal::exp(block_max - new_max);
                float probability = score == -INFINITY
                    ? 0.0f
                    : metal::exp(score - block_max);
                float block_sum = simd_sum(probability);
                running_sum = running_sum * alpha + block_scale * block_sum;
                running_max = new_max;
    """
        else:
            softmax_source = """
                float block_max = simd_max(metal::max(score0, score1));
                float new_max = metal::max(running_max, block_max);
                float alpha = running_max == -INFINITY
                    ? 0.0f
                    : metal::exp(running_max - new_max);
                float block_scale = block_max == -INFINITY
                    ? 0.0f
                    : metal::exp(block_max - new_max);
                float probability0 = score0 == -INFINITY
                    ? 0.0f
                    : metal::exp(score0 - block_max);
                float probability1 = score1 == -INFINITY
                    ? 0.0f
                    : metal::exp(score1 - block_max);
                float block_sum = simd_sum(probability0 + probability1);
                running_sum = running_sum * alpha + block_scale * block_sum;
                running_max = new_max;
    """
        source = f"""
            uint tid = thread_position_in_threadgroup.x;
            uint lane = tid % 32u;
            uint warp = tid / 32u;
            uint query_index = threadgroup_position_in_grid.x;
            uint batch = query_index / {query_heads * query_blocks}u;
            uint query_remainder = query_index % {query_heads * query_blocks}u;
            uint head = query_remainder / {query_blocks}u;
            uint query_block = query_remainder % {query_blocks}u;
            uint query_pos = query_block * {query_tile}u + warp;
            {tile_alignment}threadgroup T tile_q[{query_tile * query_head_dim}];
            {tile_alignment}threadgroup T tile_k[{key_tile * max(query_head_dim, value_head_dim)}];
            {score_storage_declaration}{pv_probability_declaration}{pv_shared_declaration}

    {query_load_source}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float accumulator[{ceil(value_head_dim / 32)}] = {{0.0f}};
            float running_max = -INFINITY;
            float running_sum = 0.0f;

            for (uint key_base = 0u; key_base < {key_length}u;
                 key_base += {key_tile}u) {{
    {key_load_source}
                threadgroup_barrier(mem_flags::mem_threadgroup);
    {qk_score_source}{pv_preload_source}
    {softmax_source}
    {score_read_barrier}
    {pv_probability_store}
                threadgroup_barrier(mem_flags::mem_threadgroup);
    {pv_accumulation_source}
            }}

            if (query_pos < {query_length}u) {{
                for (uint component = 0u;
                     component < {ceil(value_head_dim / 32)}u; ++component) {{
                    uint dimension = lane + component * 32u;
                    if (dimension < {value_head_dim}u) {{
                        uint output_offset = ({query_batch_head} * {query_length}u
                            + query_pos) * {value_head_dim}u + dimension;
                        out[output_offset] = (T)(running_sum == 0.0f
                            ? 0.0f
                            : accumulator[component] / running_sum);
                    }}
                }}
            }}
            """
        input_names = (
            ("in_0", "in_1", "in_2", "in_3")
            if has_mask
            else (
                "in_0",
                "in_1",
                "in_2",
            )
        )
        name = f"metal_attention_flash_{query_tile}x{key_tile}_{threads}_{index}"
        kernel = msl._metal_kernel(
            name,
            input_names,
            ("out",),
            source,
            "",
        )
        grid = (query_count * threads, 1, 1)
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
            header="",
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind="attention",
        )
        return artifact, runtime
