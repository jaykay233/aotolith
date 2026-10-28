from __future__ import annotations

import textwrap
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, isinf, isnan, prod
from typing import Any

import mlx.core as mx

from ..ir import DType, Graph, Operation, TensorType
from ..passes import LAYOUT_OPS, POINTWISE_OPS

_MLX_DTYPES = {
    DType.F16: mx.float16,
    DType.BF16: mx.bfloat16,
    DType.F32: mx.float32,
    DType.BOOL: mx.bool_,
    DType.I32: mx.int32,
    DType.I64: mx.int64,
}

_ELEMENTS_PER_THREAD = 4
_COMPARISON_POINTWISE_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}
_LOGICAL_POINTWISE_OPS = {
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
}
_NUMERIC_PREDICATE_POINTWISE_OPS = {"isnan", "isinf"}
_MATMUL_MICROTILE = (4, 4)
_MATMUL_LARGE_K_MICROTILE = (8, 4)
_MATMUL_LARGE_K_THRESHOLD = 1024
_MATMUL_BLOCK_ROWS = 64
_MATMUL_BLOCK_COLUMNS = 64
_MATMUL_BLOCK_REDUCTION = 32
_METAL_KERNEL_CACHE_SIZE = 256
_ATTENTION_MAX_SHARED_BYTES = 31 * 1024
_ATTENTION_LONG_SEQUENCE_THRESHOLD = 768

_MSL_HEADER = """
inline float tsc_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = metal::abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - polynomial * metal::exp(-x * x));
}
"""


@lru_cache(maxsize=_METAL_KERNEL_CACHE_SIZE)
def _metal_kernel(
    name: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    source: str,
    header: str,
) -> Any:
    """Reuse an mx.fast.metal_kernel for an identical static kernel spec."""

    return mx.fast.metal_kernel(
        name=name,
        input_names=list(input_names),
        output_names=list(output_names),
        source=source,
        header=header,
    )


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    name: str
    source: str
    header: str
    input_ids: tuple[str, ...]
    output_id: str
    tensor_type: TensorType
    kind: str


@dataclass(slots=True)
class CompiledPlan:
    graph: Graph
    kernels: tuple[KernelArtifact, ...]
    _callables: dict[str, Any]

    def __call__(self, *inputs: Any) -> tuple[mx.array, ...]:
        if len(inputs) != len(self.graph.inputs):
            raise ValueError(
                f"expected {len(self.graph.inputs)} inputs, received {len(inputs)}"
            )

        env: dict[str, mx.array] = {}
        for value_id, value in zip(self.graph.inputs, inputs, strict=True):
            array = value if isinstance(value, mx.array) else mx.array(value)
            expected = self.graph.value(value_id).type
            if tuple(array.shape) != expected.shape:
                raise ValueError(
                    f"input {value_id} has shape {tuple(array.shape)}, expected {expected.shape}"
                )
            if array.dtype != _MLX_DTYPES[expected.dtype]:
                raise TypeError(
                    f"input {value_id} has dtype {array.dtype}, expected {expected.dtype.value}"
                )
            env[value_id] = array

        for op in self.graph.ops:
            runtime = self._callables.get(op.outputs[0])
            if runtime is not None:
                env[op.outputs[0]] = runtime(*(env[value_id] for value_id in op.inputs))
                continue

            if op.op == "attention":
                env[op.outputs[0]] = _evaluate_attention_mlx(op, env)
            elif op.op == "embedding":
                weight, indices = (env[value_id] for value_id in op.inputs)
                env[op.outputs[0]] = mx.take(weight, indices, axis=0)
            elif op.op == "cat":
                rank = len(env[op.inputs[0]].shape)
                axis = int(op.attrs["dim"]) % rank
                env[op.outputs[0]] = mx.concatenate(
                    [env[value_id] for value_id in op.inputs], axis=axis
                )
            elif op.op in LAYOUT_OPS:
                env[op.outputs[0]] = _evaluate_layout_mlx(
                    self.graph, op, env[op.inputs[0]]
                )
            elif op.op == "matmul":
                left, right = (env[value_id] for value_id in op.inputs)
                if op.attrs.get("transpose_a", False):
                    left = mx.swapaxes(left, -1, -2)
                if op.attrs.get("transpose_b", False):
                    right = mx.swapaxes(right, -1, -2)
                env[op.outputs[0]] = mx.matmul(left, right)
            elif op.op == "addmm":
                bias, left, right = (env[value_id] for value_id in op.inputs)
                if op.attrs.get("transpose_a", False):
                    left = mx.swapaxes(left, -1, -2)
                if op.attrs.get("transpose_b", False):
                    right = mx.swapaxes(right, -1, -2)
                alpha = float(op.attrs.get("alpha", 1.0))
                beta = float(op.attrs.get("beta", 1.0))
                env[op.outputs[0]] = beta * bias + alpha * mx.matmul(left, right)
            else:
                env[op.outputs[0]] = _evaluate_pointwise_mlx(
                    op, env, self.graph.value(op.outputs[0]).type
                )

        return tuple(env[value_id] for value_id in self.graph.outputs)


class AppleMetalBackend:
    """Lower supported IR regions to Metal kernels with an MLX runtime."""

    def __init__(
        self,
        *,
        codegen_matmul: bool = False,
        codegen_attention: bool = False,
    ) -> None:
        self.codegen_matmul = codegen_matmul
        self.codegen_attention = codegen_attention

    def compile(self, graph: Graph) -> CompiledPlan:
        if not mx.metal.is_available():
            raise RuntimeError("Apple Metal is unavailable in this process")

        kernels: list[KernelArtifact] = []
        callables: dict[str, Any] = {}
        for index, op in enumerate(graph.ops):
            compiled: tuple[KernelArtifact, Any] | None = None
            if op.op == "fused_elementwise":
                body = op.attrs.get("body")
                if not isinstance(body, tuple) or not all(
                    isinstance(item, Operation) for item in body
                ):
                    raise ValueError(
                        "fused_elementwise body must be a tuple of operations"
                    )
                compiled = self._compile_elementwise(
                    graph, op.inputs, body, op.outputs[0], index, "fused_elementwise"
                )
            elif op.op in POINTWISE_OPS:
                compiled = self._compile_elementwise(
                    graph, op.inputs, (op,), op.outputs[0], index, op.op
                )
            elif op.op in {"softmax", "safe_softmax"}:
                compiled = self._compile_softmax(graph, op, index)
            elif op.op in {"sum", "mean"}:
                compiled = self._compile_reduction(graph, op, index)
            elif op.op == "layer_norm":
                compiled = self._compile_layer_norm(graph, op, index)
            elif op.op == "rms_norm":
                compiled = self._compile_rms_norm(graph, op, index)
            elif op.op == "attention":
                _validate_attention_op(graph, op)
                if self.codegen_attention:
                    compiled = self._compile_attention(graph, op, index)
            elif op.op == "cat":
                _validate_cat_op(graph, op)
            elif op.op == "embedding":
                _validate_embedding_op(graph, op)
            elif op.op == "cast":
                _validate_cast_op(graph, op)
                output_type = graph.value(op.outputs[0]).type
                dtype = _MLX_DTYPES[output_type.dtype]
                callables[op.outputs[0]] = (
                    lambda value, dtype=dtype: value.astype(dtype)
                )
            elif op.op in LAYOUT_OPS:
                _validate_layout_op(graph, op)
            elif self.codegen_matmul and op.op in {"matmul", "addmm"}:
                left_position = 1 if op.op == "addmm" else 0
                right_position = 2 if op.op == "addmm" else 1
                left_type = graph.value(op.inputs[left_position]).type
                right_type = graph.value(op.inputs[right_position]).type
                output_type = graph.value(op.outputs[0]).type
                batch_shape = _matmul_batch_shape(
                    left_type.shape, right_type.shape
                )
                if (
                    batch_shape is not None
                    and output_type.shape[:-2] == batch_shape
                ):
                    compiled = self._compile_matmul(graph, op, index)

            if compiled is not None:
                artifact, runtime = compiled
                kernels.append(artifact)
                callables[artifact.output_id] = runtime

        return CompiledPlan(graph, tuple(kernels), callables)

    def _compile_elementwise(
        self,
        graph: Graph,
        input_ids: tuple[str, ...],
        body: tuple[Operation, ...],
        output_id: str,
        index: int,
        kind: str,
    ) -> tuple[KernelArtifact, Any]:
        output_type = graph.value(output_id).type
        _validate_pointwise_region(graph, body, output_id)
        for value_id in input_ids:
            input_type = graph.value(value_id).type
            if not _can_broadcast(input_type.shape, output_type.shape):
                raise NotImplementedError(
                    f"cannot broadcast pointwise input {input_type.shape} "
                    f"to {output_type.shape}"
                )

        source = _emit_pointwise_source(
            graph, input_ids, body, output_id, output_type.shape
        )
        name = f"{kind}_{index}"
        kernel = _metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(input_ids))),
            ("out",),
            source,
            _MSL_HEADER,
        )

        element_count = prod(output_type.shape)
        grid = (ceil(element_count / _ELEMENTS_PER_THREAD), 1, 1)
        threadgroup = (min(256, grid[0]), 1, 1)
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
            header=_MSL_HEADER,
            input_ids=input_ids,
            output_id=output_id,
            tensor_type=output_type,
            kind=kind,
        )
        return artifact, runtime

    def _compile_attention(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        """Emit a FlashAttention-style tiled online-softmax kernel."""

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
        use_simdgroup_qk = low_precision_attention and query_head_dim % 8 == 0
        use_simdgroup_pv = low_precision_attention and value_head_dim % 8 == 0
        query_tile = 8
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
                16
                if query_head_dim % 16 == 0 and value_head_dim % 16 == 0
                else 8
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
            score_expression = (
                f"key > query_pos ? -INFINITY : ({score_expression})"
            )

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
        input_names = ("in_0", "in_1", "in_2", "in_3") if has_mask else (
            "in_0",
            "in_1",
            "in_2",
        )
        name = (
            f"metal_attention_flash_{query_tile}x{key_tile}_"
            f"{threads}_{index}"
        )
        kernel = _metal_kernel(
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
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind="attention",
        )
        return artifact, runtime

    def _compile_softmax(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        if len(op.inputs) != 1:
            raise ValueError(f"{op.op} expects one input")

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        if input_type.dtype != output_type.dtype:
            raise NotImplementedError(
                f"Metal {op.op} requires identical input/output dtypes"
            )
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"{op.op} output shape {output_type.shape} != {input_type.shape}"
            )
        if not input_type.shape:
            raise NotImplementedError(f"Metal {op.op} requires a ranked tensor")

        rank = len(input_type.shape)
        dim = op.attrs.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"{op.op}.dim must be an integer, got {dim!r}")
        axis = dim % rank
        columns = input_type.shape[axis]
        outer_size = prod(input_type.shape[:axis]) or 1
        inner_size = prod(input_type.shape[axis + 1 :]) or 1
        softmax_count = outer_size * inner_size
        threads = _reduction_thread_count(columns)
        reduction_stride = threads // 2
        if inner_size == 1:
            input_base = f"output_index * {columns}u"
        else:
            input_base = (
                f"(output_index / {inner_size}u) * {columns * inner_size}u "
                f"+ output_index % {inner_size}u"
            )
        axis_offset = "col" if inner_size == 1 else f"col * {inner_size}u"
        input_expression = f"in_0[offset + ({axis_offset})]"
        exp_expression = f"metal::exp(float({input_expression}) - max_value)"
        if op.op == "safe_softmax":
            exp_expression = (
                f"(float({input_expression}) == -INFINITY ? 0.0f : {exp_expression})"
            )
        normalized_expression = f"{exp_expression} / sum_value"
        if op.op == "safe_softmax":
            normalized_expression = (
                f"(sum_value == 0.0f ? 0.0f : {normalized_expression})"
            )
        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint output_index = threadgroup_position_in_grid.x;
        uint offset = {input_base};
        threadgroup float scratch[{threads}];

        float local_max = -INFINITY;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_max = metal::max(
                local_max, float(in_0[offset + ({axis_offset})])
            );
        }}
        scratch[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] = metal::max(scratch[tid], scratch[tid + stride]);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_value = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_sum += {exp_expression};
        }}
        scratch[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float sum_value = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            out[offset + ({axis_offset})] = (T)({normalized_expression});
        }}
        """
        name = f"metal_{op.op}_{threads}_{index}"
        kernel = _metal_kernel(
            name,
            ("in_0",),
            ("out",),
            source,
            "",
        )
        grid = (softmax_count * threads, 1, 1)
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
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime

    def _compile_reduction(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        if len(op.inputs) != 1:
            raise ValueError(f"{op.op} expects one input")

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        requested_dtype = op.attrs.get("dtype")
        if requested_dtype is not None and requested_dtype != output_type.dtype:
            raise ValueError(
                f"{op.op}.dtype {requested_dtype} does not match "
                f"output dtype {output_type.dtype}"
            )
        if not input_type.shape:
            raise NotImplementedError(f"Metal {op.op} requires a ranked tensor")

        rank = len(input_type.shape)
        dims = op.attrs.get("dims")
        if not isinstance(dims, tuple) or not dims:
            raise ValueError(f"{op.op}.dims must be a non-empty tuple")
        normalized_dims: list[int] = []
        for dim in dims:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise ValueError(f"{op.op}.dims must contain integers")
            axis = dim % rank
            if axis in normalized_dims:
                raise ValueError(f"{op.op}.dims must not contain duplicates")
            normalized_dims.append(axis)
        normalized_dims.sort()

        keepdim = op.attrs.get("keepdim")
        if not isinstance(keepdim, bool):
            raise ValueError(f"{op.op}.keepdim must be a boolean")
        reduced_rank = len(normalized_dims)
        reduced_dims = set(normalized_dims)
        columns = prod(input_type.shape[axis] for axis in normalized_dims)
        if keepdim:
            expected_output_shape = tuple(
                1 if axis in reduced_dims else dim
                for axis, dim in enumerate(input_type.shape)
            )
        else:
            expected_output_shape = tuple(
                dim
                for axis, dim in enumerate(input_type.shape)
                if axis not in reduced_dims
            )
        if output_type.shape != expected_output_shape:
            raise ValueError(
                f"{op.op} output shape {output_type.shape} "
                f"!= {expected_output_shape}"
            )

        output_count = prod(expected_output_shape) or 1
        threads = _reduction_thread_count(columns)
        reduction_stride = threads // 2
        result_expression = "scratch[0]"
        if op.op == "mean":
            result_expression = f"scratch[0] / {columns}.0f"

        first_reduced_dim = normalized_dims[0]
        is_contiguous = normalized_dims == list(
            range(first_reduced_dim, first_reduced_dim + reduced_rank)
        )
        if is_contiguous:
            inner_shape = input_type.shape[first_reduced_dim + reduced_rank :]
            inner_size = prod(inner_shape) or 1
            if inner_size == 1:
                input_base = f"output_index * {columns}u"
            else:
                input_base = (
                    f"(output_index / {inner_size}u) * {columns * inner_size}u "
                    f"+ output_index % {inner_size}u"
                )
            reduction_offset = (
                "col" if inner_size == 1 else f"col * {inner_size}u"
            )
        else:
            compact_output_shape = tuple(
                dim
                for axis, dim in enumerate(input_type.shape)
                if axis not in reduced_dims
            )
            input_base = _reduction_output_base_expression(
                input_type.shape, compact_output_shape, reduced_dims
            )
            reduction_offset = _reduction_offset_expression(
                input_type.shape, normalized_dims
            )
        input_index_source = f"        uint offset = {input_base};"
        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint output_index = threadgroup_position_in_grid.x;
{input_index_source}
        threadgroup float scratch[{threads}];

        float local_value = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_value += float(in_0[offset + ({reduction_offset})]);
        }}
        scratch[tid] = local_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0u) {{
            out[output_index] = (O)({result_expression});
        }}
        """
        name = f"metal_{op.op}_{threads}_{index}"
        kernel = _metal_kernel(
            name,
            ("in_0",),
            ("out",),
            source,
            "",
        )
        grid = (output_count * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        input_dtype = _MLX_DTYPES[input_type.dtype]
        output_dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", input_dtype), ("O", output_dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[output_dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime

    def _compile_layer_norm(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        return self._compile_normalization(graph, op, index, kind="layer_norm")

    def _compile_rms_norm(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        return self._compile_normalization(graph, op, index, kind="rms_norm")

    def _compile_normalization(
        self,
        graph: Graph,
        op: Operation,
        index: int,
        *,
        kind: str,
    ) -> tuple[KernelArtifact, Any]:
        normalized_shape = op.attrs.get("normalized_shape")
        if not isinstance(normalized_shape, tuple) or not normalized_shape:
            raise ValueError(f"{kind}.normalized_shape must be a non-empty tuple")
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
            for dim in normalized_shape
        ):
            raise ValueError(
                f"{kind}.normalized_shape must contain positive integers"
            )

        has_weight = op.attrs.get("has_weight")
        if not isinstance(has_weight, bool):
            raise ValueError(f"{kind}.has_weight must be a boolean")
        has_bias = False
        if kind == "layer_norm":
            has_bias = op.attrs.get("has_bias")
            if not isinstance(has_bias, bool):
                raise ValueError("layer_norm.has_bias must be a boolean")
        expected_inputs = 1 + int(has_weight) + int(has_bias)
        if len(op.inputs) != expected_inputs:
            raise ValueError(
                f"{kind} expects {expected_inputs} inputs, got {len(op.inputs)}"
            )

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        if input_type.dtype != output_type.dtype:
            raise NotImplementedError(
                f"Metal {kind} requires identical input/output dtypes"
            )
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"{kind} output shape {output_type.shape} != {input_type.shape}"
            )
        normalized_rank = len(normalized_shape)
        if normalized_rank > len(input_type.shape):
            raise ValueError(
                f"{kind} normalized_shape {normalized_shape} exceeds input rank"
            )
        if input_type.shape[-normalized_rank:] != normalized_shape:
            raise ValueError(
                f"{kind} normalized_shape {normalized_shape} does not match "
                f"input shape {input_type.shape}"
            )

        input_position = 1
        weight_position: int | None = None
        bias_position: int | None = None
        if has_weight:
            weight_position = input_position
            input_position += 1
        if has_bias:
            bias_position = input_position
        for position in range(1, len(op.inputs)):
            parameter_type = graph.value(op.inputs[position]).type
            if parameter_type.dtype != input_type.dtype:
                raise NotImplementedError(
                    f"Metal {kind} requires identical parameter dtypes"
                )
            if parameter_type.shape != normalized_shape:
                raise ValueError(
                    f"{kind} parameter shape {parameter_type.shape} "
                    f"!= {normalized_shape}"
                )

        columns = prod(normalized_shape)
        rows = prod(input_type.shape[:-normalized_rank]) or 1
        threads = _reduction_thread_count(columns)
        reduction_stride = threads // 2
        eps = float(op.attrs.get("eps", 1e-5))

        if kind == "layer_norm":
            reduction_source = textwrap.dedent(f"""
        float local_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_sum += float(in_0[offset + col]);
        }}
        scratch[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mean_value = scratch[0] / {columns}.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_squared_deviation = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float centered = float(in_0[offset + col]) - mean_value;
            local_squared_deviation += centered * centered;
        }}
        scratch[tid] = local_squared_deviation;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float variance = scratch[0] / {columns}.0f;
            """).strip()
            reduction_result = "variance"
            centered_expression = "current - mean_value"
        else:
            reduction_source = textwrap.dedent(f"""
        float local_square_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float current = float(in_0[offset + col]);
            local_square_sum += current * current;
        }}
        scratch[tid] = local_square_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mean_square = scratch[0] / {columns}.0f;
            """).strip()
            reduction_result = "mean_square"
            centered_expression = "current"

        scale_expression = (
            f"float(in_{weight_position}[col])"
            if weight_position is not None
            else "1.0f"
        )
        normalized_expression = (
            f"({centered_expression}) * inv_std * {scale_expression}"
        )
        if bias_position is not None:
            normalized_expression += f" + float(in_{bias_position}[col])"

        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint offset = row * {columns}u;
        threadgroup float scratch[{threads}];

{textwrap.indent(reduction_source, '        ')}
        float inv_std = metal::rsqrt({reduction_result} + {eps!r}f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float current = float(in_0[offset + col]);
            out[offset + col] = (T)({normalized_expression});
        }}
        """
        name = f"metal_{kind}_{threads}_{index}"
        kernel = _metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            "",
        )
        grid = (rows * threads, 1, 1)
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
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=kind,
        )
        return artifact, runtime

    def _compile_matmul(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        expected_inputs = 3 if op.op == "addmm" else 2
        if len(op.inputs) != expected_inputs:
            raise ValueError(f"{op.op} expects {expected_inputs} inputs")

        if op.op == "addmm":
            bias_id, left_id, right_id = op.inputs
        else:
            bias_id = None
            left_id, right_id = op.inputs

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
            raise NotImplementedError("Metal matmul requires identical input/output dtypes")

        row_blocks = ceil(rows / _MATMUL_BLOCK_ROWS)
        column_blocks = ceil(columns / _MATMUL_BLOCK_COLUMNS)
        if _supports_simdgroup_matmul(rows, columns, output_type.dtype):
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
            threads = (
                (_MATMUL_BLOCK_ROWS // micro_rows)
                * (_MATMUL_BLOCK_COLUMNS // micro_columns)
            )
            batch_suffix = f"_batch{batch_count}" if batch_count > 1 else ""
            name = (
                f"metal_{op.op}_tiled_"
                f"{_MATMUL_BLOCK_ROWS}x{_MATMUL_BLOCK_COLUMNS}x"
                f"{_MATMUL_BLOCK_REDUCTION}_{micro_rows}x{micro_columns}"
                f"{batch_suffix}_{index}"
            )
        kernel = _metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            "",
        )
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
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime


def _validate_layout_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 1:
        raise ValueError(f"{op.op} expects one input")

    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    if input_type.dtype != output_type.dtype:
        raise ValueError(f"{op.op} must preserve dtype")

    rank = len(input_type.shape)
    if op.op == "reshape":
        shape = op.attrs.get("shape")
        if not isinstance(shape, tuple) or not shape or not all(
            isinstance(dim, int) and not isinstance(dim, bool)
            for dim in shape
        ):
            raise ValueError("reshape.shape must be a non-empty tuple of integers")
        if any(dim == 0 or dim < -1 for dim in shape):
            raise ValueError(f"reshape.shape contains an invalid dimension: {shape}")
        if shape.count(-1) > 1:
            raise ValueError(f"reshape.shape has multiple inferred dimensions: {shape}")

        input_count = prod(input_type.shape) or 1
        known_count = prod(dim for dim in shape if dim != -1) or 1
        if -1 in shape:
            if input_count % known_count:
                raise ValueError(
                    f"reshape shape {shape} cannot infer a dimension for "
                    f"{input_count} elements"
                )
            inferred = input_count // known_count
            resolved_shape = tuple(inferred if dim == -1 else dim for dim in shape)
        else:
            resolved_shape = shape
        if resolved_shape != output_type.shape:
            raise ValueError(
                f"reshape output shape {output_type.shape} != {resolved_shape}"
            )
        return

    if op.op == "transpose":
        dim0 = op.attrs.get("dim0")
        dim1 = op.attrs.get("dim1")
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool)
            for dim in (dim0, dim1)
        ):
            raise ValueError("transpose dimensions must be integers")
        normalized = tuple(
            dim + rank if dim < 0 else dim for dim in (dim0, dim1)
        )
        if any(dim < 0 or dim >= rank for dim in normalized):
            raise ValueError(f"transpose dimensions {(dim0, dim1)} exceed rank {rank}")
        expected_shape = list(input_type.shape)
        expected_shape[normalized[0]], expected_shape[normalized[1]] = (
            expected_shape[normalized[1]],
            expected_shape[normalized[0]],
        )
        if tuple(expected_shape) != output_type.shape:
            raise ValueError(
                f"transpose output shape {output_type.shape} != {tuple(expected_shape)}"
            )
        return

    if op.op == "permute":
        dims = op.attrs.get("dims")
        if not isinstance(dims, tuple) or len(dims) != rank or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(
                f"permute.dims must contain {rank} integer dimensions"
            )
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        if sorted(normalized) != list(range(rank)):
            raise ValueError(f"permute dimensions {dims} are not a permutation of rank {rank}")
        expected_shape = tuple(input_type.shape[dim] for dim in normalized)
        if expected_shape != output_type.shape:
            raise ValueError(
                f"permute output shape {output_type.shape} != {expected_shape}"
            )
        return

    if op.op == "expand":
        shape = op.attrs.get("shape")
        if not isinstance(shape, tuple) or len(shape) != len(output_type.shape):
            raise ValueError(
                f"expand.shape must have rank {len(output_type.shape)}, got {shape!r}"
            )
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool) and (dim == -1 or dim > 0)
            for dim in shape
        ):
            raise ValueError(f"expand.shape contains invalid dimensions: {shape}")
        if not _can_broadcast(input_type.shape, output_type.shape):
            raise ValueError(
                f"cannot expand shape {input_type.shape} to {output_type.shape}"
            )
        resolved_shape = tuple(
            output_type.shape[axis] if dim == -1 else dim
            for axis, dim in enumerate(shape)
        )
        if resolved_shape != output_type.shape:
            raise ValueError(
                f"expand output shape {resolved_shape} != {output_type.shape}"
            )
        return

    if op.op == "slice":
        dim = op.attrs.get("dim")
        start = op.attrs.get("start")
        end = op.attrs.get("end")
        step = op.attrs.get("step")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"slice.dim must be an integer, got {dim!r}")
        if start is not None and (
            not isinstance(start, int) or isinstance(start, bool)
        ):
            raise ValueError(f"slice.start must be an integer or None, got {start!r}")
        if end is not None and (
            not isinstance(end, int) or isinstance(end, bool)
        ):
            raise ValueError(f"slice.end must be an integer or None, got {end!r}")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(f"slice.step must be a positive integer, got {step!r}")
        if dim < -rank or dim >= rank:
            raise ValueError(f"slice.dim {dim} exceeds rank {rank}")
        axis = dim + rank if dim < 0 else dim
        sliced_length = len(
            range(*slice(start, end, step).indices(input_type.shape[axis]))
        )
        expected_shape = list(input_type.shape)
        expected_shape[axis] = sliced_length
        if tuple(expected_shape) != output_type.shape:
            raise ValueError(
                f"slice output shape {output_type.shape} != {tuple(expected_shape)}"
            )
        return

    if op.op == "flip":
        dims = op.attrs.get("dims")
        if not isinstance(dims, (tuple, list)) or not dims or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(f"flip.dims must contain integers, got {dims!r}")
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        if any(dim < 0 or dim >= rank for dim in normalized):
            raise ValueError(f"flip.dims {dims} exceed rank {rank}")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"flip.dims contains duplicate axes: {dims}")
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"flip output shape {output_type.shape} != {input_type.shape}"
            )
        return

    if op.op == "roll":
        shifts = op.attrs.get("shifts")
        if not isinstance(shifts, (tuple, list)) or not shifts or not all(
            isinstance(shift, int) and not isinstance(shift, bool)
            for shift in shifts
        ):
            raise ValueError(f"roll.shifts must contain integers, got {shifts!r}")
        dims = op.attrs.get("dims")
        if dims is None:
            if len(shifts) != 1:
                raise ValueError(
                    "roll without dims requires exactly one shift, got "
                    f"{len(shifts)}"
                )
            if input_type.shape != output_type.shape:
                raise ValueError(
                    f"roll output shape {output_type.shape} != {input_type.shape}"
                )
            return
        if not isinstance(dims, (tuple, list)) or not dims or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in dims
        ):
            raise ValueError(f"roll.dims must contain integers, got {dims!r}")
        if len(dims) != len(shifts):
            raise ValueError(
                "roll shifts and dims must have the same length, got "
                f"{len(shifts)} and {len(dims)}"
            )
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        if any(dim < 0 or dim >= rank for dim in normalized):
            raise ValueError(f"roll.dims {dims} exceed rank {rank}")
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"roll output shape {output_type.shape} != {input_type.shape}"
            )
        return

    if op.op == "unsqueeze":
        dim = op.attrs.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"unsqueeze.dim must be an integer, got {dim!r}")
        output_rank = rank + 1
        axis = dim + output_rank if dim < 0 else dim
        if axis < 0 or axis >= output_rank:
            raise ValueError(f"unsqueeze.dim {dim} exceeds output rank {output_rank}")
        expected_shape = input_type.shape[:axis] + (1,) + input_type.shape[axis:]
        if expected_shape != output_type.shape:
            raise ValueError(
                f"unsqueeze output shape {output_type.shape} != {expected_shape}"
            )
        return

    if op.op == "squeeze":
        dim = op.attrs.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"squeeze.dim must be an integer, got {dim!r}")
        if rank == 0 or dim < -rank or dim >= rank:
            raise ValueError(f"squeeze.dim {dim} exceeds rank {rank}")
        axis = dim + rank if dim < 0 else dim
        expected_shape = input_type.shape
        if input_type.shape[axis] == 1:
            expected_shape = (
                input_type.shape[:axis] + input_type.shape[axis + 1 :]
            )
        if expected_shape != output_type.shape:
            raise ValueError(
                f"squeeze output shape {output_type.shape} != {expected_shape}"
            )
        return

    if input_type.shape != output_type.shape:
        raise ValueError(
            f"{op.op} output shape {output_type.shape} != {input_type.shape}"
        )


def _can_broadcast(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> bool:
    if len(input_shape) > len(output_shape):
        return False
    offset = len(output_shape) - len(input_shape)
    return all(
        input_dim in {1, output_shape[offset + axis]}
        for axis, input_dim in enumerate(input_shape)
    )


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError(f"expected a positive integer, got {value}")
    return 1 << (value - 1).bit_length()


def _reduction_output_base_expression(
    input_shape: tuple[int, ...],
    compact_output_shape: tuple[int, ...],
    reduced_dims: set[int],
) -> str:
    """Map a flattened output index to the first input element it reduces."""

    non_reduced_axes = [
        axis for axis in range(len(input_shape)) if axis not in reduced_dims
    ]
    terms: list[str] = []
    for output_axis, input_axis in enumerate(non_reduced_axes):
        output_stride = prod(compact_output_shape[output_axis + 1 :])
        output_dim = compact_output_shape[output_axis]
        coordinate = (
            f"((output_index / {output_stride}u) % {output_dim}u)"
            if output_stride > 1
            else f"(output_index % {output_dim}u)"
        )
        input_stride = prod(input_shape[input_axis + 1 :])
        terms.append(
            coordinate if input_stride == 1 else f"{coordinate} * {input_stride}u"
        )
    return " + ".join(terms) if terms else "0u"


def _reduction_offset_expression(
    input_shape: tuple[int, ...], reduced_dims: list[int]
) -> str:
    """Map a flattened reduction index to its input offset."""

    terms: list[str] = []
    reduction_stride = 1
    for axis in reversed(reduced_dims):
        dimension = input_shape[axis]
        coordinate = (
            f"(col % {dimension}u)"
            if reduction_stride == 1
            else f"((col / {reduction_stride}u) % {dimension}u)"
        )
        input_stride = prod(input_shape[axis + 1 :])
        terms.append(
            coordinate if input_stride == 1 else f"{coordinate} * {input_stride}u"
        )
        reduction_stride *= dimension
    return " + ".join(reversed(terms))


def _reduction_thread_count(columns: int) -> int:
    return min(256, _next_power_of_two(columns))


def _offset_expression(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    index_expression: str,
) -> str:
    if input_shape == output_shape:
        return index_expression

    offset = len(output_shape) - len(input_shape)
    terms: list[str] = []
    for output_axis, output_dim in enumerate(output_shape):
        input_axis = output_axis - offset
        if input_axis < 0 or input_shape[input_axis] == 1:
            continue
        output_stride = prod(output_shape[output_axis + 1 :])
        input_stride = prod(input_shape[input_axis + 1 :])
        coordinate = f"((({index_expression}) / {output_stride}) % {output_dim})"
        terms.append(f"{coordinate} * {input_stride}")
    return " + ".join(terms) if terms else "0"


def _validate_pointwise_region(
    graph: Graph, body: tuple[Operation, ...], output_id: str
) -> None:
    for op in body:
        output_type = graph.value(op.outputs[0]).type
        input_types = tuple(graph.value(value_id).type for value_id in op.inputs)

        if op.op == "scalar_tensor":
            if op.inputs:
                raise ValueError("scalar_tensor must not have tensor inputs")
            if output_type.dtype in {DType.I32, DType.I64}:
                raise ValueError("scalar_tensor does not support integer outputs")
            value = op.attrs.get("value")
            if not isinstance(value, (bool, int, float)):
                raise ValueError(
                    f"scalar_tensor.value must be scalar, got {value!r}"
                )
            continue

        if op.op in _COMPARISON_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL:
                raise ValueError(f"{op.op} must produce a boolean tensor")
            if not (
                len(input_types) == 2
                or (len(input_types) == 1 and "scalar" in op.attrs)
            ):
                raise ValueError(f"malformed comparison op: {op.op}")
            continue

        if op.op in _LOGICAL_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL:
                raise ValueError(f"{op.op} must produce a boolean tensor")
            if op.op == "logical_not":
                expected_inputs = len(input_types) == 1
            else:
                expected_inputs = len(input_types) == 2 or (
                    len(input_types) == 1
                    and isinstance(op.attrs.get("scalar"), bool)
                )
            if not expected_inputs:
                raise ValueError(f"malformed logical op: {op.op}")
            if any(input_type.dtype != DType.BOOL for input_type in input_types):
                raise ValueError(f"{op.op} requires boolean tensor inputs")
            continue

        if op.op in _NUMERIC_PREDICATE_POINTWISE_OPS:
            if output_type.dtype != DType.BOOL or len(input_types) != 1:
                raise ValueError(f"malformed predicate op: {op.op}")
            if input_types[0].dtype == DType.BOOL:
                raise ValueError(f"{op.op} requires a numeric tensor input")
            continue

        if op.op == "where":
            if len(input_types) != 3:
                raise ValueError("where requires condition, input and other tensors")
            if input_types[0].dtype != DType.BOOL:
                raise ValueError("where condition must be a boolean tensor")
            continue

        if op.op == "masked_fill":
            if len(input_types) != 2:
                raise ValueError("masked_fill requires input and mask tensors")
            if input_types[1].dtype != DType.BOOL:
                raise ValueError("masked_fill mask must be a boolean tensor")
            if output_type != input_types[0]:
                raise ValueError("masked_fill must preserve the input tensor type")
            value = op.attrs.get("value")
            if output_type.dtype == DType.BOOL:
                if not isinstance(value, bool):
                    raise ValueError("boolean masked_fill requires a boolean value")
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("numeric masked_fill requires a numeric value")
            continue

        if output_type.dtype == DType.BOOL or any(
            input_type.dtype == DType.BOOL for input_type in input_types
        ):
            raise ValueError(f"{op.op} does not accept boolean tensor operands")
        if output_type.dtype in {DType.I32, DType.I64} or any(
            input_type.dtype in {DType.I32, DType.I64}
            for input_type in input_types
        ):
            raise ValueError(f"{op.op} does not accept integer tensor operands")

    if graph.value(output_id).type.dtype == DType.BOOL:
        producer = next(
            (op for op in reversed(body) if op.outputs[0] == output_id), None
        )
        if producer is None or producer.op not in (
            _COMPARISON_POINTWISE_OPS
            | _LOGICAL_POINTWISE_OPS
            | _NUMERIC_PREDICATE_POINTWISE_OPS
            | {"where", "masked_fill", "scalar_tensor"}
        ):
            raise ValueError("boolean pointwise region has an unsupported root op")


def _emit_pointwise_source(
    graph: Graph,
    input_ids: tuple[str, ...],
    body: tuple[Operation, ...],
    output_id: str,
    output_shape: tuple[int, ...],
) -> str:
    element_count = prod(output_shape)
    lines = [
        "uint base = thread_position_in_grid.x * 4u;",
        "for (uint lane = 0u; lane < 4u; ++lane) {",
        "    uint elem = base + lane;",
        f"    if (elem >= {element_count}) {{",
        "        break;",
        "    }",
    ]
    env: dict[str, str] = {}
    common: dict[tuple[object, ...], str] = {}

    for position, value_id in enumerate(input_ids):
        variable = f"input_{position}"
        offset = _offset_expression(
            graph.value(value_id).type.shape, output_shape, "elem"
        )
        ctype = (
            "bool"
            if graph.value(value_id).type.dtype == DType.BOOL
            else "float"
        )
        lines.append(f"    {ctype} {variable} = in_{position}[{offset}];")
        env[value_id] = variable

    for op_index, op in enumerate(body):
        arguments = [env[value_id] for value_id in op.inputs]
        key = (
            op.op,
            op.inputs,
            graph.value(op.outputs[0]).type.dtype.value,
            tuple((name, repr(value)) for name, value in sorted(op.attrs.items())),
        )
        existing = common.get(key)
        if existing is not None:
            env[op.outputs[0]] = existing
            continue
        output = f"temp_{op_index}"
        expression = _emit_pointwise_expression(op, arguments)
        ctype = (
            "bool"
            if graph.value(op.outputs[0]).type.dtype == DType.BOOL
            else "float"
        )
        lines.append(f"    {ctype} {output} = {expression};")
        env[op.outputs[0]] = output
        common[key] = output

    if output_id not in env:
        raise ValueError("pointwise region does not produce its declared output")
    lines.extend(
        [
            f"    out[elem] = (T)({env[output_id]});",
            "}",
        ]
    )
    return "\n".join(f"        {line}" for line in lines)


def _binary_operands(op: Operation, arguments: list[str]) -> tuple[str, str]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar_value = op.attrs["scalar"]
    if isinstance(scalar_value, bool):
        scalar = "true" if scalar_value else "false"
    else:
        scalar = _msl_float_literal(float(scalar_value))
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _msl_float_literal(value: float) -> str:
    if isnan(value):
        return "NAN"
    if isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    return repr(value)


def _emit_pointwise_expression(op: Operation, arguments: list[str]) -> str:
    if op.op == "scalar_tensor":
        value = op.attrs.get("value")
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return _msl_float_literal(float(value))
        raise ValueError(f"scalar_tensor.value must be scalar, got {value!r}")
    if op.op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left, right = _binary_operands(op, arguments)
        operator = {
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }[op.op]
        return f"({left} {operator} {right})"
    if op.op == "logical_and":
        left, right = _binary_operands(op, arguments)
        return f"({left} && {right})"
    if op.op == "logical_or":
        left, right = _binary_operands(op, arguments)
        return f"({left} || {right})"
    if op.op == "logical_xor":
        left, right = _binary_operands(op, arguments)
        return f"({left} != {right})"
    if op.op == "logical_not":
        return f"(!{arguments[0]})"
    if op.op == "isnan":
        return f"metal::isnan({arguments[0]})"
    if op.op == "isinf":
        return f"metal::isinf({arguments[0]})"
    if op.op == "where":
        return f"({arguments[0]} ? {arguments[1]} : {arguments[2]})"
    if op.op == "masked_fill":
        value = op.attrs.get("value")
        if isinstance(value, bool):
            fill = "true" if value else "false"
        else:
            fill = _msl_float_literal(float(value))
        return f"({arguments[1]} ? {fill} : {arguments[0]})"
    if op.op == "silu":
        value = arguments[0]
        return f"{value} / (1.0f + metal::exp(-{value}))"
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            return (
                f"0.5f * {value} * (1.0f + metal::tanh("
                f"0.7978845608028654f * ({value} + "
                f"0.044715f * {value} * {value} * {value})))"
            )
        return (
            f"0.5f * {value} * (1.0f + tsc_erf("
            f"{value} * 0.7071067811865476f))"
        )
    if op.op in {"mul", "div"}:
        left, right = _binary_operands(op, arguments)
        operator = "*" if op.op == "mul" else "/"
        return f"{left} {operator} {right}"
    if op.op == "pow":
        left, right = _binary_operands(op, arguments)
        return f"metal::pow({left}, {right})"
    if op.op in {"add", "sub", "rsub"}:
        left, right = _binary_operands(op, arguments)
        operator = "+" if op.op == "add" else "-"
        alpha = float(op.attrs.get("alpha", 1.0))
        return f"{left} {operator} {alpha!r} * {right}"
    if op.op == "neg":
        return f"-{arguments[0]}"
    if op.op == "abs":
        return f"metal::abs({arguments[0]})"
    if op.op == "sigmoid":
        return f"1.0f / (1.0f + metal::exp(-{arguments[0]}))"
    if op.op == "tanh":
        return f"metal::tanh({arguments[0]})"
    if op.op == "relu":
        return f"metal::max(0.0f, {arguments[0]})"
    if op.op == "leaky_relu":
        value = arguments[0]
        slope = _msl_float_literal(float(op.attrs.get("negative_slope", 0.01)))
        return f"(({value}) >= 0.0f ? ({value}) : {slope} * ({value}))"
    if op.op == "elu":
        value = arguments[0]
        alpha = _msl_float_literal(float(op.attrs.get("alpha", 1.0)))
        return (
            f"(({value}) >= 0.0f ? ({value}) : "
            f"{alpha} * (metal::exp({value}) - 1.0f))"
        )
    if op.op == "hardswish":
        value = arguments[0]
        body = (
            f"({value}) * metal::min(metal::max(({value}) + 3.0f, 0.0f), 6.0f) "
            "/ 6.0f"
        )
        return f"(metal::isnan({value}) ? NAN : {body})"
    if op.op == "exp":
        return f"metal::exp({arguments[0]})"
    if op.op == "erf":
        return f"tsc_erf({arguments[0]})"
    if op.op == "sin":
        return f"metal::sin({arguments[0]})"
    if op.op == "cos":
        return f"metal::cos({arguments[0]})"
    if op.op == "sqrt":
        return f"metal::sqrt({arguments[0]})"
    if op.op == "rsqrt":
        return f"metal::rsqrt({arguments[0]})"
    if op.op == "reciprocal":
        return f"1.0f / {arguments[0]}"
    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        lower = _clamp_bound_expression(op, arguments, "min")
        upper = _clamp_bound_expression(op, arguments, "max")
        if lower is not None and upper is not None:
            clipped = f"metal::min(metal::max({value}, {lower}), {upper})"
        elif lower is not None:
            clipped = f"metal::max({value}, {lower})"
        elif upper is not None:
            clipped = f"metal::min({value}, {upper})"
        else:
            return value
        invalid_inputs = [value]
        if lower is not None:
            invalid_inputs.append(lower)
        if upper is not None:
            invalid_inputs.append(upper)
        nan_check = " || ".join(
            f"metal::isnan({expression})" for expression in invalid_inputs
        )
        return f"(({nan_check}) ? NAN : {clipped})"
    raise NotImplementedError(f"cannot emit fused pointwise op: {op.op}")


def _clamp_bound_expression(
    op: Operation, arguments: list[str], name: str
) -> str | None:
    input_position = op.attrs.get(f"{name}_input")
    if input_position is not None:
        if (
            not isinstance(input_position, int)
            or isinstance(input_position, bool)
            or input_position < 1
            or input_position >= len(arguments)
        ):
            raise ValueError(f"clamp.{name}_input is malformed")
        return arguments[input_position]
    bound = op.attrs.get(name)
    if bound is None:
        return None
    if not isinstance(bound, (int, float)) or isinstance(bound, bool):
        raise ValueError(f"clamp.{name} must be a scalar number")
    value = float(bound)
    if isnan(value):
        return "NAN"
    if isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    return repr(value)


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
        bias_id, _, _ = op.inputs
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = ()
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
    output_batch_offset = (
        f"batch * {rows * columns}u + " if batch_count > 1 else ""
    )
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
                f"                    + k_base + local_k);",
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
                f"                    + col_base + local_column);",
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
            left_index = (
                f"(k_base + local_k) * {rows}u + row_base + local_row"
            )
        else:
            left_index = (
                f"(row_base + local_row) * {reduction}u + k_base + local_k"
            )
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
            right_index = (
                f"(col_base + local_column) * {reduction}u + k_base + local_k"
            )
        else:
            right_index = (
                f"(k_base + local_k) * {columns}u + col_base + local_column"
            )
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
        value = (
            f"{beta!r} * float(in_0[{bias_offset}]) "
            f"+ {alpha!r} * {value}"
        )
    lines.extend(
        [
            f"    out[linear] = (T)({value});",
            "}",
        ]
    )
    return "\n".join(f"        {line}" for line in lines)


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
        bias_id, _, _ = op.inputs
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = ()
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
        right_index = (
            f"(col_base + local_column) * {reduction}u + k_base + local_k"
        )
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

    lines.append(f"    for (uint local_k = 0u; local_k < {block_reduction}u; ++local_k) {{")
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
            batch_offset = (
                f"batch * {rows * columns}u + " if batch_count > 1 else ""
            )
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
                    f"{beta!r} * float(in_0[{bias_offset}]) "
                    f"+ {alpha!r} * {accumulator}"
                )
            else:
                value = accumulator
            lines.append(f"{indent}out[{linear}] = (T)({value});")
            if not exact_tiles:
                lines.append("}")

    return "\n".join(f"        {line}" for line in lines)


def _evaluate_layout_mlx(
    graph: Graph, op: Operation, value: mx.array
) -> mx.array:
    if op.op == "reshape":
        return mx.reshape(value, graph.value(op.outputs[0]).type.shape)

    rank = len(value.shape)
    if op.op == "transpose":
        dim0 = int(op.attrs["dim0"])
        dim1 = int(op.attrs["dim1"])
        dim0 = dim0 + rank if dim0 < 0 else dim0
        dim1 = dim1 + rank if dim1 < 0 else dim1
        return mx.swapaxes(value, dim0, dim1)

    if op.op == "permute":
        dims = tuple(int(dim) for dim in op.attrs["dims"])
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        return mx.transpose(value, normalized)

    if op.op == "expand":
        return mx.broadcast_to(value, graph.value(op.outputs[0]).type.shape)

    if op.op == "slice":
        index: list[slice] = [slice(None)] * rank
        axis = int(op.attrs["dim"]) % rank
        start = (
            int(op.attrs["start"]) if op.attrs.get("start") is not None else None
        )
        end = int(op.attrs["end"]) if op.attrs.get("end") is not None else None
        step = int(op.attrs["step"])
        start, end, step = slice(start, end, step).indices(value.shape[axis])
        index[axis] = slice(start, end, step)
        return value[tuple(index)]

    if op.op == "flip":
        dims = tuple(int(dim) for dim in op.attrs["dims"])
        normalized = tuple(dim + rank if dim < 0 else dim for dim in dims)
        return mx.flip(value, axis=normalized)

    if op.op == "roll":
        shifts = tuple(int(shift) for shift in op.attrs["shifts"])
        dims = op.attrs.get("dims")
        if dims is None:
            return mx.roll(value, shift=shifts[0], axis=None)
        normalized = tuple(
            int(dim) + rank if int(dim) < 0 else int(dim) for dim in dims
        )
        return mx.roll(value, shift=shifts, axis=normalized)

    if op.op == "unsqueeze":
        output_rank = rank + 1
        dim = int(op.attrs["dim"])
        axis = dim + output_rank if dim < 0 else dim
        return mx.expand_dims(value, axis=axis)

    if op.op == "squeeze":
        dim = int(op.attrs["dim"])
        axis = dim + rank if dim < 0 else dim
        if value.shape[axis] != 1:
            return value
        return mx.squeeze(value, axis=axis)

    if op.op == "contiguous":
        return mx.contiguous(value)

    if op.op == "alias":
        return value

    raise NotImplementedError(f"no Apple runtime lowering for {op.op}")


def _evaluate_pointwise_mlx(
    op: Operation, env: dict[str, mx.array], output_type: TensorType
) -> mx.array:
    arguments = [env[value_id] for value_id in op.inputs]
    if op.op == "scalar_tensor":
        value = op.attrs.get("value")
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"scalar_tensor.value must be scalar, got {value!r}")
        return mx.array(value, dtype=_MLX_DTYPES[output_type.dtype])
    if op.op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "eq":
            return left == right
        if op.op == "ne":
            return left != right
        if op.op == "lt":
            return left < right
        if op.op == "le":
            return left <= right
        if op.op == "gt":
            return left > right
        return left >= right
    if op.op == "logical_and":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_and(left, right)
    if op.op == "logical_or":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_or(left, right)
    if op.op == "logical_xor":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_xor(left, right)
    if op.op == "logical_not":
        return mx.logical_not(arguments[0])
    if op.op == "isnan":
        return mx.isnan(arguments[0])
    if op.op == "isinf":
        return mx.isinf(arguments[0])
    if op.op == "where":
        return mx.where(arguments[0], arguments[1], arguments[2])
    if op.op == "masked_fill":
        fill = mx.array(op.attrs["value"], dtype=arguments[0].dtype)
        return mx.where(arguments[1], fill, arguments[0])
    if op.op == "silu":
        value = arguments[0]
        return value / (1.0 + mx.exp(-value))
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            inner = 0.7978845608028654 * (
                value + 0.044715 * value * value * value
            )
            return 0.5 * value * (1.0 + mx.tanh(inner))
        return 0.5 * value * (1.0 + mx.erf(value * 0.7071067811865476))
    if op.op in {"mul", "div", "add", "sub", "rsub", "pow"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "mul":
            return left * right
        if op.op == "div":
            return left / right
        if op.op == "pow":
            return mx.power(left, right)
        alpha = float(op.attrs.get("alpha", 1.0))
        if op.op == "add":
            return left + alpha * right
        return left - alpha * right
    if op.op == "neg":
        return -arguments[0]
    if op.op == "abs":
        return mx.abs(arguments[0])
    if op.op == "sigmoid":
        return 1.0 / (1.0 + mx.exp(-arguments[0]))
    if op.op == "tanh":
        return mx.tanh(arguments[0])
    if op.op == "relu":
        return mx.maximum(arguments[0], 0.0)
    if op.op == "leaky_relu":
        value = arguments[0]
        slope = float(op.attrs.get("negative_slope", 0.01))
        return mx.where(value >= 0.0, value, slope * value)
    if op.op == "elu":
        value = arguments[0]
        alpha = float(op.attrs.get("alpha", 1.0))
        return mx.where(value >= 0.0, value, alpha * (mx.exp(value) - 1.0))
    if op.op == "hardswish":
        value = arguments[0]
        body = value * mx.minimum(mx.maximum(value + 3.0, 0.0), 6.0) / 6.0
        return mx.where(mx.isnan(value), float("nan"), body)
    if op.op == "exp":
        return mx.exp(arguments[0])
    if op.op == "erf":
        return mx.erf(arguments[0])
    if op.op == "sin":
        return mx.sin(arguments[0])
    if op.op == "cos":
        return mx.cos(arguments[0])
    if op.op == "sqrt":
        return mx.sqrt(arguments[0])
    if op.op == "rsqrt":
        return 1.0 / mx.sqrt(arguments[0])
    if op.op == "reciprocal":
        return 1.0 / arguments[0]
    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        lower: mx.array | float = -float("inf")
        upper: mx.array | float = float("inf")
        if "min_input" in op.attrs:
            lower = arguments[int(op.attrs["min_input"])]
        elif "min" in op.attrs:
            lower = float(op.attrs["min"])
        if "max_input" in op.attrs:
            upper = arguments[int(op.attrs["max_input"])]
        elif "max" in op.attrs:
            upper = float(op.attrs["max"])
        clipped = mx.minimum(mx.maximum(value, lower), upper)
        invalid = mx.isnan(value) | mx.isnan(lower) | mx.isnan(upper)
        return mx.where(invalid, float("nan"), clipped)
    raise NotImplementedError(f"no Apple runtime lowering for {op.op}")


def _binary_mx_operands(
    op: Operation, arguments: list[mx.array]
) -> tuple[mx.array | float, mx.array | float]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar_value = op.attrs["scalar"]
    scalar = (
        mx.array(scalar_value)
        if isinstance(scalar_value, bool)
        else float(scalar_value)
    )
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _native_attention_mask_supported(
    query_rank: int, mask_shape: tuple[int, ...] | None
) -> bool:
    if mask_shape is None:
        return False
    if query_rank == 4:
        return len(mask_shape) == 4
    if query_rank == 3:
        return len(mask_shape) in {2, 3}
    return False


def _attention_shared_memory_bytes(
    query_tile: int,
    key_tile: int,
    query_head_dim: int,
    value_head_dim: int,
    pv_value_tile: int,
) -> int:
    element_bytes = 2
    if key_tile == 64:
        score_workspace_bytes = query_tile * key_tile * 4
        probability_and_pv_bytes = (
            query_tile * key_tile * element_bytes
            + query_tile * pv_value_tile * 4
        )
        score_and_pv_bytes = max(
            score_workspace_bytes, probability_and_pv_bytes
        )
        probability_bytes = 0
        pv_bytes = 0
    else:
        score_and_pv_bytes = query_tile * key_tile * 4
        probability_bytes = query_tile * key_tile * element_bytes
        pv_bytes = query_tile * pv_value_tile * 4
    return (
        query_tile * query_head_dim * element_bytes
        + key_tile * max(query_head_dim, value_head_dim) * element_bytes
        + score_and_pv_bytes
        + probability_bytes
        + pv_bytes
    )


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


def _validate_cat_op(graph: Graph, op: Operation) -> None:
    if not op.inputs:
        raise ValueError("cat expects at least one input tensor")
    dim = op.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"cat.dim must be an integer, got {dim!r}")

    input_types = [graph.value(value_id).type for value_id in op.inputs]
    output_type = graph.value(op.outputs[0]).type
    rank = len(input_types[0].shape)
    if rank == 0:
        raise NotImplementedError("cat requires ranked tensors")
    axis = dim % rank
    if any(len(input_type.shape) != rank for input_type in input_types):
        raise ValueError("cat inputs must have the same rank")
    if any(input_type.dtype != output_type.dtype for input_type in input_types):
        raise ValueError("cat must preserve input dtype")
    for position, input_type in enumerate(input_types):
        for input_axis, (input_dim, output_dim) in enumerate(
            zip(input_type.shape, output_type.shape, strict=True)
        ):
            if input_axis != axis and input_dim != output_dim:
                raise ValueError(
                    f"cat input {position} shape {input_type.shape} is incompatible "
                    f"with output shape {output_type.shape} on axis {input_axis}"
                )

    expected_shape = list(output_type.shape)
    expected_shape[axis] = sum(input_type.shape[axis] for input_type in input_types)
    if tuple(expected_shape) != output_type.shape:
        raise ValueError(
            f"cat output shape {output_type.shape} != {tuple(expected_shape)}"
        )


def _validate_embedding_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 2:
        raise ValueError(f"embedding expects two inputs, got {len(op.inputs)}")
    weight_type = graph.value(op.inputs[0]).type
    indices_type = graph.value(op.inputs[1]).type
    output_type = graph.value(op.outputs[0]).type
    if len(weight_type.shape) != 2:
        raise ValueError(
            f"embedding weight must have rank 2, got {weight_type.shape}"
        )
    if weight_type.dtype not in {DType.F16, DType.BF16, DType.F32}:
        raise ValueError(
            f"embedding weight must be floating point, got {weight_type.dtype.value}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "embedding indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )
    expected_shape = indices_type.shape + (weight_type.shape[1],)
    if output_type.shape != expected_shape or output_type.dtype != weight_type.dtype:
        raise ValueError(
            f"embedding output metadata {(output_type.shape, output_type.dtype)} does "
            f"not match expected {(expected_shape, weight_type.dtype)}"
        )

    allowed_attrs = {"padding_idx", "scale_grad_by_freq", "sparse"}
    unknown_attrs = set(op.attrs) - allowed_attrs
    if unknown_attrs:
        raise ValueError(
            f"embedding has unsupported attributes: {sorted(unknown_attrs)}"
        )
    padding_idx = op.attrs.get("padding_idx", -1)
    if padding_idx is not None:
        if not isinstance(padding_idx, int) or isinstance(padding_idx, bool):
            raise ValueError(
                f"embedding.padding_idx must be an integer, got {padding_idx!r}"
            )
        if not -weight_type.shape[0] <= padding_idx < weight_type.shape[0]:
            raise ValueError(
                f"embedding.padding_idx {padding_idx} is outside the weight table"
            )
    for name in ("scale_grad_by_freq", "sparse"):
        if not isinstance(op.attrs.get(name, False), bool):
            raise ValueError(f"embedding.{name} must be a boolean")


def _validate_cast_op(graph: Graph, op: Operation) -> None:
    if len(op.inputs) != 1:
        raise ValueError(f"cast expects one input, got {len(op.inputs)}")
    input_shape = graph.value(op.inputs[0]).type.shape
    output_shape = graph.value(op.outputs[0]).type.shape
    if input_shape != output_shape:
        raise ValueError(f"cast shape {input_shape} != {output_shape}")


def _validate_attention_op(graph: Graph, op: Operation) -> None:
    expected_inputs = 4 if op.attrs.get("has_mask") else 3
    if len(op.inputs) != expected_inputs:
        raise ValueError(
            f"attention expects {expected_inputs} inputs, got {len(op.inputs)}"
        )

    query_type = graph.value(op.inputs[0]).type
    key_type = graph.value(op.inputs[1]).type
    value_type = graph.value(op.inputs[2]).type
    output_type = graph.value(op.outputs[0]).type
    query_shape = query_type.shape
    key_shape = key_type.shape
    value_shape = value_type.shape
    if len(query_shape) not in {3, 4}:
        raise NotImplementedError(
            f"attention requires rank-3 or rank-4 inputs, got {query_shape}"
        )
    if len(key_shape) != len(query_shape) or len(value_shape) != len(query_shape):
        raise ValueError("attention Q/K/V ranks must match")
    if query_shape[:1] != key_shape[:1]:
        raise ValueError("attention Q/K batch dimensions must match")
    if key_shape[:1] != value_shape[:1]:
        raise ValueError("attention K/V batch dimensions must match")
    if query_shape[-1] != key_shape[-1]:
        raise ValueError("attention Q/K head dimensions must match")
    if key_shape[-2] != value_shape[-2]:
        raise ValueError("attention K/V sequence dimensions must match")
    if len(query_shape) == 4:
        query_heads, key_value_heads = query_shape[-3], key_shape[-3]
        if query_heads < key_value_heads or query_heads % key_value_heads != 0:
            raise ValueError(
                "attention query heads must be a multiple of key/value heads, "
                f"got {query_heads} and {key_value_heads}"
            )
    expected_output_shape = query_shape[:-1] + (value_shape[-1],)
    if output_type.shape != expected_output_shape:
        raise ValueError(
            f"attention output shape {output_type.shape} != {expected_output_shape}"
        )
    if not (query_type.dtype == key_type.dtype == value_type.dtype == output_type.dtype):
        raise NotImplementedError(
            "attention requires identical Q/K/V/output dtypes"
        )

    scale = op.attrs.get("scale")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError(f"attention.scale must be a scalar number, got {scale!r}")
    causal = op.attrs.get("causal", False)
    if not isinstance(causal, bool):
        raise ValueError(f"attention.causal must be a boolean, got {causal!r}")
    safe_softmax = op.attrs.get("safe_softmax", False)
    if not isinstance(safe_softmax, bool):
        raise ValueError(
            f"attention.safe_softmax must be a boolean, got {safe_softmax!r}"
        )
    if causal and expected_inputs == 4:
        raise ValueError("attention cannot combine an explicit mask with causal mode")

    if expected_inputs == 4:
        mask_type = graph.value(op.inputs[3]).type
        score_shape = (
            (query_shape[0], query_shape[-2], key_shape[-2])
            if len(query_shape) == 3
            else (
                query_shape[:1]
                + (query_shape[-3], query_shape[-2], key_shape[-2])
            )
        )
        if mask_type.dtype not in {query_type.dtype, DType.BOOL}:
            raise NotImplementedError(
                "attention mask must be boolean or match the Q/K/V dtype"
            )
        if not _can_broadcast(mask_type.shape, score_shape):
            raise ValueError(
                f"attention mask shape {mask_type.shape} cannot broadcast "
                f"to {score_shape}"
            )


def _evaluate_attention_mlx(op: Operation, env: dict[str, mx.array]) -> mx.array:
    query, key, value = (env[value_id] for value_id in op.inputs[:3])
    explicit_mask = env[op.inputs[3]] if len(op.inputs) == 4 else None
    mask = explicit_mask
    if op.attrs.get("causal"):
        mask = "causal"
    scale = float(op.attrs["scale"])

    if query.ndim == 4:
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )
        if op.attrs.get("safe_softmax") and explicit_mask is not None:
            output = _zero_fully_masked_rows(explicit_mask, output)
        return output
    if query.ndim != 3:
        raise NotImplementedError(
            f"attention requires rank-3 or rank-4 inputs, got {query.shape}"
        )

    query = mx.expand_dims(query, axis=1)
    key = mx.expand_dims(key, axis=1)
    value = mx.expand_dims(value, axis=1)
    if mask is not None and mask.ndim == 3:
        mask = mx.expand_dims(mask, axis=1)
    output = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=scale, mask=mask
    )
    output = mx.squeeze(output, axis=1)
    if op.attrs.get("safe_softmax") and explicit_mask is not None:
        output = _zero_fully_masked_rows(
            mx.squeeze(explicit_mask, axis=1), output
        )
    return output


def _zero_fully_masked_rows(mask: mx.array, output: mx.array) -> mx.array:
    score_shape = output.shape[:-1] + (mask.shape[-1],)
    broadcast_mask = mx.broadcast_to(mask, score_shape)
    if broadcast_mask.dtype == mx.bool_:
        all_masked = mx.all(mx.logical_not(broadcast_mask), axis=-1, keepdims=True)
    else:
        all_masked = mx.all(broadcast_mask == -mx.inf, axis=-1, keepdims=True)
    return mx.where(all_masked, mx.array(0, dtype=output.dtype), output)
