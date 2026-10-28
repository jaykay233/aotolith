from __future__ import annotations

import textwrap
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, prod
from typing import Any

import mlx.core as mx

from ..ir import DType, Graph, Operation, TensorType
from ..passes import LAYOUT_OPS, POINTWISE_OPS

_MLX_DTYPES = {
    DType.F16: mx.float16,
    DType.BF16: mx.bfloat16,
    DType.F32: mx.float32,
}

_ELEMENTS_PER_THREAD = 4
_MATMUL_MICROTILE = (4, 4)
_MATMUL_LARGE_K_MICROTILE = (8, 4)
_MATMUL_LARGE_K_THRESHOLD = 1024
_MATMUL_BLOCK_ROWS = 64
_MATMUL_BLOCK_COLUMNS = 64
_MATMUL_BLOCK_REDUCTION = 32
_METAL_KERNEL_CACHE_SIZE = 256

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
            elif op.op in LAYOUT_OPS:
                env[op.outputs[0]] = _evaluate_layout_mlx(
                    self.graph, op, env[op.inputs[0]]
                )
            elif op.op == "matmul":
                env[op.outputs[0]] = mx.matmul(env[op.inputs[0]], env[op.inputs[1]])
            elif op.op == "addmm":
                bias, left, right = (env[value_id] for value_id in op.inputs)
                alpha = float(op.attrs.get("alpha", 1.0))
                beta = float(op.attrs.get("beta", 1.0))
                env[op.outputs[0]] = beta * bias + alpha * mx.matmul(left, right)
            else:
                env[op.outputs[0]] = _evaluate_pointwise_mlx(op, env)

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
                if len(left_type.shape) == 2 and len(right_type.shape) == 2:
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
        for value_id in input_ids:
            input_type = graph.value(value_id).type
            if input_type.dtype != output_type.dtype:
                raise NotImplementedError(
                    "pointwise kernels require identical input/output dtypes"
                )
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
        """Emit a correctness-first fused attention kernel for common shapes."""

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
        threads = _next_power_of_two(max(query_head_dim, value_head_dim))
        if threads > 256:
            return None

        has_mask = bool(op.attrs.get("has_mask"))
        mask_shape = graph.value(op.inputs[3]).type.shape if has_mask else None
        if has_mask and not _native_attention_mask_supported(rank, mask_shape):
            return None

        batch, query_length, key_length = query_shape[0], query_shape[-2], key_shape[-2]
        query_heads = query_shape[-3] if rank == 4 else 1
        key_value_heads = key_shape[-3] if rank == 4 else 1
        group_size = query_heads // key_value_heads
        query_count = batch * query_heads * query_length

        if rank == 4:
            query_input_offset = (
                f"(((batch * {query_heads}u + head) * {query_length}u + query_pos) "
                f"* {query_head_dim}u + tid)"
            )
            key_offset = (
                f"(((batch * {key_value_heads}u + head / {group_size}u) "
                f"* {key_length}u + key) * {query_head_dim}u + tid)"
            )
            value_offset = (
                f"(((batch * {key_value_heads}u + head / {group_size}u) "
                f"* {key_length}u + key) * {value_head_dim}u + tid)"
            )
            output_offset = (
                f"(((batch * {query_heads}u + head) * {query_length}u + query_pos) "
                f"* {value_head_dim}u + tid)"
            )
        else:
            query_input_offset = (
                f"((batch * {query_length}u + query_pos) * {query_head_dim}u + tid)"
            )
            key_offset = (
                f"((batch * {key_length}u + key) * {query_head_dim}u + tid)"
            )
            value_offset = (
                f"((batch * {key_length}u + key) * {value_head_dim}u + tid)"
            )
            output_offset = (
                f"((batch * {query_length}u + query_pos) * {value_head_dim}u + tid)"
            )

        score_expression = f"partial[0] * {float(op.attrs['scale'])!r}"
        if has_mask:
            mask_offset = _attention_mask_offset_expression(rank, mask_shape)
            score_expression += f" + float(in_3[{mask_offset}])"
        if op.attrs.get("causal"):
            score_expression = (
                f"key > query_pos ? -INFINITY : ({score_expression})"
            )

        reduction_stride = threads // 2
        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint query_index = threadgroup_position_in_grid.x;
        uint batch = query_index / {query_heads * query_length}u;
        uint query_remainder = query_index % {query_heads * query_length}u;
        uint head = query_remainder / {query_length}u;
        uint query_pos = query_remainder % {query_length}u;
        threadgroup float partial[{threads}];
        threadgroup float state[4];

        if (tid == 0u) {{
            state[0] = -INFINITY;
            state[1] = 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float query_value = 0.0f;
        if (tid < {query_head_dim}u) {{
            query_value = float(in_0[{query_input_offset}]);
        }}
        float accumulator = 0.0f;

        for (uint key = 0u; key < {key_length}u; ++key) {{
            float partial_value = 0.0f;
            if (tid < {query_head_dim}u) {{
                partial_value = query_value * float(in_1[{key_offset}]);
            }}
            partial[tid] = partial_value;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
                if (tid < stride) {{
                    partial[tid] += partial[tid + stride];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            if (tid == 0u) {{
                float score = {score_expression};
                float previous_max = state[0];
                float new_max = metal::max(previous_max, score);
                float alpha = previous_max == -INFINITY
                    ? 0.0f
                    : metal::exp(previous_max - new_max);
                float probability = score == -INFINITY
                    ? 0.0f
                    : metal::exp(score - new_max);
                state[0] = new_max;
                state[1] = state[1] * alpha + probability;
                state[2] = alpha;
                state[3] = probability;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < {value_head_dim}u) {{
                accumulator = accumulator * state[2]
                    + state[3] * float(in_2[{value_offset}]);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid < {value_head_dim}u) {{
            out[{output_offset}] = (T)(state[1] == 0.0f
                ? 0.0f
                : accumulator / state[1]);
        }}
        """
        input_names = ("in_0", "in_1", "in_2", "in_3") if has_mask else (
            "in_0",
            "in_1",
            "in_2",
        )
        name = f"metal_attention_{threads}_{index}"
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
        if len(left_type.shape) != 2 or len(right_type.shape) != 2:
            raise NotImplementedError("Metal matmul currently requires rank-2 tensors")
        if len(output_type.shape) != 2:
            raise NotImplementedError("Metal matmul currently produces rank-2 tensors")

        rows, reduction = left_type.shape
        right_reduction, columns = right_type.shape
        if reduction != right_reduction:
            raise ValueError(
                f"matmul reduction dimensions differ: {reduction} and {right_reduction}"
            )
        if output_type.shape != (rows, columns):
            raise ValueError(
                f"matmul output shape {output_type.shape} != {(rows, columns)}"
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

        micro_rows, micro_columns = _select_matmul_microtile(reduction)
        source = _emit_tiled_matmul_source(
            op,
            rows,
            reduction,
            columns,
            micro_rows,
            micro_columns,
            graph,
        )
        threads = (
            (_MATMUL_BLOCK_ROWS // micro_rows)
            * (_MATMUL_BLOCK_COLUMNS // micro_columns)
        )
        row_blocks = ceil(rows / _MATMUL_BLOCK_ROWS)
        column_blocks = ceil(columns / _MATMUL_BLOCK_COLUMNS)
        name = (
            f"metal_{op.op}_tiled_"
            f"{_MATMUL_BLOCK_ROWS}x{_MATMUL_BLOCK_COLUMNS}x"
            f"{_MATMUL_BLOCK_REDUCTION}_{micro_rows}x{micro_columns}_{index}"
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
            1,
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
    common: dict[tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]], str] = {}

    for position, value_id in enumerate(input_ids):
        variable = f"input_{position}"
        offset = _offset_expression(
            graph.value(value_id).type.shape, output_shape, "elem"
        )
        lines.append(f"    float {variable} = float(in_{position}[{offset}]);")
        env[value_id] = variable

    for op_index, op in enumerate(body):
        arguments = [env[value_id] for value_id in op.inputs]
        key = (
            op.op,
            op.inputs,
            tuple((name, repr(value)) for name, value in sorted(op.attrs.items())),
        )
        existing = common.get(key)
        if existing is not None:
            env[op.outputs[0]] = existing
            continue
        output = f"temp_{op_index}"
        expression = _emit_pointwise_expression(op, arguments)
        lines.append(f"    float {output} = {expression};")
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
    scalar = repr(float(op.attrs["scalar"]))
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _emit_pointwise_expression(op: Operation, arguments: list[str]) -> str:
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
    if op.op == "exp":
        return f"metal::exp({arguments[0]})"
    if op.op == "erf":
        return f"tsc_erf({arguments[0]})"
    if op.op == "sqrt":
        return f"metal::sqrt({arguments[0]})"
    if op.op == "rsqrt":
        return f"metal::rsqrt({arguments[0]})"
    if op.op == "reciprocal":
        return f"1.0f / {arguments[0]}"
    raise NotImplementedError(f"cannot emit fused pointwise op: {op.op}")


def _select_matmul_microtile(reduction: int) -> tuple[int, int]:
    if reduction >= _MATMUL_LARGE_K_THRESHOLD:
        return _MATMUL_LARGE_K_MICROTILE
    return _MATMUL_MICROTILE


def _emit_tiled_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    micro_rows: int,
    micro_columns: int,
    graph: Graph,
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
        f"uint block_row = threadgroup_index / {column_blocks}u;",
        f"uint block_column = threadgroup_index % {column_blocks}u;",
        f"uint row_base = block_row * {block_rows}u;",
        f"uint col_base = block_column * {block_columns}u;",
        f"uint micro_row = tid / {micro_columns_per_block}u;",
        f"uint micro_column = tid % {micro_columns_per_block}u;",
        f"threadgroup float tile_a[{block_rows * block_reduction}];",
        f"threadgroup float tile_b[{block_reduction * block_columns}];",
    ]
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
    load_a = (
        f"float(in_{left_position}[(row_base + local_row) * {reduction}u "
        f"+ k_base + local_k])"
    )
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
    load_b = (
        f"float(in_{right_position}[(k_base + local_k) * {columns}u "
        f"+ col_base + local_column])"
    )
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
            lines.append(
                f"{indent}uint {linear} = "
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
        index[axis] = slice(
            int(op.attrs["start"]) if op.attrs.get("start") is not None else None,
            int(op.attrs["end"]) if op.attrs.get("end") is not None else None,
            int(op.attrs["step"]),
        )
        return value[tuple(index)]

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


def _evaluate_pointwise_mlx(op: Operation, env: dict[str, mx.array]) -> mx.array:
    arguments = [env[value_id] for value_id in op.inputs]
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
    if op.op in {"mul", "div", "add", "sub", "rsub"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "mul":
            return left * right
        if op.op == "div":
            return left / right
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
    if op.op == "exp":
        return mx.exp(arguments[0])
    if op.op == "erf":
        return mx.erf(arguments[0])
    if op.op == "sqrt":
        return mx.sqrt(arguments[0])
    if op.op == "rsqrt":
        return 1.0 / mx.sqrt(arguments[0])
    if op.op == "reciprocal":
        return 1.0 / arguments[0]
    raise NotImplementedError(f"no Apple runtime lowering for {op.op}")


def _binary_mx_operands(
    op: Operation, arguments: list[mx.array]
) -> tuple[mx.array | float, mx.array | float]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar = float(op.attrs["scalar"])
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
        if mask_type.dtype != query_type.dtype:
            raise NotImplementedError("attention mask dtype must match Q/K/V")
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
    all_masked = mx.all(broadcast_mask == -mx.inf, axis=-1, keepdims=True)
    return mx.where(all_masked, mx.array(0, dtype=output.dtype), output)
