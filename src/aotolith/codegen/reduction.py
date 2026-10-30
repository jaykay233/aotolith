"""Reduction kernels: a threadgroup cooperation, then one write.

A reduction over a strided axis cannot be one thread per output, so the kernel
is a tree: each thread accumulates its strided share, then the threadgroup
reduces across lanes in shared memory. The three helpers here are the shape of
that tree - how many threads, where each starts, and the tree itself - so the
reduction, the softmax and the normalization kernels can all reuse it.
"""

from __future__ import annotations

import textwrap
import mlx.core as mx
from math import prod
from typing import Any
from ..ir import Graph
from ..ir import Operation
from .dtypes import _MLX_DTYPES
from . import msl
from .msl import KernelArtifact


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


def _emit_threadgroup_reduction(
    threads: int,
    accumulator: str,
    label: str,
) -> str:
    """Reduce ``accumulator`` across the threadgroup and publish it in ``label``.

    A normalization row is often shorter than the threadgroup, so the classic
    barrier tree spends its time in synchronization rather than arithmetic. When
    the threadgroup covers whole SIMD groups the partial sums collapse with one
    shuffle reduction per group and a single cross-group step, which needs two
    barriers instead of ``log2(threads)``.
    """

    if threads >= 32:
        groups = threads // 32
        return textwrap.dedent(
            f"""
        {accumulator} = metal::simd_sum({accumulator});
        if ((tid & 31u) == 0u) {{
            scratch[tid >> 5u] = {accumulator};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {accumulator} = (tid < {groups}u) ? scratch[tid] : 0.0f;
        {accumulator} = metal::simd_sum({accumulator});
        if (tid == 0u) {{
            scratch[0] = {accumulator};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float {label} = scratch[0];
        """
        ).strip()

    return textwrap.dedent(
        f"""
        scratch[tid] = {accumulator};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = {threads // 2}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float {label} = scratch[0];
        """
    ).strip()


class ReductionLowering:
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
                f"{op.op} output shape {output_type.shape} != {expected_output_shape}"
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
            reduction_offset = "col" if inner_size == 1 else f"col * {inner_size}u"
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
        kernel = msl._metal_kernel(
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
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind=op.op,
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
        kernel = msl._metal_kernel(
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
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind=op.op,
        )
        return artifact, runtime
