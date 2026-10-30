"""Normalization kernels: reduce the row, then scale it in the same pass.

Layer norm and RMS norm both need a row statistic before they can write any
element, so the kernel reduces into threadgroup memory first and then re-reads
the row to normalize it. ``_emit_thread_rms_norm_prologue`` and its warp variant
are that first phase, kept as prologues because the epilogue differs between the
two normalizations while the reduction does not.
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
from .reduction import _emit_threadgroup_reduction, _reduction_thread_count


def _emit_thread_rms_norm_prologue(
    op: Operation,
    norm_position: int,
    left_position: int,
    reduction: int,
    left_batch_prefix: str,
) -> list[str]:
    """Emit an RMS scale computed by one thread over the whole reduction."""

    eps = float(op.attrs["rms_norm_eps"])
    return [
        "float norm_square_sum = 0.0f;",
        f"for (uint norm_index = 0u; norm_index < {reduction}u; norm_index += 1u) {{",
        f"    float norm_value = float(in_{left_position}"
        f"[{left_batch_prefix}norm_index]);",
        "    norm_square_sum = fma(norm_value, norm_value, norm_square_sum);",
        "}",
        f"float norm_scale = metal::rsqrt(norm_square_sum / {reduction}.0f + {eps!r});",
    ]


def _emit_warp_rms_norm_prologue(
    op: Operation,
    norm_position: int,
    left_position: int,
    reduction: int,
    left_batch_prefix: str,
) -> list[str]:
    """Emit a warp-wide RMS scale that every lane of the warp can read back."""

    eps = float(op.attrs["rms_norm_eps"])
    vectors = reduction // 4
    return [
        "const device vec<T, 4>* norm_source = "
        "reinterpret_cast<const device vec<T, 4>*>("
        f"in_{left_position} + {left_batch_prefix}0u);",
        "const device vec<T, 4>* norm_weight_values = "
        "reinterpret_cast<const device vec<T, 4>*>("
        f"in_{norm_position});",
        "float norm_square_sum = 0.0f;",
        f"for (uint norm_index = lane; norm_index < {vectors}u; norm_index += 32u) {{",
        "    vec<T, 4> current = norm_source[norm_index];",
        "    norm_square_sum = fma(float(current[0]), float(current[0]), norm_square_sum);",
        "    norm_square_sum = fma(float(current[1]), float(current[1]), norm_square_sum);",
        "    norm_square_sum = fma(float(current[2]), float(current[2]), norm_square_sum);",
        "    norm_square_sum = fma(float(current[3]), float(current[3]), norm_square_sum);",
        "}",
        f"float norm_scale = metal::rsqrt(simd_sum(norm_square_sum) / {reduction}.0f + {eps!r});",
    ]


class NormLowering:
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
            raise ValueError(f"{kind}.normalized_shape must contain positive integers")

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
        # ``layer_norm`` may also write the per-row statistics it already
        # reduces, which is what makes the tuple-returning ATen signature one
        # kernel instead of a main output plus a re-derived pair.
        stat_types = tuple(graph.value(value_id).type for value_id in op.outputs[1:])
        if stat_types and kind != "layer_norm":
            raise ValueError(f"{kind} only writes one output, got {len(op.outputs)}")
        if len(op.outputs) not in {1, 3}:
            raise ValueError(
                f"{kind} writes one or three values, got {len(op.outputs)}"
            )
        if input_type.dtype != output_type.dtype:
            raise NotImplementedError(
                f"Metal {kind} requires identical input/output dtypes"
            )
        for position, stat_type in enumerate(stat_types):
            # ATen reports the statistics with the normalized axes collapsed to
            # one, so the expected shape keeps them as singleton dimensions.
            expected_stat = input_type.shape[: -len(normalized_shape)] + (1,) * len(
                normalized_shape
            )
            if stat_type.shape != expected_stat:
                raise ValueError(
                    f"{kind} statistic {position + 1} shape {stat_type.shape} "
                    f"does not match the reduced {expected_stat}"
                )
            if stat_type.dtype != input_type.dtype:
                raise NotImplementedError(
                    f"Metal {kind} requires identical statistic dtypes"
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
        eps = float(op.attrs.get("eps", 1e-5))

        if kind == "layer_norm":
            reduction_source = (
                textwrap.dedent(
                    f"""
            float local_sum = 0.0f;
            for (uint col = tid; col < {columns}u; col += {threads}u) {{
                local_sum += float(in_0[offset + col]);
            }}
            """
                ).strip()
                + "\n"
                + _emit_threadgroup_reduction(threads, "local_sum", "total_sum")
                + "\n"
                + textwrap.dedent(
                    f"""
            float mean_value = total_sum / {columns}.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float local_squared_deviation = 0.0f;
            for (uint col = tid; col < {columns}u; col += {threads}u) {{
                float centered = float(in_0[offset + col]) - mean_value;
                local_squared_deviation += centered * centered;
            }}
            """
                ).strip()
                + "\n"
                + _emit_threadgroup_reduction(
                    threads, "local_squared_deviation", "total_deviation"
                )
                + "\n"
                + f"        float variance = total_deviation / {columns}.0f;"
            )
            reduction_result = "variance"
            centered_expression = "current - mean_value"
        else:
            reduction_source = (
                textwrap.dedent(
                    f"""
            float local_square_sum = 0.0f;
            for (uint col = tid; col < {columns}u; col += {threads}u) {{
                float current = float(in_0[offset + col]);
                local_square_sum += current * current;
            }}
            """
                ).strip()
                + "\n"
                + _emit_threadgroup_reduction(
                    threads, "local_square_sum", "total_square_sum"
                )
                + "\n"
                + f"        float mean_square = total_square_sum / {columns}.0f;"
            )
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

        # One thread per row writes the statistics the reduction already
        # computed, so the tuple ATen returns costs two stores, not a second
        # pass over the input.
        stat_write_source = (
            "        if (tid == 0) {\n"
            "            mean_out[row] = (T)mean_value;\n"
            "            rstd_out[row] = (T)inv_std;\n"
            "        }\n"
            if stat_types
            else ""
        )

        source = f"""
            uint tid = thread_position_in_threadgroup.x;
            uint row = threadgroup_position_in_grid.x;
            uint offset = row * {columns}u;
            threadgroup float scratch[{threads}];

    {textwrap.indent(reduction_source, "        ")}
            float inv_std = metal::rsqrt({reduction_result} + {eps!r}f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
    {stat_write_source}
            for (uint col = tid; col < {columns}u; col += {threads}u) {{
                float current = float(in_0[offset + col]);
                out[offset + col] = (T)({normalized_expression});
            }}
            """
        name = f"metal_{kind}_{threads}_{index}"
        write_stats = bool(stat_types)
        output_names = ("out",)
        output_shapes = [output_type.shape]
        if write_stats:
            output_names = ("out", "mean_out", "rstd_out")
            output_shapes = [output_type.shape] + [
                stat_type.shape for stat_type in stat_types
            ]
        kernel = msl._metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            output_names,
            source,
            "",
        )
        grid = (rows * threads, 1, 1)
        threadgroup = (threads, 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array | tuple[mx.array, ...]:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=output_shapes,
                output_dtypes=[dtype] * len(output_shapes),
            )
            if not write_stats:
                return outputs[0]
            return tuple(outputs)

        artifact = KernelArtifact(
            name=name,
            source=source,
            header="",
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=(output_type,) + stat_types,
            kind=kind,
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
