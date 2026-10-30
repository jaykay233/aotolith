"""Indexing-family codegen: rope, and the note on what is not here.

Most layout and indexing operations never reach codegen at all - they are MLX
views and go to ``runtime.mlx_ops``. ``rope`` is the exception: it is a layout
operation in the IR but the kernel does the rotation in place, four neighbouring
columns per thread, which is why it has a hand-written source at all.
"""

from __future__ import annotations

import mlx.core as mx
from math import prod
from typing import Any
from ..ir import Graph
from ..ir import Operation
from .dtypes import _MLX_DTYPES
from .expr import _ELEMENTS_PER_THREAD, _offset_expression
from . import msl
from .msl import KernelArtifact, _MSL_HEADER


def _emit_rope_source(
    columns: int,
    half: int,
    output_shape: tuple[int, ...],
    cosine_shape: tuple[int, ...],
    sine_shape: tuple[int, ...],
) -> str:
    """Emit the rotary embedding kernel for one contiguous output tensor.

    ``out = x * cos + rotate_half(x) * sin`` and ``rotate_half`` pairs element
    ``d`` of the head with element ``d ^ half``, negating the first half. One
    thread owns four neighbouring columns; because the head dimension and the
    half both divide by four, a group never straddles that boundary, so the
    partner loads stay contiguous and vectorizable.
    """

    vectorized = columns % 4 == 0 and half % 4 == 0
    if not vectorized:
        raise NotImplementedError(
            "Metal rope requires a head dimension divisible by four"
        )
    lines = [
        "uint group = thread_position_in_grid.x;",
        "uint base = group * 4u;",
        f"uint column = base % {columns}u;",
        "uint row = base - column;",
        f"uint partner = row + (column < {half}u ? column + {half}u : column - {half}u);",
        f"float sign = column < {half}u ? -1.0f : 1.0f;",
        "vec<T, 4> current = *(const device vec<T, 4>*)(in_0 + base);",
        "vec<T, 4> rotated = *(const device vec<T, 4>*)(in_0 + partner);",
        "vec<T, 4> result;",
    ]
    for offset in range(4):
        element = f"(base + {offset}u)"
        cosine_offset = _offset_expression(cosine_shape, output_shape, element)
        sine_offset = _offset_expression(sine_shape, output_shape, element)
        lines.append(
            f"result[{offset}] = (T)(float(current[{offset}]) "
            f"* float(in_1[{cosine_offset}]) "
            f"+ sign * float(rotated[{offset}]) * float(in_2[{sine_offset}]));"
        )
    lines.append("*(device vec<T, 4>*)(out + base) = result;")
    return "\n".join(f"        {line}" for line in lines)


class IndexLowering:
    def _compile_rope(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        _, cosine_id, sine_id = op.inputs
        output_type = graph.value(op.outputs[0]).type
        cosine_type = graph.value(cosine_id).type
        sine_type = graph.value(sine_id).type
        columns = output_type.shape[-1]
        half = columns // 2
        source = _emit_rope_source(
            columns,
            half,
            output_type.shape,
            cosine_type.shape,
            sine_type.shape,
        )
        name = f"metal_rope_{columns}_{index}"
        kernel = msl._metal_kernel(
            name,
            ("in_0", "in_1", "in_2"),
            ("out",),
            source,
            _MSL_HEADER,
        )
        element_count = prod(output_type.shape)
        groups = element_count // _ELEMENTS_PER_THREAD
        grid = (groups, 1, 1)
        threadgroup = (min(256, groups), 1, 1)
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
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=(output_type,),
            kind="rope",
        )
        return artifact, runtime
