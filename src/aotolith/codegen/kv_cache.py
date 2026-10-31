"""The KV-cache roll as one kernel: slide the window, write the new slot.

A decode step appends one key and one value per layer to a fixed-capacity
window. The frontend expresses that in torch as a slice of the window plus a
concatenate of the two new rows, and the compiler's rewrite turns the whole
spelling into :op:`kv_cache_append`. What is left for the backend is a copy with
one row substituted, which is small - a layer's window is under a hundred
kilobytes - and therefore entirely latency rather than bandwidth. That is why it
is worth exactly one launch.

The window is contiguous. Per (half, batch, head) the slots live in one run, so
"drop the oldest slot and append one" is the same run read one row further
along: the destination is the source shifted by ``head_dim`` elements, and only
the last row of each run has to come from somewhere else. That is what lets the
copy move four elements per thread and branch exactly once.

The value half needs no rope, so it is read straight out of the projection row
the step already computed - ``value_start`` says where. That keeps three view
operators (slice, reshape, transpose) out of the graph, which is why the operator
takes the packed row rather than a shaped value.
"""

from __future__ import annotations

import mlx.core as mx
from math import prod
from typing import Any
from ..ir import Graph
from ..ir import Operation
from .dtypes import _MLX_DTYPES
from . import msl
from .msl import KernelArtifact

#: Threads per threadgroup. The window is small, so the launch is dominated by
#: getting the threadgroups resident rather than by the copy itself.
_KV_CACHE_THREADS = 256

#: Elements per thread. Eight bytes for a bf16 window: the widest move that
#: stays inside one instruction while the row arithmetic remains static.
_VECTOR = 4


def _emit_kv_cache_append_source(
    *,
    batch: int,
    heads: int,
    slots: int,
    head_dim: int,
    layer: int,
    value_start: int,
    source_width: int,
    source_steps: int,
) -> str:
    """Emit the copy-with-one-row-substituted kernel for one static layer."""

    vectors = prod((2, batch, heads, slots, head_dim)) // _VECTOR
    pair_stride = batch * heads * slots * head_dim
    head_stride = slots * head_dim
    window = (slots - 1) * head_dim
    layer_vectors = layer * 2 * pair_stride // _VECTOR
    head_vectors = head_dim // _VECTOR
    batch_heads = batch * heads
    width_vectors = source_width // _VECTOR
    value_vectors = value_start // _VECTOR
    lines = [
        f"device vec<T, {_VECTOR}>* target = "
        f"reinterpret_cast<device vec<T, {_VECTOR}>*>(out);",
        f"const device vec<T, {_VECTOR}>* window = "
        f"reinterpret_cast<const device vec<T, {_VECTOR}>*>(in_0);",
        f"const device vec<T, {_VECTOR}>* keys = "
        f"reinterpret_cast<const device vec<T, {_VECTOR}>*>(in_1);",
        f"const device vec<T, {_VECTOR}>* projected = "
        f"reinterpret_cast<const device vec<T, {_VECTOR}>*>(in_2);",
        "uint gid = thread_position_in_grid.x;",
        f"if (gid >= {vectors}u) {{ return; }}",
        f"uint within = (gid * {_VECTOR}u) % {head_stride}u;",
        f"if (within < {window}u) {{",
        # Every slot but the last is the same run, one row further along.
        f"    target[gid] = window[gid + {layer_vectors}u + {head_vectors}u];",
        "} else {",
        f"    uint block = (gid * {_VECTOR}u) / {head_stride}u;",
        f"    uint offset = (within - {window}u) / {_VECTOR}u;",
        f"    uint head_of_batch = block % {batch_heads}u;",
        f"    bool key_half = block < {batch_heads}u;",
        "    uint index = key_half",
        f"        ? head_of_batch * {head_vectors}u + offset",
        f"        : (head_of_batch / {heads}u) * {source_steps * width_vectors}u"
        f" + {(source_steps - 1) * width_vectors + value_vectors}u"
        f" + (head_of_batch % {heads}u) * {head_vectors}u + offset;",
        "    target[gid] = key_half ? keys[index] : projected[index];",
        "}",
    ]
    return "\n".join(lines)


class KVCacheLowering:
    """The KV-cache roll kernel, lowered for the Apple backend."""

    def _compile_kv_cache_append(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any] | None:
        cache_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        source_type = graph.value(op.inputs[2]).type
        layers, pair, batch, heads, slots, head_dim = cache_type.shape
        if pair != 2:
            raise ValueError(
                "kv_cache_append expects a key/value pair on axis 1, got "
                f"{cache_type.shape}"
            )
        value_start = int(op.attrs["value_start"])
        # The vector move needs the row width and both halves of the projection
        # row to be whole vectors; a shape that is not gets the MLX fallback.
        if (
            head_dim % _VECTOR
            or value_start % _VECTOR
            or source_type.shape[2] % _VECTOR
        ):
            return None
        layer = int(op.attrs["layer"])
        source = _emit_kv_cache_append_source(
            batch=batch,
            heads=heads,
            slots=slots,
            head_dim=head_dim,
            layer=layer,
            value_start=value_start,
            source_width=source_type.shape[2],
            source_steps=source_type.shape[1],
        )
        name = (
            f"metal_kv_cache_append_{layers}_{batch}_{heads}_{slots}"
            f"_{head_dim}_{layer}_{index}"
        )
        kernel = msl._metal_kernel(
            name,
            ("in_0", "in_1", "in_2"),
            ("out",),
            source,
            msl._MSL_HEADER,
        )
        vectors = prod((2, batch, heads, slots, head_dim)) // _VECTOR
        grid = (vectors, 1, 1)
        threadgroup = (min(_KV_CACHE_THREADS, vectors), 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> tuple[mx.array, ...]:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            rolled = outputs[0]
            # The two halves are views of the pair that was just written, so
            # attention reads the same bytes the cache now holds without a
            # second copy of the window.
            return (rolled, rolled[0], rolled[1])

        artifact = KernelArtifact(
            name=name,
            source=source,
            header=msl._MSL_HEADER,
            input_ids=op.inputs,
            output_ids=op.outputs,
            tensor_types=tuple(graph.value(value_id).type for value_id in op.outputs),
            kind="kv_cache_append",
        )
        return artifact, runtime
