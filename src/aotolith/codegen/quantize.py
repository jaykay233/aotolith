"""The unpack kernels: packed words back into a matrix, streamed once.

A graph keeps an explicit unpack node in exactly one situation: the
matmul that reads it is too wide to run on the packed form (a prefill pass
multiplies the same weight by a whole prompt, so the weight stream is no longer
the cost and a packed kernel would only add decoding work to a compute-bound
pass). The MLX fallback would unpack with one elementwise pass per code and a stack,
writing several full-sized temporaries per layer; this kernel does it in one
pass, one thread per packed word, so the traffic is the 250 MB of words in and
the 990 MB of weights out and nothing else.
"""

from __future__ import annotations

import mlx.core as mx
from typing import Any

from ..ir import Graph, Operation
from .dtypes import _MLX_DTYPES
from .msl import KernelArtifact

#: Threads per threadgroup. The work is pure streaming, so this only has to
#: fill a threadgroup's worth of in-flight loads.
_DEQUANTIZE_THREADS = 256


def _emit_dequantize_source(
    rows: int,
    words: int,
    groups: int,
    words_per_group: int,
) -> str:
    """One thread per packed word: decode eight nibbles, scale, two vector stores.

    The nibble is moved into the sign position and shifted back down with an
    arithmetic shift, which sign-extends it in two instructions; a mask plus a
    conditional subtract would be four.
    """

    inner = words * 8
    return f"""
        uint index = thread_position_in_grid.x;
        if (index >= {rows * words}u) {{ return; }}
        uint row = index / {words}u;
        uint word_index = index - row * {words}u;
        uint word = (uint)in_0[index];
        T scale = (T)float(in_1[row * {groups}u + word_index / {words_per_group}u]);
        vec<T, 4> low;
        vec<T, 4> high;
        low[0] = (T)((int)(word << 28u) >> 28u);
        low[1] = (T)((int)(word << 24u) >> 28u);
        low[2] = (T)((int)(word << 20u) >> 28u);
        low[3] = (T)((int)(word << 16u) >> 28u);
        high[0] = (T)((int)(word << 12u) >> 28u);
        high[1] = (T)((int)(word << 8u) >> 28u);
        high[2] = (T)((int)(word << 4u) >> 28u);
        high[3] = (T)((int)word >> 28u);
        uint base = row * {inner}u + word_index * 8u;
        device vec<T, 4>* destination =
            reinterpret_cast<device vec<T, 4>*>(out + base);
        destination[0] = low * scale;
        destination[1] = high * scale;
        """


def _emit_dequantize_int8_source(
    rows: int,
    words: int,
    groups: int,
    words_per_group: int,
) -> str:
    """One thread per packed word: decode four bytes, scale, one vector store.

    Four codes fill a ``vec<T, 4>`` exactly, so this is a single store where
    the four-bit version needs two. The sign extension is the same arithmetic
    shift the four-bit path uses, one byte at a time.
    """

    inner = words * 4
    return f"""
        uint index = thread_position_in_grid.x;
        if (index >= {rows * words}u) {{ return; }}
        uint row = index / {words}u;
        uint word_index = index - row * {words}u;
        uint word = (uint)in_0[index];
        T scale = (T)float(in_1[row * {groups}u + word_index / {words_per_group}u]);
        vec<T, 4> values;
        values[0] = (T)((int)(word << 24u) >> 24);
        values[1] = (T)((int)(word << 16u) >> 24);
        values[2] = (T)((int)(word << 8u) >> 24);
        values[3] = (T)((int)word >> 24);
        uint base = row * {inner}u + word_index * 4u;
        device vec<T, 4>* destination =
            reinterpret_cast<device vec<T, 4>*>(out + base);
        destination[0] = values * scale;
        """


class QuantizationLowering:
    """Metal lowering for the weight unpacks."""

    def _compile_dequantize_i4(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        packed_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        group_size = int(op.attrs["group_size"])
        rows, words = packed_type.shape
        inner = words * 8
        if group_size % 8 != 0:
            raise ValueError(
                "dequantize_i4 needs a group size that covers whole packed words, "
                f"got {group_size}"
            )
        source = _emit_dequantize_source(
            rows,
            words,
            inner // group_size,
            group_size // 8,
        )
        threads = _DEQUANTIZE_THREADS
        total = rows * words
        name = f"metal_dequantize_i4_g{group_size}_{index}"
        kernel = mx.fast.metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            "",
        )
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(packed: mx.array, scales: mx.array) -> mx.array:
            outputs = kernel(
                inputs=[packed, scales],
                template=[("T", dtype)],
                grid=((total + threads - 1) // threads * threads, 1, 1),
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
            kind="dequantize_i4",
        )
        return artifact, runtime

    def _compile_dequantize_i8(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        packed_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        group_size = int(op.attrs["group_size"])
        rows, words = packed_type.shape
        inner = words * 4
        if group_size % 4 != 0:
            raise ValueError(
                "dequantize_i8 needs a group size that covers whole packed words, "
                f"got {group_size}"
            )
        source = _emit_dequantize_int8_source(
            rows,
            words,
            inner // group_size,
            group_size // 4,
        )
        threads = _DEQUANTIZE_THREADS
        total = rows * words
        name = f"metal_dequantize_i8_g{group_size}_{index}"
        kernel = mx.fast.metal_kernel(
            name,
            tuple(f"in_{position}" for position in range(len(op.inputs))),
            ("out",),
            source,
            "",
        )
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(packed: mx.array, scales: mx.array) -> mx.array:
            outputs = kernel(
                inputs=[packed, scales],
                template=[("T", dtype)],
                grid=((total + threads - 1) // threads * threads, 1, 1),
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
            kind="dequantize_i8",
        )
        return artifact, runtime
