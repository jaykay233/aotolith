"""View evaluation: the shape operations, done by touching no data.

A view is metadata, so its runtime is a reshape, a transpose, or a slice - all of
which MLX implements without a copy. Keeping them together makes it obvious that
this family cannot fail numerically: there is nothing to compute.
"""

from __future__ import annotations

import mlx.core as mx
from ...ir import Graph
from ...op_sets import LAYOUT_OPS
from ...ir import Operation
from ...codegen.dtypes import _MLX_DTYPES

_MLX_LAYOUT_OPS = LAYOUT_OPS | {"pad"}


def _evaluate_layout_mlx(graph: Graph, op: Operation, value: mx.array) -> mx.array:
    if op.op == "reshape":
        return mx.reshape(value, graph.value(op.outputs[0]).type.shape)

    rank = len(value.shape)
    if op.op == "pad":
        pads = op.attrs.get("pads")
        if (
            not isinstance(pads, tuple)
            or len(pads) % 2 != 0
            or not all(isinstance(item, int) for item in pads)
        ):
            raise ValueError(f"pad.pads must be paired integers, got {pads!r}")
        if len(pads) > 2 * rank:
            raise ValueError(f"pad has {len(pads)} values for a rank-{rank} input")
        if any(item < 0 for item in pads):
            raise ValueError(f"pad.pads must be non-negative, got {pads!r}")
        width = [(0, 0)] * rank
        for pair in range(len(pads) // 2):
            # PyTorch lists the pad pairs from the last dimension backwards.
            width[rank - 1 - pair] = (pads[2 * pair], pads[2 * pair + 1])
        source_type = graph.value(op.inputs[0]).type
        expected = tuple(
            size + before + after
            for size, (before, after) in zip(value.shape, width, strict=True)
        )
        output_type = graph.value(op.outputs[0]).type
        if expected != output_type.shape:
            raise ValueError(
                f"pad output shape {output_type.shape} does not match the "
                f"padded input {expected}"
            )
        fill = float(op.attrs.get("value", 0.0))
        return mx.pad(
            value,
            width,
            mode="constant",
            constant_values=mx.array(fill, dtype=_MLX_DTYPES[source_type.dtype]),
        )

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
        start = int(op.attrs["start"]) if op.attrs.get("start") is not None else None
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
