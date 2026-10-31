"""Convolution, through the kernel MLX already has.

Torch stores an activation channels-first and its weight output-major; MLX wants
the activation channels-last and the weight kernel-major. Both are the same data
with the axes in a different order, so one call is a pair of transposes around
the primitive, plus the bias Torch folds in as a final broadcast add.
"""

from __future__ import annotations

import mlx.core as mx
from ...ir import Graph
from ...ir import Operation


def _channels_last(value: mx.array) -> mx.array:
    """A channels-first activation as MLX wants it: (batch, *spatial, channels)."""

    return value.transpose(0, *range(2, value.ndim), 1)


def _kernel_last(weight: mx.array) -> mx.array:
    """An output-major weight as MLX wants it: (out, *kernel, in)."""

    return weight.transpose(0, *range(2, weight.ndim), 1)


def _geometry(op: Operation, name: str) -> int | tuple[int, ...]:
    """One geometry attribute, as the primitive of this rank spells it.

    ``conv1d`` takes plain integers for its single axis while ``conv2d`` and
    ``conv3d`` take one entry per axis, so the tuple collapses at rank one.
    """

    values = tuple(int(item) for item in op.attrs[name])
    return values[0] if len(values) == 1 else values


def _mlx_convolution(graph: Graph, op: Operation, env: dict[str, mx.array]) -> mx.array:
    value = env[op.inputs[0]]
    weight = env[op.inputs[1]]
    primitive = {1: mx.conv1d, 2: mx.conv2d, 3: mx.conv3d}.get(value.ndim - 2)
    if primitive is None:
        raise NotImplementedError(
            f"convolution over {value.ndim - 2} spatial axes is not supported"
        )
    result = primitive(
        _channels_last(value),
        _kernel_last(weight),
        _geometry(op, "stride"),
        _geometry(op, "padding"),
        _geometry(op, "dilation"),
        int(op.attrs["groups"]),
    )
    # The kernel answers channels-last; the IR is channels-first throughout.
    spatial = result.ndim - 2
    result = result.transpose(0, result.ndim - 1, *range(1, result.ndim - 1))
    if len(op.inputs) == 3:
        bias = env[op.inputs[2]]
        result = result + bias.reshape(1, -1, *([1] * spatial))
    return result
