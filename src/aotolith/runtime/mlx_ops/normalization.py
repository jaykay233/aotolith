"""The interpreter's side of the operators that were kernel-only.

Four families had no MLX implementation at all: ``softmax``/``safe_softmax``,
``sum``/``mean``, ``layer_norm`` and ``rms_norm``. The Apple backend always
generates a kernel for each of them, so the fallback path never ran and never
needed one. That stops being true the moment a plan is built by a backend that
generates nothing - the MLX reference target - so the implementations live here,
next to the rest of the interpreter.

They are written to match the generated kernels, not to be merely close: the
same axis normalization, the same biased variance for ``layer_norm``, the same
``-inf -> 0`` rule for ``safe_softmax``, and the same statistic shapes for a
``layer_norm`` that also writes its mean and inverse standard deviation. A
disagreement here would show up as the reference plan and the kernel plan
producing different numbers from the same graph.
"""

from __future__ import annotations

import mlx.core as mx

from ...codegen.dtypes import _MLX_DTYPES
from ...ir import Operation, TensorType


def _normalized_axes(op: Operation, input_type: TensorType) -> tuple[int, ...]:
    """The trailing axes ``normalized_shape`` names, as positive indices."""

    normalized_shape = op.attrs.get("normalized_shape")
    if not isinstance(normalized_shape, tuple) or not normalized_shape:
        raise ValueError(
            f"{op.op}.normalized_shape must be a non-empty tuple"
        )
    rank = len(input_type.shape)
    if len(normalized_shape) > rank:
        raise ValueError(
            f"{op.op}.normalized_shape {normalized_shape} exceeds input rank {rank}"
        )
    start = rank - len(normalized_shape)
    return tuple(range(start, rank))


def _normalization_inputs(
    op: Operation, arguments: tuple[mx.array, ...]
) -> tuple[mx.array, mx.array | None, mx.array | None]:
    """Split a normalization's operands into ``(value, weight, bias)``."""

    has_weight = op.attrs.get("has_weight")
    if not isinstance(has_weight, bool):
        raise ValueError(f"{op.op}.has_weight must be a boolean")
    has_bias = op.attrs.get("has_bias", False)
    if not isinstance(has_bias, bool):
        raise ValueError(f"{op.op}.has_bias must be a boolean")
    if op.op == "rms_norm" and has_bias:
        raise ValueError("rms_norm.has_bias is not part of the operator")

    expected = 1 + int(has_weight) + int(has_bias)
    if len(arguments) != expected:
        raise ValueError(
            f"{op.op} expects {expected} inputs, got {len(arguments)}"
        )

    value = arguments[0]
    position = 1
    weight = None
    if has_weight:
        weight = arguments[position]
        position += 1
    bias = arguments[position] if has_bias else None
    return value, weight, bias


def _evaluate_normalization_mlx(
    op: Operation,
    arguments: tuple[mx.array, ...],
    output_types: tuple[TensorType, ...],
) -> mx.array | tuple[mx.array, ...]:
    """Run ``layer_norm`` or ``rms_norm`` on MLX, including their statistics."""

    value, weight, bias = _normalization_inputs(op, arguments)
    eps = float(op.attrs.get("eps", 1e-5))
    axes = _normalized_axes(op, output_types[0])
    rank = len(value.shape)
    if axes != tuple(range(rank - len(axes), rank)):
        raise ValueError(f"{op.op}.normalized_shape must name trailing axes")

    # The kernels accumulate in float32 whatever the input dtype is, so the
    # interpreter does too: doing this in the input dtype would put the two
    # paths on different statistics and the difference would look like a bug in
    # one of them.
    compute = value.astype(mx.float32)
    if op.op == "layer_norm":
        mean = mx.mean(compute, axis=axes, keepdims=True)
        centered = compute - mean
        variance = mx.mean(centered * centered, axis=axes, keepdims=True)
        inverse = mx.rsqrt(variance + eps)
        normalized = centered * inverse
    else:
        square_mean = mx.mean(compute * compute, axis=axes, keepdims=True)
        inverse = mx.rsqrt(square_mean + eps)
        normalized = compute * inverse

    if weight is not None:
        normalized = normalized * weight.astype(mx.float32)
    if bias is not None:
        normalized = normalized + bias.astype(mx.float32)

    result = normalized.astype(_MLX_DTYPES[output_types[0].dtype])
    if len(op.outputs) == 1:
        return result
    if op.op != "layer_norm":
        raise ValueError(f"{op.op} writes one value, got {len(op.outputs)}")
    # The statistics are float32 in the op's contract, and they are the same
    # values the main output was normalized with rather than a second reduction.
    return (
        result,
        mean.astype(mx.float32),
        inverse.astype(mx.float32),
    )


def _evaluate_softmax_mlx(op: Operation, value: mx.array) -> mx.array:
    """Softmax along one axis, with the masked-row rule of ``safe_softmax``."""

    rank = len(value.shape)
    if rank == 0:
        raise NotImplementedError(f"{op.op} requires a ranked tensor")
    dim = op.attrs.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{op.op}.dim must be an integer, got {dim!r}")
    axis = dim % rank
    if op.op == "softmax":
        return mx.softmax(value, axis=axis)

    # ``safe_softmax`` answers zero for a row whose entries are all ``-inf``.
    # The generated kernel does that by clamping the row maximum, and skipping
    # the clamp here would produce NaN through 0/0 - the one disagreement that
    # would be visible as a NaN rather than as a decimal.
    masked = value == -mx.inf
    candidates = mx.where(
        masked, mx.array(-mx.inf, dtype=value.dtype), value
    )
    maximum = mx.max(candidates, axis=axis, keepdims=True)
    maximum = mx.where(
        mx.isinf(maximum), mx.array(0.0, dtype=value.dtype), maximum
    )
    exponentials = mx.where(
        masked,
        mx.array(0.0, dtype=value.dtype),
        mx.exp(value - maximum),
    )
    total = mx.sum(exponentials, axis=axis, keepdims=True)
    safe_total = mx.where(total == 0, mx.array(1.0, dtype=value.dtype), total)
    return mx.where(
        total == 0,
        mx.array(0.0, dtype=value.dtype),
        exponentials / safe_total,
    )


def _evaluate_reduction_mlx(
    op: Operation, value: mx.array, output_type: TensorType
) -> mx.array:
    """``sum`` and ``mean`` over an explicit axis set, with ``keepdim``."""

    rank = len(value.shape)
    dims = op.attrs.get("dims")
    if not isinstance(dims, tuple) or not dims:
        raise ValueError(f"{op.op}.dims must be a non-empty tuple")
    axes: list[int] = []
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"{op.op}.dims must contain integers")
        axis = dim % rank
        if axis in axes:
            raise ValueError(f"{op.op}.dims must not contain duplicates")
        axes.append(axis)
    keepdim = op.attrs.get("keepdim")
    if not isinstance(keepdim, bool):
        raise ValueError(f"{op.op}.keepdim must be a boolean")

    reduced = (
        mx.sum(value, axis=tuple(axes), keepdims=keepdim)
        if op.op == "sum"
        else mx.mean(value, axis=tuple(axes), keepdims=keepdim)
    )
    dtype = _MLX_DTYPES[output_type.dtype]
    if reduced.dtype == dtype:
        return reduced
    return reduced.astype(dtype)
