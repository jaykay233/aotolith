"""The spatial kernels: convolution through MLX's own primitive, and pooling
built from a strided read of the padded input.

Torch stores an activation channels-first and its weight output-major; MLX wants
the activation channels-last and the weight kernel-major. Both are the same data
with the axes in a different order, so one call is a pair of transposes around
the primitive, plus the bias Torch folds in as a final broadcast add.

MLX has no pooling kernel. What it has is ``as_strided``, and a pooling window is
exactly a strided read: pad the input once, then read the windows as a view whose
last axes are the kernel offsets and reduce over them. The padding value is the
identity of the reduction (``-inf`` for a maximum, zero for a sum), and the
counted divisor an average needs is the same view over a mask, which is why
``count_include_pad`` costs one small reduction rather than a second pass.
"""

from __future__ import annotations

import mlx.core as mx
import itertools
from math import prod
from ...ir import Graph
from ...ir import Operation
from ...codegen.dtypes import _MLX_DTYPES


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


def _element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _window_geometry(
    shape: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    ceil_mode: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The output extent of each spatial axis, and how far the last window
    reaches past the padded input.

    ``ceil_mode`` lets a window start inside the padded input and overhang its
    end, which ATen still counts as a window as long as it starts in bounds; the
    overhang is then read as padding, and the extra read is what the second half
    of the tuple reports.
    """

    sizes: list[int] = []
    overhangs: list[int] = []
    for size, step, window, spread, pad in zip(
        shape, stride, kernel, dilation, padding
    ):
        extent = size + 2 * pad - spread * (window - 1) - 1
        if ceil_mode:
            out = -(-extent // step) + 1
            if (out - 1) * step >= size + pad:
                out -= 1
        else:
            out = extent // step + 1
        if out <= 0:
            raise ValueError("pool.kernel does not fit the padded input")
        sizes.append(out)
        overhangs.append(
            max(0, (out - 1) * step + spread * (window - 1) + 1 - (size + 2 * pad))
        )
    return tuple(sizes), tuple(overhangs)


def _window_view(
    padded: mx.array,
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    sizes: tuple[int, ...],
) -> mx.array:
    """The padded input read as one row of windows per output position.

    The result's last ``len(kernel)`` axes are the offsets inside a window, so a
    reduction over them is the pooling itself.
    """

    strides = _element_strides(padded.shape)
    shape = (*padded.shape[:2], *sizes, *kernel)
    steps = (
        *strides[:2],
        *(step * strides[2 + axis] for axis, step in enumerate(stride)),
        *(spread * strides[2 + axis] for axis, spread in enumerate(dilation)),
    )
    # A strided view is reduced straight out of the padded buffer, but MLX's
    # reduction kernel misreads a view whose inner offsets are not unit steps
    # (a dilated window): it mixes neighbours into the reduction. Materializing
    # the windows first is the one copy that keeps the read exact.
    return mx.contiguous(mx.as_strided(padded, shape, steps))


def _padded_windows(
    value: mx.array,
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    ceil_mode: bool,
    fill: float,
) -> tuple[mx.array, tuple[int, ...]]:
    """The windows of ``value`` after one padding pass, and their count per axis."""

    sizes, overhangs = _window_geometry(
        value.shape[2:], kernel, stride, padding, dilation, ceil_mode
    )
    widths = [(0, 0), (0, 0)] + [
        (pad, pad + overhang) for pad, overhang in zip(padding, overhangs)
    ]
    padded = mx.pad(value, widths, constant_values=fill)
    return _window_view(padded, kernel, stride, dilation, sizes), sizes


def _window_offsets(kernel: tuple[int, ...]) -> mx.array:
    """The flat offset of every position in a window, as a broadcast constant.

    A position is the row-major offset inside one window, so in two dimensions
    the entry ``(i, j)`` reads ``i * kw + j``.
    """

    offsets = mx.zeros(tuple(kernel), dtype=mx.int32)
    for axis, width in enumerate(kernel):
        shaped = mx.arange(width, dtype=mx.int32)
        for _ in range(len(kernel) - axis - 1):
            shaped = shaped[..., None]
        offsets = offsets + shaped * prod(kernel[axis + 1 :])
    return offsets.reshape(1, 1, *([1] * len(kernel)), *kernel)


def _plane_strides(shape: tuple[int, ...]) -> list[int]:
    """How far one step along each spatial axis of ``shape`` moves the plane."""

    plane = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        plane[axis] = plane[axis + 1] * shape[axis + 1]
    return plane


def _max_window_indices(
    windows: mx.array,
    maxima: mx.array,
    sizes: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    input_shape: tuple[int, ...],
) -> mx.array:
    """Where each maximum sat, as ATen reports it: a flat index in its plane.

    The position is read from the same window as the values: entries equal to
    the maximum are kept as their offsets and the smallest one wins, so a
    repeated maximum reports the first, the way ATen does. The offset is then
    turned into input coordinates, which is where the geometry comes back in.
    """

    axes = tuple(range(-len(kernel), 0))
    expanded = maxima.reshape(*maxima.shape, *([1] * len(kernel)))
    matches = windows == expanded
    if windows.dtype in (mx.float16, mx.bfloat16, mx.float32, mx.float64):
        # A NaN maximum is not equal to itself, and ATen propagates it.
        matches = matches | (mx.isnan(windows) & mx.isnan(expanded))
    offsets = _window_offsets(kernel)
    picked = mx.min(mx.where(matches, offsets, prod(kernel)), axis=axes)

    plane = _plane_strides(input_shape[2:])
    index = None
    for axis, width in enumerate(kernel):
        coordinate = (picked // prod(kernel[axis + 1 :])) % width
        start = mx.arange(sizes[axis], dtype=mx.int32).reshape(
            [sizes[axis] if item == axis else 1 for item in range(len(sizes))]
        )
        position = start * stride[axis] - padding[axis] + coordinate * dilation[axis]
        contribution = position * plane[axis]
        index = contribution if index is None else index + contribution
    assert index is not None
    return index.astype(mx.int64)


def _count_windows(
    shape: tuple[int, ...],
    sizes: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    ceil_mode: bool,
    count_include_pad: bool,
) -> mx.array:
    """How many positions of each window are inside the region being averaged.

    An average divides by the positions its window covers, which is the whole
    window except where padding was read (when padding is excluded) or where a
    ceil-mode window overhangs the padded input. Both are read off a mask of
    ones, so the count costs one small window read rather than a second pass
    over the data.
    """

    if not count_include_pad:
        counts = mx.ones((1, 1, *shape), dtype=mx.float32)
        windows, _ = _padded_windows(
            counts, kernel, stride, padding, dilation, ceil_mode, fill=0.0
        )
        return windows
    # Padding counts as a position, so the mask already covers the padded
    # extent and only a ceil-mode overhang is excluded.
    _, overhangs = _window_geometry(shape, kernel, stride, padding, dilation, ceil_mode)
    extents = [size + 2 * pad for size, pad in zip(shape, padding)]
    counts = mx.ones((1, 1, *extents), dtype=mx.float32)
    if any(overhangs):
        counts = mx.pad(
            counts, [(0, 0), (0, 0), *[(0, overhang) for overhang in overhangs]]
        )
    return _window_view(counts, kernel, stride, dilation, sizes)


def _mlx_pool(
    graph: Graph, op: Operation, env: dict[str, mx.array]
) -> mx.array | tuple[mx.array, ...]:
    """A fixed-window pooling: one strided read, one reduction.

    The reduction is the whole difference between the two modes. A maximum pads
    with ``-inf`` and reduces with ``max``; an average pads with zero, sums, and
    divides by the count of positions that actually fell inside the region - a
    second window read over a mask, which is also what makes
    ``count_include_pad=False`` and the clamped windows of ``ceil_mode`` free.
    """

    value = env[op.inputs[0]]
    kernel = tuple(int(item) for item in op.attrs["kernel"])
    stride = tuple(int(item) for item in op.attrs["stride"])
    padding = tuple(int(item) for item in op.attrs["padding"])
    dilation = tuple(int(item) for item in op.attrs["dilation"])
    ceil_mode = bool(op.attrs.get("ceil_mode", False))
    mode = op.attrs["mode"]
    output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
    # Half precision reduces in float32, which is the precision ATen's pooling
    # kernels accumulate in.
    compute = (
        value.astype(mx.float32) if value.dtype in (mx.float16, mx.bfloat16) else value
    )
    windows, sizes = _padded_windows(
        compute,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode,
        fill=-float("inf") if mode == "max" else 0.0,
    )
    axes = tuple(range(-len(kernel), 0))
    if mode == "max":
        maxima = mx.max(windows, axis=axes)
        if len(op.outputs) == 1:
            return maxima.astype(output_dtype)
        indices = _max_window_indices(
            windows, maxima, sizes, kernel, stride, padding, dilation, value.shape
        )
        return maxima.astype(output_dtype), indices
    total = mx.sum(windows, axis=axes)
    divisor_override = op.attrs.get("divisor_override")
    if divisor_override is None:
        divisor = mx.sum(
            _count_windows(
                value.shape[2:],
                sizes,
                kernel,
                stride,
                padding,
                dilation,
                ceil_mode,
                bool(op.attrs.get("count_include_pad", True)),
            ),
            axis=axes,
        )
    else:
        divisor = float(divisor_override)
    return (total / divisor).astype(output_dtype)


def _adaptive_bounds(size: int, out: int) -> tuple[tuple[int, int], ...]:
    """The window each output position reduces over, as ATen derives it.

    An adaptive pooling has no fixed window: position ``i`` starts at
    ``floor(i * size / out)`` and ends at ``ceil((i + 1) * size / out)``, the
    smallest window that covers the output position from both directions.
    """

    return tuple((i * size // out, -(-(i + 1) * size // out)) for i in range(out))


def _mlx_adaptive_pool(
    graph: Graph, op: Operation, env: dict[str, mx.array]
) -> mx.array | tuple[mx.array, ...]:
    """Adaptive pooling, one window at a time.

    The windows no longer share a stride, so there is no view that holds them
    all: each output position reads its own slice and the results are stacked
    back into the output shape. The sizes are compile-time constants, so this is
    a fixed unrolling rather than a runtime loop.
    """

    value = env[op.inputs[0]]
    sizes = tuple(int(item) for item in op.attrs["output_size"])
    maximum = op.attrs["mode"] == "max"
    output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
    compute = (
        value.astype(mx.float32) if value.dtype in (mx.float16, mx.bfloat16) else value
    )
    bounds = [_adaptive_bounds(size, out) for size, out in zip(value.shape[2:], sizes)]
    plane = _plane_strides(value.shape[2:])
    want_indices = len(op.outputs) > 1
    reduced_values: list[mx.array] = []
    reduced_indices: list[mx.array] = []
    for position in itertools.product(*(range(size) for size in sizes)):
        selection = [slice(*bounds[axis][step]) for axis, step in enumerate(position)]
        window = compute[(slice(None), slice(None), *selection)]
        axes = tuple(range(2, window.ndim))
        if maximum:
            reduced_values.append(mx.max(window, axis=axes))
            if want_indices:
                local = mx.argmax(window.reshape(*window.shape[:2], -1), axis=-1)
                index = None
                for axis, (start, _) in enumerate(
                    bounds[dim][step] for dim, step in enumerate(position)
                ):
                    coordinate = (
                        local // prod(window.shape[3 + axis :])
                    ) % window.shape[2 + axis]
                    contribution = (start + coordinate) * plane[axis]
                    index = contribution if index is None else index + contribution
                assert index is not None
                reduced_indices.append(index.astype(mx.int64))
        else:
            reduced_values.append(mx.mean(window, axis=axes))
    shape = (*value.shape[:2], *sizes)
    values = mx.stack(
        [item.reshape(*value.shape[:2], 1) for item in reduced_values], axis=-1
    ).reshape(shape)
    values = values.astype(output_dtype)
    if not want_indices:
        return values
    indices = mx.stack(
        [item.reshape(*value.shape[:2], 1) for item in reduced_indices], axis=-1
    ).reshape(shape)
    return values, indices


def _resize_positions(
    source: int, target: int, align_corners: bool, scale: float
) -> tuple[mx.array, mx.array]:
    """The fractional source coordinate of every output position, and its floor.

    This is ATen's ``area_pixel_compute_source_index``: with ``align_corners``
    the two end points line up, and without it the output pixel's centre maps to
    the input pixel's centre, which can land at a negative coordinate that the
    caller clamps.
    """

    positions = mx.arange(target, dtype=mx.float32)
    if align_corners:
        factor = (source - 1) / (target - 1) if target > 1 else 0.0
        return positions * factor, positions
    coordinate = (positions + 0.5) * scale - 0.5
    return coordinate, positions


def _linear_axis(
    source: int, target: int, align_corners: bool, scale: float
) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...]]:
    """The two taps and weights of a linear interpolation along one axis."""

    coordinate, _ = _resize_positions(source, target, align_corners, scale)
    # ATen clamps a linear coordinate that lands outside the first pixel; the
    # upper end is handled by holding the last pixel rather than reading past it.
    coordinate = mx.maximum(coordinate, 0.0)
    low = mx.minimum(mx.floor(coordinate).astype(mx.int32), source - 1)
    high = mx.minimum(low + 1, source - 1)
    weight = coordinate - low.astype(mx.float32)
    return (low, high), (mx.ones_like(weight) - weight, weight)


def _cubic_coefficients(offset: mx.array) -> tuple[mx.array, ...]:
    """ATen's four cubic weights for one fractional offset.

    ``A = -0.75`` is the coefficient PyTorch fixes for bicubic resizing; the
    four weights sum to one for every offset in ``[0, 1)``.
    """

    a = -0.75

    def near(t: mx.array) -> mx.array:
        return ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a

    def far(t: mx.array) -> mx.array:
        return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0

    return (
        near(offset + 1.0),
        far(offset),
        far(1.0 - offset),
        near(2.0 - offset),
    )


def _cubic_axis(
    source: int, target: int, align_corners: bool, scale: float
) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...]]:
    """The four taps and weights of a cubic interpolation along one axis."""

    coordinate, _ = _resize_positions(source, target, align_corners, scale)
    lower = mx.floor(coordinate).astype(mx.int32)
    # A cubic window reaches one pixel past each end, and ATen holds the edge
    # pixel rather than shrinking the window, so every tap is clamped.
    indices = tuple(mx.clip(lower + shift, 0, source - 1) for shift in (-1, 0, 1, 2))
    return indices, _cubic_coefficients(coordinate - lower.astype(mx.float32))


def _nearest_axis(
    source: int, target: int, align_corners: bool, scale: float
) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...]]:
    """The single tap a nearest resizing reads along one axis."""

    positions = mx.arange(target, dtype=mx.float32)
    if align_corners:
        factor = (source - 1) / (target - 1) if target > 1 else 0.0
        coordinate = positions * factor
    else:
        coordinate = positions * scale
    index = mx.clip(mx.floor(coordinate).astype(mx.int32), 0, source - 1)
    return (index,), (mx.ones((target,), dtype=mx.float32),)


_RESIZE_AXES = {
    "nearest": _nearest_axis,
    "linear": _linear_axis,
    "bicubic": _cubic_axis,
}


def _broadcast_weights(weights: mx.array, axis: int, rank: int) -> mx.array:
    """Weights for one spatial axis, shaped to broadcast over the activation."""

    shape = [1] * rank
    shape[2 + axis] = weights.shape[0]
    return weights.reshape(shape)


def _mlx_resize(graph: Graph, op: Operation, env: dict[str, mx.array]) -> mx.array:
    """Resize by interpolating one spatial axis at a time.

    A separable interpolation is the whole of ATen's bilinear, linear and
    bicubic resizing: each pass blends along one axis, so a rank-5 activation
    costs a handful of reads rather than one read per corner of a 4x4x4 window.
    """

    value = env[op.inputs[0]]
    output_dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
    compute = (
        value.astype(mx.float32) if value.dtype in (mx.float16, mx.bfloat16) else value
    )
    output_size = tuple(int(item) for item in op.attrs["output_size"])
    scales = tuple(float(item) for item in op.attrs["source_scale"])
    align_corners = bool(op.attrs.get("align_corners", False))
    tap_geometry = _RESIZE_AXES[op.attrs["mode"]]
    result = compute
    for axis, (size, target, scale) in enumerate(
        zip(result.shape[2:], output_size, scales)
    ):
        taps, weights = tap_geometry(size, target, align_corners, scale)
        blended = None
        for index, weight in zip(taps, weights):
            taken = mx.take(result, index, axis=2 + axis)
            term = taken * _broadcast_weights(weight, axis, taken.ndim)
            blended = term if blended is None else blended + term
        assert blended is not None
        result = blended
    return result.astype(output_dtype)
