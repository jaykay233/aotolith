"""Scatter evaluation, including the reductions ``array.at`` cannot do alone.

Scatter writes through an index, so the coordinates are computed once and shared
by every reduction. ``mean`` is not an ``array.at`` method - MLX would need to
divide by a per-destination count it does not track - so it is accumulated as a
sum plus a write count and divided at the end.
"""

from __future__ import annotations

import mlx.core as mx
from typing import Any

_SCATTER_REDUCE_METHODS = {
    "add": "add",
    "prod": "multiply",
    "amax": "maximum",
    "amin": "minimum",
}


_SCATTER_REDUCE_IDENTITIES = {
    "add": 0.0,
    "prod": 1.0,
    "amax": float("-inf"),
    "amin": float("inf"),
}


def _scatter_coordinates(
    source_shape: tuple[int, ...], indices: mx.array, axis: int
) -> tuple[tuple[slice, ...], tuple[int, ...], tuple[Any, ...]]:
    """Return the written region, its shape and one index per scatter axis."""

    rank = len(source_shape)
    region = tuple(
        slice(0, indices.shape[dim]) if dim != axis else slice(None)
        for dim in range(rank)
    )
    region_shape = tuple(
        source_shape[dim] if dim == axis else indices.shape[dim] for dim in range(rank)
    )
    grid: list[Any] = []
    for dim in range(rank):
        if dim == axis:
            grid.append(indices)
            continue
        view = [1] * rank
        view[dim] = indices.shape[dim]
        grid.append(mx.arange(indices.shape[dim]).reshape(view))
    return region, region_shape, tuple(grid)


def _mlx_scatter_mean(
    source: mx.array,
    indices: mx.array,
    values: Any,
    axis: int,
    include_self: bool,
) -> mx.array:
    """Accumulate values and write counts, then divide the two sums.

    ``include_self`` puts one self element on every output position, so the
    untouched part of the source keeps its value under a count of one. Without
    it an untouched position has no count at all and keeps the source value.
    """

    region, region_shape, grid = _scatter_coordinates(source.shape, indices, axis)
    ones = mx.ones(indices.shape, dtype=source.dtype)
    base = 1.0 if include_self else 0.0
    totals = mx.full(region_shape, 0.0, dtype=source.dtype)
    counts = mx.full(region_shape, base, dtype=source.dtype)
    if include_self:
        totals = source[region]
    totals = totals.at[grid].add(values)
    counts = counts.at[grid].add(ones)
    means = totals / mx.maximum(counts, 1.0)
    written = mx.put_along_axis(
        mx.zeros(region_shape, dtype=mx.bool_),
        indices,
        mx.ones(indices.shape, dtype=mx.bool_),
        axis=axis,
    )
    output = mx.array(source)
    output[region] = mx.where(written, means, source[region])
    return output


def _mlx_scatter_reduce(
    source: mx.array,
    indices: mx.array,
    values: Any,
    axis: int,
    reduce: str,
    include_self: bool,
) -> mx.array:
    """Accumulate ``values`` into ``source`` at ``indices`` along ``axis``.

    MLX has no reducing ``put_along_axis``, but advanced indexing reaches every
    coordinate the scatter form can: the non-scatter axes are coordinate grids
    broadcast to the index shape, so each source element is addressed exactly
    once per update. Duplicate indices accumulate in whatever order the device
    schedules them, which is also what CUDA ``scatter_add`` does.

    ``include_self=False`` means an updated element is the reduction of just
    the scattered values, so those positions start from the reduction's
    identity and are then spliced over the untouched source region.
    """

    if reduce == "mean":
        return _mlx_scatter_mean(source, indices, values, axis, include_self)
    method = _SCATTER_REDUCE_METHODS[reduce]
    region, region_shape, grid = _scatter_coordinates(source.shape, indices, axis)
    if include_self:
        accumulator = source[region]
        updated = getattr(accumulator.at[grid], method)(values)
        output = mx.array(source)
        output[region] = updated
        return output
    accumulator = mx.full(
        region_shape,
        _SCATTER_REDUCE_IDENTITIES[reduce],
        dtype=source.dtype,
    )
    updated = getattr(accumulator.at[grid], method)(values)
    written = mx.put_along_axis(
        mx.zeros(region_shape, dtype=mx.bool_),
        indices,
        mx.ones(indices.shape, dtype=mx.bool_),
        axis=axis,
    )
    output = mx.array(source)
    output[region] = mx.where(written, updated, source[region])
    return output
