"""Reading layout operations: transpose, reshape, flip, roll.

The folding passes need to know when two layouts cancel, and the attention and
rope matchers need to read a transpose or a roll as a specific permutation.
"""

from __future__ import annotations

from math import prod

from ..checks import (
    _valid_int,
    _valid_ints,
)
from ...ir import Graph, Operation


def _is_identity_layout(graph: Graph, op: Operation) -> bool:
    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    rank = len(input_type.shape)
    if input_type.dtype != output_type.dtype:
        return False

    if op.op in {"contiguous", "alias"}:
        return input_type.shape == output_type.shape
    if op.op in {"reshape", "expand"}:
        return input_type.shape == output_type.shape
    if op.op == "squeeze":
        return input_type.shape == output_type.shape
    if op.op == "transpose":
        axes = _transpose_axes(op, rank)
        return axes is not None and axes[0] == axes[1]
    if op.op == "permute":
        dims = op.attrs.get("dims")
        if not isinstance(dims, (tuple, list)) or len(dims) != rank:
            return False
        normalized = _normalized_dims(op, rank)
        return normalized == tuple(range(rank))
    if op.op == "flip":
        if input_type.shape != output_type.shape:
            return False
        dims = _normalized_flip_dims(op, rank)
        return dims is not None and all(input_type.shape[dim] == 1 for dim in dims)
    if op.op == "roll":
        if input_type.shape != output_type.shape:
            return False
        shifts = _roll_shifts_by_axis(op, input_type.shape)
        return shifts is not None and _roll_shift_is_identity(shifts, input_type.shape)
    if op.op == "pad":
        if input_type.shape != output_type.shape:
            return False
        pads = op.attrs.get("pads")
        return (
            isinstance(pads, (tuple, list))
            and len(pads) % 2 == 0
            and all(_valid_int(item) and int(item) == 0 for item in pads)
        )
    if op.op == "slice":
        dim = op.attrs.get("dim")
        if not _valid_int(dim) or rank == 0 or int(dim) < -rank or int(dim) >= rank:
            return False
        axis = int(dim) + rank if int(dim) < 0 else int(dim)
        start = op.attrs.get("start")
        end = op.attrs.get("end")
        step = op.attrs.get("step")
        if start is not None and not _valid_int(start):
            return False
        if end is not None and not _valid_int(end):
            return False
        if not _valid_int(step) or int(step) <= 0:
            return False
        return slice(start, end, step).indices(input_type.shape[axis]) == (
            0,
            input_type.shape[axis],
            1,
        )
    return False


def _is_last_two_transpose(op: Operation, rank: int) -> bool:
    if op.op == "transpose":
        dim0, dim1 = op.attrs.get("dim0"), op.attrs.get("dim1")
        if not _valid_ints((dim0, dim1)):
            return False
        return {int(dim0) % rank, int(dim1) % rank} == {rank - 2, rank - 1}
    if op.op == "permute":
        dims = op.attrs.get("dims")
        if not isinstance(dims, (tuple, list)) or len(dims) != rank:
            return False
        expected = list(range(rank))
        expected[-2], expected[-1] = expected[-1], expected[-2]
        normalized = _normalized_dims(op, rank)
        return normalized == tuple(expected)
    return False


def _transpose_axes(op: Operation, rank: int) -> tuple[int, int] | None:
    dim0, dim1 = op.attrs.get("dim0"), op.attrs.get("dim1")
    if not _valid_ints((dim0, dim1)):
        return None
    normalized = tuple(
        int(dim) + rank if int(dim) < 0 else int(dim) for dim in (dim0, dim1)
    )
    if any(dim < 0 or dim >= rank for dim in normalized):
        return None
    return normalized


def _normalized_dims(op: Operation, rank: int) -> tuple[int, ...] | None:
    dims = op.attrs.get("dims")
    if not isinstance(dims, (tuple, list)) or not all(_valid_int(dim) for dim in dims):
        return None
    normalized = tuple(int(dim) + rank if int(dim) < 0 else int(dim) for dim in dims)
    if any(dim < 0 or dim >= rank for dim in normalized):
        return None
    return normalized


def _normalized_flip_dims(op: Operation, rank: int) -> tuple[int, ...] | None:
    dims = _normalized_dims(op, rank)
    if dims is None or not dims or len(set(dims)) != len(dims):
        return None
    return dims


def _roll_shifts_by_axis(
    op: Operation, shape: tuple[int, ...]
) -> dict[int | None, int] | None:
    shifts = op.attrs.get("shifts")
    if not isinstance(shifts, (tuple, list)) or not shifts:
        return None
    if not all(_valid_int(shift) for shift in shifts):
        return None

    dims = op.attrs.get("dims")
    if dims is None:
        if len(shifts) != 1:
            return None
        return {0 if len(shape) == 1 else None: int(shifts[0])}
    if not isinstance(dims, (tuple, list)) or len(dims) != len(shifts):
        return None
    if not all(_valid_int(dim) for dim in dims):
        return None

    rank = len(shape)
    shifts_by_axis: dict[int | None, int] = {}
    for shift, dim in zip(shifts, dims, strict=True):
        axis = int(dim) + rank if int(dim) < 0 else int(dim)
        if axis < 0 or axis >= rank:
            return None
        shifts_by_axis[axis] = shifts_by_axis.get(axis, 0) + int(shift)
    return shifts_by_axis


def _roll_shift_is_identity(
    shifts_by_axis: dict[int | None, int], shape: tuple[int, ...]
) -> bool:
    element_count = prod(shape)
    for axis, shift in shifts_by_axis.items():
        dimension = element_count if axis is None else shape[axis]
        if dimension <= 0 or shift % dimension != 0:
            return False
    return True
