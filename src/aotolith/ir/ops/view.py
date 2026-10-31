"""Views: the operators whose output aliases the input's storage.

A view changes shape or stride order and nothing else, so its output shape is a
pure function of the input shape and its attributes. ``LAYOUT_RULES`` holds one
such function per operator, which is what lets ``_check_layout`` verify a view
without a branch per kind, and what the layout-folding pass reads to know which
operators it may move.

``pad`` is grouped here because it is a shape operation, but it allocates rather
than viewing: the folding pass must never treat it as free to move.
"""

from __future__ import annotations

from collections.abc import Callable
from .. import Graph, Operation
from .rules import (
    AttrKind,
    AttrSpec,
    DtypeRule,
    OpContract,
    ShapeRule,
    normalize_axis,
)
from math import prod
from ...shapes import can_broadcast

LayoutRule = Callable[[Operation, tuple[int, ...]], tuple[int, ...]]


def _reshape_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    target = op.attrs["shape"]
    if not target and (prod(shape) != 1):
        raise ValueError(
            f"a rank-0 reshape needs exactly one input element, got {shape}"
        )
    if any(dim == 0 or dim < -1 for dim in target):
        raise ValueError(f"reshape.shape contains an invalid dimension: {target}")
    if target.count(-1) > 1:
        raise ValueError(f"reshape.shape has multiple inferred dimensions: {target}")
    input_count = prod(shape) or 1
    known_count = prod(dim for dim in target if dim != -1) or 1
    if -1 not in target:
        return target
    if input_count % known_count:
        raise ValueError(
            f"reshape shape {target} cannot infer a dimension for "
            f"{input_count} elements"
        )
    inferred = input_count // known_count
    return tuple(inferred if dim == -1 else dim for dim in target)


def _transpose_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    first, second = op.attrs["dim0"], op.attrs["dim1"]
    normalized = tuple(
        normalize_axis(dim, len(shape), "transpose dimension")
        for dim in (first, second)
    )
    expected = list(shape)
    expected[normalized[0]], expected[normalized[1]] = (
        expected[normalized[1]],
        expected[normalized[0]],
    )
    return tuple(expected)


def _permute_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    dims = op.attrs["dims"]
    if len(dims) != len(shape):
        raise ValueError(f"permute.dims must contain {len(shape)} integer dimensions")
    normalized = [normalize_axis(dim, len(shape), "permute dimensions") for dim in dims]
    if sorted(normalized) != list(range(len(shape))):
        raise ValueError(
            f"permute dimensions {dims} are not a permutation of rank {len(shape)}"
        )
    return tuple(shape[dim] for dim in normalized)


def _expand_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    target = op.attrs["shape"]
    if not all(dim == -1 or dim > 0 for dim in target):
        raise ValueError(f"expand.shape contains invalid dimensions: {target}")
    aligned = (1,) * (len(target) - len(shape)) + shape
    resolved = tuple(
        aligned[axis] if dim == -1 else dim for axis, dim in enumerate(target)
    )
    if not can_broadcast(aligned, resolved):
        raise ValueError(f"cannot expand shape {shape} to {resolved}")
    return resolved


def _slice_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    axis = normalize_axis(op.attrs["dim"], len(shape), "slice.dim")
    start = op.attrs["start"]
    end = op.attrs["end"]
    step = op.attrs["step"]
    sliced_length = len(range(*slice(start, end, step).indices(shape[axis])))
    return shape[:axis] + (sliced_length,) + shape[axis + 1 :]


def _identity_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape


def _flip_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    _axes(op, "dims", shape, "flip.dims")
    return shape


def _roll_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    shifts = op.attrs["shifts"]
    dims = op.attrs.get("dims")
    if dims is None:
        if len(shifts) != 1:
            raise ValueError(
                f"roll without dims requires exactly one shift, got {len(shifts)}"
            )
        return shape
    if len(dims) != len(shifts):
        raise ValueError(
            "roll shifts and dims must have the same length, got "
            f"{len(shifts)} and {len(dims)}"
        )
    _axes(op, "dims", shape, "roll.dims")
    return shape


def _axes(
    op: Operation, attribute: str, shape: tuple[int, ...], label: str
) -> tuple[int, ...]:
    values = op.attrs[attribute]
    if not values:
        raise ValueError(f"{label} must not be empty")
    normalized = tuple(normalize_axis(dim, len(shape), label) for dim in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} contains duplicate axes: {values}")
    return normalized


def _unsqueeze_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    output_rank = len(shape) + 1
    axis = normalize_axis(op.attrs["dim"], output_rank, "unsqueeze.dim")
    return shape[:axis] + (1,) + shape[axis:]


def _squeeze_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    axis = normalize_axis(op.attrs["dim"], len(shape), "squeeze.dim")
    if shape[axis] != 1:
        return shape
    return shape[:axis] + shape[axis + 1 :]


def _unfold_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    """``Tensor.unfold``: a window count at ``dim``, and the window appended last."""

    axis = normalize_axis(op.attrs["dim"], len(shape), "unfold.dim")
    size = op.attrs["size"]
    step = op.attrs["step"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"unfold.size must be a positive integer, got {size!r}")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError(f"unfold.step must be a positive integer, got {step!r}")
    if size > shape[axis]:
        raise ValueError(f"unfold.size {size} exceeds the axis length {shape[axis]}")
    windows = (shape[axis] - size) // step + 1
    # The window count takes the unfolded axis's place and the window itself is
    # appended last, so the axes after it keep their own dimensions.
    return (*shape[:axis], windows, *shape[axis + 1 :], size)


LAYOUT_RULES: dict[str, LayoutRule] = {
    "reshape": _reshape_shape,
    "transpose": _transpose_shape,
    "permute": _permute_shape,
    "expand": _expand_shape,
    "slice": _slice_shape,
    "flip": _flip_shape,
    "roll": _roll_shape,
    "unsqueeze": _unsqueeze_shape,
    "squeeze": _squeeze_shape,
    "contiguous": _identity_shape,
    "alias": _identity_shape,
    "unfold": _unfold_shape,
}


def _check_layout(graph: Graph, op: Operation) -> None:
    input_shape = graph.value(op.inputs[0]).type.shape
    output_shape = graph.value(op.outputs[0]).type.shape
    expected = LAYOUT_RULES[op.op](op, input_shape)
    if expected != output_shape:
        raise ValueError(f"{op.op} output shape {output_shape} != {expected}")


CONTRACTS: dict[str, OpContract] = {
    "reshape": OpContract(
        arity=(1, 1),
        rank=(0, 8),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("shape", AttrKind.INT_TUPLE),),
        extra=_check_layout,
    ),
    "transpose": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim0", AttrKind.INT),
            AttrSpec("dim1", AttrKind.INT),
        ),
        extra=_check_layout,
    ),
    "permute": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        extra=_check_layout,
    ),
    "expand": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("shape", AttrKind.INT_TUPLE),),
        extra=_check_layout,
    ),
    "slice": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim", AttrKind.INT),
            AttrSpec("start", AttrKind.INT_OR_NONE),
            AttrSpec("end", AttrKind.INT_OR_NONE),
            AttrSpec("step", AttrKind.INT),
        ),
        extra=_check_layout,
    ),
    "flip": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        extra=_check_layout,
    ),
    "roll": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("shifts", AttrKind.INT_TUPLE),),
        extra=_check_layout,
    ),
    "unsqueeze": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_layout,
    ),
    "squeeze": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_layout,
    ),
    "unfold": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim", AttrKind.INT),
            AttrSpec("size", AttrKind.INT),
            AttrSpec("step", AttrKind.INT),
        ),
        extra=_check_layout,
    ),
    "contiguous": OpContract(arity=(1, 1), shape=ShapeRule.FREE, extra=_check_layout),
    "alias": OpContract(arity=(1, 1), shape=ShapeRule.FREE, extra=_check_layout),
    "pad": OpContract(arity=(1, 1), dtype=DtypeRule.ANY, shape=ShapeRule.FREE),
}
