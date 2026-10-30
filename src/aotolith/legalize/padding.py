"""Padding and repetition: operators that read their output size from attributes
rather than from the input.

``constant_pad_nd`` has a fast path - a negative pad is a crop, which is a plain
``slice`` - and a general path that materialises zeros and writes the interior.
``repeat`` is the same problem along a different axis: it is tiling, which is a
concatenation of views."""

from __future__ import annotations

from math import isfinite
from typing import Any
from ..ir import Operation, TensorType, Value
from .fx import _node_type
from .state import Lowering
from .state import Handler
from .refs import _argument_ids
from .targets import (
    _CONSTANT_PAD_TARGET,
    _PAD_MODE_TARGETS,
    _REPEAT_TARGET,
)


def _constant_pad_crops(node: object) -> bool:
    """Report whether a constant pad carries negative (cropping) entries."""

    args = getattr(node, "args", ())
    if len(args) < 2:
        return False
    pads = args[1]
    if not isinstance(pads, (tuple, list)):
        return False
    return any(
        isinstance(item, int) and not isinstance(item, bool) and item < 0
        for item in pads
    )


def _constant_pad_crop_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower a constant pad with negative entries as slice-then-pad.

    ATen derives the fill region from the distance past the source, so cropping
    the *padded* extent is the same as cropping the source first and padding
    what survives. The pad op only ever sees non-negative widths.
    """

    input_node, pads, value = _constant_pad_arguments(
        getattr(node, "args", ()), getattr(node, "kwargs", {})
    )
    input_type = _node_type(input_node)
    shape = input_type.shape
    rank = len(shape)
    padded_dims = len(pads) // 2
    if padded_dims > rank:
        raise ValueError(
            f"constant_pad_nd holds {padded_dims} padding pairs for a rank-{rank} input"
        )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, input_type.dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    current_shape = shape
    clamped: list[int] = []
    for position in range(padded_dims):
        axis = rank - 1 - position
        before, after = pads[2 * position], pads[2 * position + 1]
        clamped.extend((max(before, 0), max(after, 0)))
        if before >= 0 and after >= 0:
            continue
        start, end = max(-before, 0), shape[axis] - max(-after, 0)
        if start >= end:
            raise NotImplementedError(
                f"constant_pad_nd crop ({before}, {after}) empties axis {axis} "
                f"of size {shape[axis]}, which the static IR cannot represent"
            )
        current_shape = (
            current_shape[:axis] + (end - start,) + current_shape[axis + 1 :]
        )
        current = emit(
            "slice",
            (current,),
            {"dim": axis, "start": start, "end": end, "step": 1},
            current_shape,
        )

    expected = _node_type(node).shape
    if any(clamped):
        padded = list(current_shape)
        for position in range(padded_dims):
            axis = rank - 1 - position
            padded[axis] += clamped[2 * position] + clamped[2 * position + 1]
        current_shape = tuple(padded)
        current = emit(
            "pad",
            (current,),
            {"pads": tuple(clamped), "value": value},
            current_shape,
        )
    if current_shape != expected:
        raise ValueError(
            f"constant_pad_nd output shape {expected} does not match the "
            f"cropped and padded {current_shape}"
        )
    return tuple(operations), current


def _constant_pad_arguments(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[object, tuple[int, ...], float]:
    """Validate the constant pad arguments shared by both lowerings."""

    if len(args) not in {2, 3}:
        raise ValueError(
            f"constant_pad_nd expects two or three arguments, got {len(args)}"
        )
    input_node, pads = args[:2]
    value = args[2] if len(args) == 3 else kwargs.get("value", 0.0)
    mode = kwargs.get("mode")
    if mode is not None and mode != "constant":
        raise NotImplementedError(
            f"constant_pad_nd mode {mode!r} is not supported; only "
            "constant padding is lowered"
        )
    if not isinstance(pads, (tuple, list)) or not pads:
        raise ValueError(f"pad.pads must be a non-empty sequence, got {pads!r}")
    if len(pads) % 2 != 0:
        raise ValueError(
            "pad.pads must hold one (before, after) pair per padded "
            f"dimension, got {len(pads)} values"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in pads):
        raise ValueError(f"pad.pads must contain integers, got {pads!r}")
    if not isinstance(value, (bool, int, float)):
        raise ValueError(f"pad.value must be a scalar number, got {value!r}")
    if not isfinite(float(value)):
        raise ValueError(f"pad.value must be finite, got {value!r}")
    return input_node, tuple(int(item) for item in pads), float(value)


def _padding_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower reflect/replicate padding into slice, flip/expand and cat."""

    target = str(getattr(node, "target", ""))
    mode, padded_dims = _PAD_MODE_TARGETS[target]
    args = getattr(node, "args", ())
    if len(args) != 2:
        raise ValueError(f"{target} expects two arguments, got {len(args)}")
    input_node, padding = args
    if not isinstance(padding, (tuple, list)) or len(padding) != 2 * padded_dims:
        raise ValueError(
            f"{target}.padding must hold {2 * padded_dims} entries, got {padding!r}"
        )
    if not all(isinstance(pad, int) and not isinstance(pad, bool) for pad in padding):
        raise ValueError(f"{target}.padding must contain integers, got {padding!r}")

    input_type = _node_type(input_node)
    shape = input_type.shape
    rank = len(shape)
    if rank < padded_dims:
        raise ValueError(
            f"{target} needs at least rank {padded_dims}, got shape {shape}"
        )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, input_type.dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    current_shape = shape
    for position in range(padded_dims):
        axis = rank - 1 - position
        before, after = padding[2 * position], padding[2 * position + 1]
        size = current_shape[axis]
        if before == 0 and after == 0:
            continue
        # Negative entries crop the padded extent the way ATen does: build the
        # full pad first and slice the window afterwards, which keeps a mirror
        # window anchored on the original extent.
        pad_before, pad_after = max(before, 0), max(after, 0)
        crop_before, crop_after = max(-before, 0), max(-after, 0)
        if mode == "reflect" and (pad_before >= size or pad_after >= size):
            raise ValueError(
                f"{target} reflection padding {pad_before}/{pad_after} must stay "
                f"below the {size} elements of axis {axis}"
            )

        pieces: list[str] = []
        if pad_before:
            single_shape = current_shape[:axis] + (1,) + current_shape[axis + 1 :]
            if mode == "reflect":
                window = (
                    current_shape[:axis] + (pad_before,) + current_shape[axis + 1 :]
                )
                piece = emit(
                    "slice",
                    (current,),
                    {"dim": axis, "start": 1, "end": pad_before + 1, "step": 1},
                    window,
                )
                piece = emit("flip", (piece,), {"dims": (axis,)}, window)
            else:
                piece = emit(
                    "slice",
                    (current,),
                    {"dim": axis, "start": 0, "end": 1, "step": 1},
                    single_shape,
                )
                window = (
                    current_shape[:axis] + (pad_before,) + current_shape[axis + 1 :]
                )
                piece = emit("expand", (piece,), {"shape": window}, window)
            pieces.append(piece)
        pieces.append(current)
        if pad_after:
            if mode == "reflect":
                window = current_shape[:axis] + (pad_after,) + current_shape[axis + 1 :]
                piece = emit(
                    "slice",
                    (current,),
                    {
                        "dim": axis,
                        "start": size - pad_after - 1,
                        "end": size - 1,
                        "step": 1,
                    },
                    window,
                )
                piece = emit("flip", (piece,), {"dims": (axis,)}, window)
            else:
                single_shape = current_shape[:axis] + (1,) + current_shape[axis + 1 :]
                piece = emit(
                    "slice",
                    (current,),
                    {"dim": axis, "start": size - 1, "end": size, "step": 1},
                    single_shape,
                )
                window = current_shape[:axis] + (pad_after,) + current_shape[axis + 1 :]
                piece = emit("expand", (piece,), {"shape": window}, window)
            pieces.append(piece)

        padded_shape = (
            current_shape[:axis]
            + (size + pad_before + pad_after,)
            + current_shape[axis + 1 :]
        )
        current = (
            pieces[0]
            if len(pieces) == 1
            else emit("cat", tuple(pieces), {"dim": axis}, padded_shape)
        )
        current_shape = padded_shape

        if crop_before or crop_after:
            padded_size = padded_shape[axis]
            start, end = crop_before, padded_size - crop_after
            if start >= end:
                raise ValueError(
                    f"{target} crop ({before}, {after}) empties axis {axis} of "
                    f"size {padded_size}"
                )
            cropped_shape = (
                current_shape[:axis] + (end - start,) + current_shape[axis + 1 :]
            )
            current = emit(
                "slice",
                (current,),
                {"dim": axis, "start": start, "end": end, "step": 1},
                cropped_shape,
            )
            current_shape = cropped_shape

    expected = _node_type(node).shape
    if current_shape != expected:
        raise ValueError(
            f"{target} output shape {expected} does not match the padded "
            f"{current_shape}"
        )
    return tuple(operations), current


def _repeat_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.repeat`` into reshape/expand/reshape.

    PyTorch aligns repeats with the trailing axes and left-pads both the shape
    and the repeat list with ones, then reads ``out[i, j] = in[i % si, j % sj]``.
    Splitting every axis into ``(repeat, size)`` and broadcasting turns exactly
    that into a view chain.
    """

    args = getattr(node, "args", ())
    if len(args) != 2:
        raise ValueError(f"repeat expects two arguments, got {len(args)}")
    input_node, repeats = args
    if not isinstance(repeats, (tuple, list)) or not repeats:
        raise ValueError(
            f"repeat.repeats must be a non-empty sequence, got {repeats!r}"
        )
    if not all(
        isinstance(repeat, int) and not isinstance(repeat, bool) for repeat in repeats
    ):
        raise ValueError(f"repeat.repeats must contain integers, got {repeats!r}")
    if any(repeat < 0 for repeat in repeats):
        raise ValueError(f"repeat.repeats must not be negative, got {repeats!r}")
    if any(repeat == 0 for repeat in repeats):
        raise NotImplementedError(
            "repeat with a zero repeat produces an empty dimension, which the "
            "static IR does not model"
        )

    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    new_rank = max(rank, len(repeats))
    padded_shape = (1,) * (new_rank - rank) + input_type.shape
    padded_repeats = (1,) * (new_rank - len(repeats)) + tuple(repeats)
    output_shape = tuple(
        size * repeat for size, repeat in zip(padded_shape, padded_repeats)
    )
    if output_shape != _node_type(node).shape:
        raise ValueError(
            f"repeat output shape {_node_type(node).shape} does not match the "
            f"computed {output_shape}"
        )
    if padded_shape == output_shape:
        return (), _argument_ids(input_node, node_ids)

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        shape: tuple[int, ...],
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(shape, input_type.dtype)),
            )
        )
        return value_id

    interleaved = tuple(item for size in padded_shape for item in (1, size))
    expanded = tuple(
        item
        for size, repeat in zip(padded_shape, padded_repeats)
        for item in (repeat, size)
    )
    current = emit(
        "reshape",
        (_argument_ids(input_node, node_ids),),
        {"shape": interleaved},
        interleaved,
    )
    current = emit("expand", (current,), {"shape": expanded}, expanded)
    current = emit("reshape", (current,), {"shape": output_shape}, output_shape)
    return tuple(operations), current


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _lower_constant_pad_crop(lowering: Lowering, node: Any, target: str) -> bool:
    """A negative constant pad is a crop, which is a plain slice.

    Declines when the pad is not all-negative, so the node falls through to the
    general padding path.
    """

    if not _constant_pad_crops(node):
        return False
    produced, output_id = _constant_pad_crop_operations(
        node, lowering.node_ids, lowering.next_index()
    )
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    return True


def _lower_padding(lowering: Lowering, node: Any, target: str) -> bool:
    produced, output_id = _padding_operations(
        node, lowering.node_ids, lowering.next_index()
    )
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    return True


def _lower_repeat(lowering: Lowering, node: Any, target: str) -> bool:
    produced, output_id = _repeat_operations(
        node, lowering.node_ids, lowering.next_index()
    )
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    return True


HANDLERS: dict[str, tuple[Handler, ...]] = {
    _CONSTANT_PAD_TARGET: (_lower_constant_pad_crop,),
    **{target: (_lower_padding,) for target in _PAD_MODE_TARGETS},
    _REPEAT_TARGET: (_lower_repeat,),
}
