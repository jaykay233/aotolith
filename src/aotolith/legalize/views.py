"""Shape manipulation and selection: turning one ATen node into the one or two
view operations that express it.

Each builder is a pure function of the FX node and the ids already assigned,
returning the operations and values it produced. That is what lets a view whose
ATen form carries several axes (``select``, ``unbind``, advanced indexing) lower to
a canonical ``slice``/``squeeze``/``permute`` sequence without the legalizer loop
knowing any of the details."""

from __future__ import annotations

import operator
from typing import Any
from ..ir import DType, Operation, TensorType, Value
from .fx import _node_type, _tensor_type
from .state import Lowering
from .state import Handler
from .refs import _argument_ids
from .targets import (
    _INDEX_TARGET,
    _MATRIX_TRANSPOSE_TARGET,
    _SELECT_TARGET,
    _SEQUENCE_VIEW_TARGETS,
    _SQUEEZE_ALL_TARGET,
    _UNFOLD_TARGET,
    _STACK_TARGET,
)


def _broadcast_shapes(
    shapes: tuple[tuple[int, ...], ...], context: str
) -> tuple[int, ...]:
    rank = max(len(shape) for shape in shapes)
    result = [1] * rank
    for shape in shapes:
        offset = rank - len(shape)
        for axis, dim in enumerate(shape):
            current = result[offset + axis]
            if current != dim and current != 1 and dim != 1:
                raise ValueError(f"{context} shapes cannot broadcast: {shapes}")
            result[offset + axis] = max(current, dim)
    return tuple(result)


def _sequence_view_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[Operation, Value], ...]:
    """Expand split/chunk/unbind into single-output slice or squeeze ops."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if kwargs:
        raise ValueError(f"{target} received unsupported keyword arguments: {kwargs!r}")

    if target in {"aten.split.Tensor", "aten.split_with_sizes.default"}:
        if len(args) != 3:
            raise ValueError(f"{target} expects three arguments, got {len(args)}")
        input_node, _, dim = args
    else:
        if target == "aten.unbind.int" and len(args) == 1:
            # ``torch.unbind(x)`` traces with the default ``dim=0`` elided.
            input_node = args[0]
            dim = 0
        elif len(args) == 2:
            input_node, dim = args
        else:
            raise ValueError(f"{target} expects two arguments, got {len(args)}")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")

    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")
    axis = dim + rank if dim < 0 else dim
    if axis < 0 or axis >= rank:
        raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")

    metadata = getattr(node, "meta", {}).get("val")
    if not isinstance(metadata, (tuple, list)) or not metadata:
        raise ValueError(f"missing concrete split output metadata for FX node: {node}")
    output_types = tuple(
        _tensor_type(tensor, f"output {index} of FX node {node}")
        for index, tensor in enumerate(metadata)
    )
    if target == "aten.unbind.int":
        expected_shape = input_type.shape[:axis] + input_type.shape[axis + 1 :]
        if any(output_type.shape != expected_shape for output_type in output_types):
            raise ValueError(
                f"unbind output shapes must all be {expected_shape}, got "
                f"{[output_type.shape for output_type in output_types]}"
            )
    else:
        if (
            sum(output_type.shape[axis] for output_type in output_types)
            != input_type.shape[axis]
        ):
            raise ValueError(
                f"split output lengths do not preserve input axis {axis} of shape "
                f"{input_type.shape}"
            )
        for output_type in output_types:
            if (
                output_type.shape[:axis] != input_type.shape[:axis]
                or output_type.shape[axis + 1 :] != input_type.shape[axis + 1 :]
            ):
                raise ValueError(
                    f"split output shape {output_type.shape} is incompatible with "
                    f"input shape {input_type.shape} on axis {axis}"
                )

    input_id = _argument_ids(input_node, node_ids)
    operations: list[tuple[Operation, Value]] = []
    output_ids: list[str] = []
    start = 0
    next_op_index = first_op_index
    for output_type in output_types:
        if target == "aten.unbind.int":
            sliced_id = f"v{next_op_index}"
            next_op_index += 1
            sliced_shape = input_type.shape[:axis] + (1,) + input_type.shape[axis + 1 :]
            operations.append(
                (
                    Operation(
                        "slice",
                        (input_id,),
                        (sliced_id,),
                        {
                            "dim": axis,
                            "start": start,
                            "end": start + 1,
                            "step": 1,
                        },
                    ),
                    Value(sliced_id, TensorType(sliced_shape, input_type.dtype)),
                )
            )
            start += 1
            output_id = f"v{next_op_index}"
            next_op_index += 1
            operation = Operation("squeeze", (sliced_id,), (output_id,), {"dim": axis})
        else:
            output_id = f"v{next_op_index}"
            next_op_index += 1
            end = start + output_type.shape[axis]
            operation = Operation(
                "slice",
                (input_id,),
                (output_id,),
                {"dim": axis, "start": start, "end": end, "step": 1},
            )
            start = end
        output_ids.append(output_id)
        operations.append((operation, Value(output_id, output_type)))

    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise ValueError(f"{target} outputs must be selected with getitem")
        user_args = getattr(user, "args", ())
        if len(user_args) != 2 or not isinstance(user_args[1], int):
            raise ValueError(f"malformed sequence output selection: {user!r}")
        index = user_args[1]
        if index < 0 or index >= len(output_ids):
            raise ValueError(f"{target} output index {index} is out of range")
        node_ids[user] = output_ids[index]

    return tuple(operations)


def _select_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[Operation, Value], ...]:
    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if kwargs:
        raise ValueError(f"{target} received unsupported keyword arguments: {kwargs!r}")
    if len(args) != 3:
        raise ValueError(f"{target} expects three arguments, got {len(args)}")

    input_node, dim, index = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(f"{target}.index must be an integer, got {index!r}")

    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")
    axis = dim + rank if dim < 0 else dim
    if axis < 0 or axis >= rank:
        raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")
    resolved_index = index + input_type.shape[axis] if index < 0 else index
    if resolved_index < 0 or resolved_index >= input_type.shape[axis]:
        raise ValueError(
            f"{target}.index {index} is outside axis {axis} with length "
            f"{input_type.shape[axis]}"
        )

    sliced_shape = input_type.shape[:axis] + (1,) + input_type.shape[axis + 1 :]
    sliced_id = f"v{first_op_index}"
    output_id = f"v{first_op_index + 1}"
    sliced_type = TensorType(sliced_shape, input_type.dtype)
    output_type = _node_type(node)
    expected_shape = input_type.shape[:axis] + input_type.shape[axis + 1 :]
    if output_type.shape != expected_shape or output_type.dtype != input_type.dtype:
        raise ValueError(
            f"{target} output metadata {(output_type.shape, output_type.dtype)} does "
            f"not match expected {(expected_shape, input_type.dtype)}"
        )

    return (
        (
            Operation(
                "slice",
                (_argument_ids(input_node, node_ids),),
                (sliced_id,),
                {
                    "dim": axis,
                    "start": resolved_index,
                    "end": resolved_index + 1,
                    "step": 1,
                },
            ),
            Value(sliced_id, sliced_type),
        ),
        (
            Operation(
                "squeeze",
                (sliced_id,),
                (output_id,),
                {"dim": axis},
            ),
            Value(output_id, output_type),
        ),
    )


def _stack_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str, TensorType]:
    """Expand stack into unsqueeze ops followed by one cat op."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if len(args) not in {1, 2}:
        raise ValueError(f"stack expects one or two arguments, got {len(args)}")
    tensors = args[0]
    if not isinstance(tensors, (tuple, list)) or not tensors:
        raise ValueError("stack expects a non-empty tensor sequence")
    if len(args) == 2:
        if "dim" in kwargs:
            raise ValueError("stack received duplicate dim arguments")
        dim = args[1]
    else:
        dim = kwargs.get("dim", 0)
    unknown_kwargs = set(kwargs) - {"dim"}
    if unknown_kwargs:
        raise ValueError(
            f"stack received unsupported keyword arguments: {sorted(unknown_kwargs)}"
        )
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"stack.dim must be an integer, got {dim!r}")

    input_types = tuple(_node_type(tensor) for tensor in tensors)
    first_type = input_types[0]
    if any(input_type != first_type for input_type in input_types[1:]):
        raise ValueError(
            "stack inputs must have identical shapes and dtypes, got "
            f"{[input_type.shape for input_type in input_types]}"
        )

    output_type = _node_type(node)
    output_rank = len(first_type.shape) + 1
    axis = dim + output_rank if dim < 0 else dim
    if axis < 0 or axis >= output_rank:
        raise ValueError(f"stack.dim {dim} exceeds output rank {output_rank}")
    expected_shape = first_type.shape[:axis] + (len(tensors),) + first_type.shape[axis:]
    if output_type.shape != expected_shape or output_type.dtype != first_type.dtype:
        raise ValueError(
            f"stack output metadata {(output_type.shape, output_type.dtype)} does not "
            f"match expected {(expected_shape, first_type.dtype)}"
        )

    operations: list[tuple[Operation, Value]] = []
    expanded_ids: list[str] = []
    expanded_shape = first_type.shape[:axis] + (1,) + first_type.shape[axis:]
    for offset, tensor in enumerate(tensors):
        expanded_id = f"v{first_op_index + offset}"
        expanded_ids.append(expanded_id)
        operations.append(
            (
                Operation(
                    "unsqueeze",
                    (_argument_ids(tensor, node_ids),),
                    (expanded_id,),
                    {"dim": axis},
                ),
                Value(expanded_id, TensorType(expanded_shape, first_type.dtype)),
            )
        )

    output_id = f"v{first_op_index + len(tensors)}"
    operations.append(
        (
            Operation("cat", tuple(expanded_ids), (output_id,), {"dim": axis}),
            Value(output_id, output_type),
        )
    )
    return tuple(operations), output_id, output_type


def _advanced_index_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType, tuple[int, ...] | None]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    if kwargs:
        raise ValueError(
            f"index received unsupported keyword arguments: {sorted(kwargs)}"
        )
    if len(args) != 2:
        raise ValueError(f"index expects two arguments, got {len(args)}")

    source, indices = args
    if not isinstance(indices, (tuple, list)) or not indices:
        raise ValueError("index expects a non-empty sequence of indices")
    source_type = _node_type(source)
    if len(indices) > len(source_type.shape):
        raise ValueError(
            f"index has {len(indices)} indices for rank-{len(source_type.shape)} input"
        )

    index_nodes = [index for index in indices if index is not None]
    if not index_nodes:
        raise ValueError("index requires at least one tensor index")
    index_types = tuple(_node_type(index) for index in index_nodes)
    for index_type in index_types:
        if index_type.dtype not in {DType.I32, DType.I64}:
            raise ValueError(
                f"index indices must be int32 or int64, got {index_type.dtype.value}"
            )

    indexed_axes = tuple(
        axis for axis, index in enumerate(indices) if index is not None
    )
    first_axis = indexed_axes[0]
    last_axis = indexed_axes[-1]
    # PyTorch inserts the broadcast index dimensions where the indexed axes
    # were when they are adjacent, and at the front when they are separated by
    # a fixed axis. Moving the separated axes to the front turns the second
    # rule into the first, so the gather itself stays the same op.
    permute_dims: tuple[int, ...] | None = None
    if indexed_axes != tuple(range(first_axis, last_axis + 1)):
        permute_dims = indexed_axes + tuple(
            axis for axis in range(len(source_type.shape)) if axis not in indexed_axes
        )

    broadcast_shape = _broadcast_shapes(
        tuple(index_type.shape for index_type in index_types),
        "advanced index",
    )
    if permute_dims is None:
        expected_shape = (
            source_type.shape[:first_axis]
            + broadcast_shape
            + source_type.shape[last_axis + 1 :]
        )
        op_axes = indexed_axes
    else:
        remaining = tuple(
            source_type.shape[axis] for axis in permute_dims[len(indexed_axes) :]
        )
        expected_shape = broadcast_shape + remaining
        op_axes = tuple(range(len(indexed_axes)))
    expected_type = TensorType(expected_shape, source_type.dtype)
    output_type = _node_type(node)
    if output_type != expected_type:
        raise ValueError(
            f"index output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(expected_type.shape, expected_type.dtype)}"
        )

    inputs = (_argument_ids(source, node_ids),) + tuple(
        _argument_ids(index, node_ids) for index in index_nodes
    )
    return inputs, {"axes": op_axes}, output_type, permute_dims


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
# A handler consumes one FX node and returns True. Returning False means "not
# this shape after all", and the loop tries the next handler for the target, so
# a conditional lowering (constant padding with a crop) can decline and let the
# generic path take it.


def _lower_matrix_transpose(lowering: Lowering, node: Any, target: str) -> bool:
    input_node = node.args[0]
    input_type = _node_type(input_node)
    if len(input_type.shape) != 2:
        raise ValueError(
            f"aten.t is only supported for rank-2 tensors, got shape {input_type.shape}"
        )
    output_id = lowering.emit(
        "transpose",
        (_argument_ids(input_node, lowering.node_ids),),
        {"dim0": 1, "dim1": 0},
        _node_type(node),
    )
    lowering.bind(node, output_id)
    return True


def _lower_squeeze_all(lowering: Lowering, node: Any, target: str) -> bool:
    # ``torch.squeeze(x)`` drops every singleton axis, which is a pure reshape
    # and needs no per-axis sweep.
    output_type = _node_type(node)
    output_id = lowering.emit(
        "reshape",
        (_argument_ids(node.args[0], lowering.node_ids),),
        {"shape": output_type.shape},
        output_type,
    )
    lowering.bind(node, output_id)
    return True


def _lower_select(lowering: Lowering, node: Any, target: str) -> bool:
    produced = _select_operations(node, lowering.node_ids, lowering.next_index())
    lowering.emit_all(produced)
    lowering.bind(node, produced[-1][0].outputs[0])
    return True


def _lower_advanced_index(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type, permute_dims = _advanced_index_operation(
        node, lowering.node_ids
    )
    if permute_dims is not None:
        # ATen's advanced indexing reads the indexed axes first; the core `index`
        # op keeps the source layout, so the permutation is made explicit here
        # rather than being a special case inside the kernel.
        source_type = lowering.values[inputs[0]].type
        permuted_shape = tuple(source_type.shape[axis] for axis in permute_dims)
        permuted_id = lowering.emit(
            "permute",
            (inputs[0],),
            {"dims": permute_dims},
            TensorType(permuted_shape, source_type.dtype),
        )
        inputs = (permuted_id,) + inputs[1:]
    output_id = lowering.emit("index", inputs, attrs, output_type)
    lowering.bind(node, output_id)
    return True


def _lower_unfold(lowering: Lowering, node: Any, target: str) -> bool:
    """``Tensor.unfold``: one strided read, so the window is never copied."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if len(args) < 4:
        raise ValueError(f"{target} expects four arguments, got {len(args)}")
    input_id = _argument_ids(args[0], lowering.node_ids)
    input_type = lowering.values[input_id].type
    axis = args[1] if not kwargs else kwargs.get("dimension", args[1])
    size = args[2] if not kwargs else kwargs.get("size", args[2])
    step = args[3] if not kwargs else kwargs.get("step", args[3])
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise ValueError(f"{target}.dimension must be an integer, got {axis!r}")
    normalized = axis % len(input_type.shape)
    # The window count sits where the unfolded axis was and the window is
    # appended last, which is the shape ATen's view reports.
    windows = (input_type.shape[normalized] - size) // step + 1
    shape = (
        *input_type.shape[:normalized],
        windows,
        *input_type.shape[normalized + 1 :],
        size,
    )
    attrs = {"dim": axis, "size": size, "step": step}
    lowering.bind(
        node,
        lowering.emit(
            "unfold",
            (input_id,),
            attrs,
            TensorType(shape, input_type.dtype),
        ),
    )
    return True


def _lower_sequence_view(lowering: Lowering, node: Any, target: str) -> bool:
    lowering.emit_all(
        _sequence_view_operations(node, lowering.node_ids, lowering.next_index())
    )
    return True


def _lower_stack(lowering: Lowering, node: Any, target: str) -> bool:
    produced, output_id, output_type = _stack_operations(
        node, lowering.node_ids, lowering.next_index()
    )
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    lowering.values[output_id] = Value(output_id, output_type)
    return True


HANDLERS: dict[str, tuple[Handler, ...]] = {
    _MATRIX_TRANSPOSE_TARGET: (_lower_matrix_transpose,),
    _SQUEEZE_ALL_TARGET: (_lower_squeeze_all,),
    _UNFOLD_TARGET: (_lower_unfold,),
    _SELECT_TARGET: (_lower_select,),
    _INDEX_TARGET: (_lower_advanced_index,),
    _STACK_TARGET: (_lower_stack,),
    **{target: (_lower_sequence_view,) for target in _SEQUENCE_VIEW_TARGETS},
}
