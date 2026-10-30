"""The small linear-algebra operators that have no native kernel and decompose into
shaped reads of their operand.

``mv`` and ``trace`` are kept because their decomposition is worse than a direct
lowering: ``mv`` really is a reshape plus a matmul, and ``trace`` is a diagonal
read plus a sum. ``diagonal`` is a view, and the builder emits the permute that
makes the diagonal contiguous so the reduction that follows is a plain one."""

from __future__ import annotations

from typing import Any
from ..ir import Operation, TensorType, Value
from .fx import _node_type
from .state import Lowering
from .state import Handler
from .refs import _argument_ids
from .targets import (
    _DIAGONAL_COPY_TARGET,
    _DIAGONAL_TARGET,
    _MV_TARGET,
    _TRACE_TARGET,
)


def _diagonal_arguments(
    node: object,
) -> tuple[object, int, int, int]:
    """Normalize the ``diagonal`` offset and axis arguments."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError("diagonal expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank < 2:
        raise NotImplementedError(
            f"diagonal requires rank 2 or more, got shape {input_type.shape}"
        )

    offset = args[1] if len(args) > 1 else kwargs.get("offset", 0)
    dim1 = args[2] if len(args) > 2 else kwargs.get("dim1", 0)
    dim2 = args[3] if len(args) > 3 else kwargs.get("dim2", 1)
    for name, value in (("offset", offset), ("dim1", dim1), ("dim2", dim2)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"diagonal.{name} must be an integer, got {value!r}")
    axis1, axis2 = dim1 % rank, dim2 % rank
    if axis1 == axis2:
        raise ValueError(f"diagonal.dim1 and dim2 must differ, both are {axis1}")
    return input_node, offset, axis1, axis2


def _mv_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.mv`` into reshape plus a rank-2 ``matmul``."""

    args = getattr(node, "args", ())
    if len(args) != 2:
        raise ValueError(f"mv expects two arguments, got {len(args)}")
    matrix_id = _argument_ids(args[0], node_ids)
    vector_id = _argument_ids(args[1], node_ids)
    matrix_type = _node_type(args[0])
    if len(matrix_type.shape) != 2:
        raise ValueError(f"mv requires a rank-2 matrix, got shape {matrix_type.shape}")
    rows, columns = matrix_type.shape
    vector_type = _node_type(args[1])
    if vector_type.shape != (columns,):
        raise ValueError(
            f"mv vector shape {vector_type.shape} does not match the matrix "
            f"reduction size {columns}"
        )
    dtype = matrix_type.dtype

    operations: list[tuple[Operation, Value]] = []
    column_id = f"v{first_op_index}"
    operations.append(
        (
            Operation("reshape", (vector_id,), (column_id,), {"shape": (columns, 1)}),
            Value(column_id, TensorType((columns, 1), vector_type.dtype)),
        )
    )
    product_id = f"v{first_op_index + 1}"
    operations.append(
        (
            Operation("matmul", (matrix_id, column_id), (product_id,), {}),
            Value(product_id, TensorType((rows, 1), dtype)),
        )
    )
    output_id = f"v{first_op_index + 2}"
    operations.append(
        (
            Operation("reshape", (product_id,), (output_id,), {"shape": (rows,)}),
            Value(output_id, _node_type(node)),
        )
    )
    return tuple(operations), output_id


def _trace_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.trace`` into a diagonal strided slice plus a reduction."""

    args = getattr(node, "args", ())
    if len(args) != 1:
        raise ValueError(f"trace expects one argument, got {len(args)}")
    input_id = _argument_ids(args[0], node_ids)
    input_type = _node_type(args[0])
    if len(input_type.shape) != 2:
        raise ValueError(
            f"trace requires a rank-2 matrix, got shape {input_type.shape}"
        )
    rows, columns = input_type.shape
    length = min(rows, columns)
    dtype = input_type.dtype

    operations: list[tuple[Operation, Value]] = []
    flattened_id = f"v{first_op_index}"
    operations.append(
        (
            Operation(
                "reshape",
                (input_id,),
                (flattened_id,),
                {"shape": (rows * columns,)},
            ),
            Value(flattened_id, TensorType((rows * columns,), dtype)),
        )
    )
    step = columns + 1
    diagonal_id = f"v{first_op_index + 1}"
    operations.append(
        (
            Operation(
                "slice",
                (flattened_id,),
                (diagonal_id,),
                {"dim": 0, "start": 0, "end": length * step, "step": step},
            ),
            Value(diagonal_id, TensorType((length,), dtype)),
        )
    )
    output_id = f"v{first_op_index + 2}"
    operations.append(
        (
            Operation(
                "sum",
                (diagonal_id,),
                (output_id,),
                {"dims": (0,), "keepdim": False},
            ),
            Value(output_id, _node_type(node)),
        )
    )
    return tuple(operations), output_id


def _diagonal_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.diagonal`` into permute/reshape/slice.

    ``A[i, i + offset]`` sits at flat index ``i * (m + 1) + offset`` once the
    two axes are trailing and flattened, so the diagonal is a strided slice
    with no copy.
    """

    input_node, offset, dim1, dim2 = _diagonal_arguments(node)
    input_type = _node_type(input_node)
    shape = input_type.shape
    rank = len(shape)
    rows, columns = shape[dim1], shape[dim2]
    start_row = max(0, -offset)
    start_column = max(0, offset)
    length = min(rows - start_row, columns - start_column)
    if length <= 0:
        raise NotImplementedError(
            f"diagonal offset {offset} leaves no elements on a {rows}x{columns} "
            "matrix, and the static IR has no zero-sized dimensions"
        )
    remaining = tuple(
        size for axis, size in enumerate(shape) if axis not in {dim1, dim2}
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
    order = tuple(axis for axis in range(rank) if axis not in {dim1, dim2}) + (
        dim1,
        dim2,
    )
    if order != tuple(range(rank)):
        current = emit(
            "permute", (current,), {"dims": order}, tuple(shape[axis] for axis in order)
        )
    flattened_shape = remaining + (rows * columns,)
    current = emit("reshape", (current,), {"shape": flattened_shape}, flattened_shape)
    step = columns + 1
    start = start_row * columns + start_column
    sliced_shape = remaining + (length,)
    current = emit(
        "slice",
        (current,),
        {
            "dim": len(remaining),
            "start": start,
            "end": start + length * step,
            "step": step,
        },
        sliced_shape,
    )
    expected = _node_type(node).shape
    if sliced_shape != expected:
        raise ValueError(
            f"diagonal output shape {expected} does not match the strided "
            f"view {sliced_shape}"
        )
    return tuple(operations), current


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _lower_mv_or_trace(lowering: Lowering, node: Any, target: str) -> bool:
    builder = {_MV_TARGET: _mv_operations, _TRACE_TARGET: _trace_operations}[target]
    produced, output_id = builder(node, lowering.node_ids, lowering.next_index())
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    return True


def _lower_diagonal(lowering: Lowering, node: Any, target: str) -> bool:
    produced, output_id = _diagonal_operations(
        node, lowering.node_ids, lowering.next_index()
    )
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    return True


HANDLERS: dict[str, tuple[Handler, ...]] = {
    _MV_TARGET: (_lower_mv_or_trace,),
    _TRACE_TARGET: (_lower_mv_or_trace,),
    _DIAGONAL_TARGET: (_lower_diagonal,),
    _DIAGONAL_COPY_TARGET: (_lower_diagonal,),
}
