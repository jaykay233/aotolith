"""The linear-algebra operators: the ones that decompose into shaped reads, and
the factorizations and solves that each get one IR operation.

``mv`` and ``trace`` are kept because their decomposition is worse than a direct
lowering: ``mv`` really is a reshape plus a matmul, and ``trace`` is a diagonal
read plus a sum. ``diagonal`` is a view, and the builder emits the permute that
makes the diagonal contiguous so the reduction that follows is a plain one.

The factorizations are the opposite case: one ATen call returns a tuple, the
graph reads one member of it through ``getitem``, and each member is its own IR
operator. ``_LINALG_RESULTS`` below is that mapping - output index to the
operator that computes it - so adding a result is one row rather than one more
lowering function."""

from __future__ import annotations

import operator
from typing import Any
from ..ir import Operation, TensorType, Value
from .fx import _node_type
from .state import Lowering
from .state import Handler
from .refs import _argument_ids
from .targets import (
    _ADDBMM_TARGET,
    _ADDMV_TARGET,
    _DIAGONAL_COPY_TARGET,
    _DIAGONAL_TARGET,
    _LINALG_CHECK_TARGET,
    _MV_TARGET,
    _TRACE_TARGET,
    _TRILINEAR_TARGET,
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


# ---------------------------------------------------------------------------
# The factorization family
# ---------------------------------------------------------------------------


def _argument(node: Any, position: int, name: str, default: object) -> object:
    """One argument of an ATen call, positional or by keyword."""

    args = getattr(node, "args", ())
    if len(args) > position:
        return args[position]
    return getattr(node, "kwargs", {}).get(name, default)


def _triangle(node: Any, position: int, name: str, default: object) -> bool:
    """A UPLO-style argument, as the boolean the IR records."""

    value = _argument(node, position, name, default)
    if isinstance(value, bool):
        return value
    if value not in {"L", "U"}:
        raise ValueError(f"{_target(node)} triangle must be 'L' or 'U', got {value!r}")
    return value == "U"


def _target(node: Any) -> str:
    return str(getattr(node, "target", "?"))


def _plain(op_name: str) -> Any:
    """A result computed by one operator from the first operand alone."""

    def spec(node: Any) -> tuple[str, dict[str, object], str | None]:
        return op_name, {}, None

    return spec


def _solve_spec(node: Any) -> tuple[str, dict[str, object], str | None]:
    if not bool(_argument(node, 2, "left", True)):
        raise NotImplementedError(
            f"{_target(node)} over a left-hand-side matrix is not supported"
        )
    return "solve", {}, None


def _triangular_spec(node: Any) -> tuple[str, dict[str, object], str | None]:
    kwargs = getattr(node, "kwargs", {})
    upper = kwargs.get("upper")
    if not isinstance(upper, bool):
        raise ValueError(f"{_target(node)} requires a boolean 'upper'")
    if not bool(kwargs.get("left", True)):
        raise NotImplementedError(
            f"{_target(node)} over a left-hand-side matrix is not supported"
        )
    attrs = {
        "upper": upper,
        "unit_diagonal": bool(kwargs.get("unitriangular", False)),
        # ATen's ``trans`` argument is gone by the time the decomposition runs:
        # it transposes the operand instead, which the graph already contains.
        "transpose": False,
    }
    return "solve_triangular", attrs, None


def _cholesky_spec(node: Any) -> tuple[str, dict[str, object], str | None]:
    # ``cholesky`` answers the lower factor; the upper one is its transpose.
    upper = bool(_argument(node, 1, "upper", False))
    return "cholesky", {}, "transpose_last_two" if upper else None


def _eigh_spec(op_name: str) -> Any:
    def spec(node: Any) -> tuple[str, dict[str, object], str | None]:
        # ``eigvalsh`` and ``eigh`` differ only in which result is read, so both
        # arrive here; the triangle argument is the one ATen always passes on.
        return op_name, {"upper": _triangle(node, 1, "UPLO", "L")}, None

    return spec


def _svd_spec(op_name: str) -> Any:
    def spec(node: Any) -> tuple[str, dict[str, object], str | None]:
        # ATen's ``_linalg_svd`` defaults to the reduced factorization; the
        # public ``torch.linalg.svd`` passes its ``full_matrices=True`` through
        # explicitly, which is where the difference in result shapes comes from.
        full = bool(_argument(node, 1, "full_matrices", False))
        compute_uv = _argument(node, 2, "compute_uv", True)
        if not compute_uv:
            raise ValueError(f"{_target(node)} does not compute vectors")
        return op_name, {"full_matrices": full}, None

    return spec


def _qr_spec(op_name: str) -> Any:
    def spec(node: Any) -> tuple[str, dict[str, object], str | None]:
        mode = _argument(node, 1, "mode", "reduced")
        if mode != "reduced":
            raise NotImplementedError(
                f"{_target(node)} mode {mode!r} is not supported; only the "
                "reduced factorization is"
            )
        return op_name, {}, None

    return spec


def _pinv_spec(node: Any) -> tuple[str, dict[str, object], str | None]:
    kwargs = getattr(node, "kwargs", {})
    if bool(kwargs.get("hermitian", False)):
        raise NotImplementedError(f"{_target(node)} does not support hermitian=True")
    for name in ("atol", "rtol", "rcond"):
        if _argument(node, 1, name, None) is not None:
            raise NotImplementedError(
                f"{_target(node)} does not support a custom {name}; the cutoff "
                "is the decomposition's own"
            )
    return "pinv", {}, None


# Output index -> the operator that computes it. Index order is ATen's, which is
# also the order ``getitem`` selects from.
_LINALG_RESULTS: dict[str, dict[int, Any]] = {
    "aten._linalg_solve_ex.default": {0: _solve_spec},
    "aten.linalg_solve.default": {0: _solve_spec},
    "aten.linalg_solve_triangular.default": {0: _triangular_spec},
    "aten.linalg_inv_ex.default": {0: _plain("inv")},
    "aten._linalg_det.default": {0: _plain("det")},
    "aten._linalg_slogdet.default": {
        0: _plain("slogdet_sign"),
        1: _plain("slogdet_logabsdet"),
    },
    "aten.linalg_cholesky_ex.default": {0: _cholesky_spec},
    "aten.cholesky.default": {0: _cholesky_spec},
    "aten._linalg_eigh.default": {
        0: _eigh_spec("eigh_values"),
        1: _eigh_spec("eigh_vectors"),
    },
    "aten._linalg_svd.default": {
        0: _svd_spec("svd_u"),
        1: _plain("svd_values"),
        2: _svd_spec("svd_vh"),
    },
    "aten.linalg_qr.default": {0: _qr_spec("qr_q"), 1: _qr_spec("qr_r")},
    "aten.linalg_pinv.atol_rtol_tensor": {0: _pinv_spec},
    "aten.linalg_lu_factor_ex.default": {
        0: _plain("lu_factor"),
        1: _plain("lu_pivots"),
    },
    "aten.linalg_matrix_exp.default": {0: _plain("matrix_exp")},
}


def _linalg_readers(node: Any) -> tuple[dict[int, Any], bool]:
    """Which results of a factorization the graph reads.

    Returns the read result indices with the node that reads them - the
    ``getitem`` for a factorization, or the call itself when Torch answers a
    single tensor - and whether the call is that single-result form. A result
    whose only reader is ``_linalg_check_errors`` is a diagnostic the graph
    never computes with, so it is not read either.
    """

    selections = [
        user
        for user in getattr(node, "users", {})
        if getattr(user, "target", None) is operator.getitem
    ]
    if not selections:
        return {0: node}, True
    wanted: dict[int, Any] = {}
    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise ValueError(f"{_target(node)} results must be selected with getitem")
        readers = [
            reader
            for reader in user.users
            if str(getattr(reader, "target", "")) != _LINALG_CHECK_TARGET
        ]
        if readers:
            if len(user.args) != 2 or not isinstance(user.args[1], int):
                raise ValueError(f"malformed {_target(node)} getitem: {user!r}")
            wanted[user.args[1]] = user
    return wanted, False


def _linalg_operations(
    node: Any,
    node_ids: dict[object, str],
    first_op_index: int,
    target: str,
) -> tuple[tuple[tuple[Operation, Value], ...], dict[int, str]]:
    """Lower the results of one factorization that the graph actually reads.

    The operands are the call's own - a solve takes two, everything else one -
    and each result's type is the reading node's, which is where Torch recorded
    the shape the factorization promises.
    """

    spec = _LINALG_RESULTS[target]
    wanted, direct = _linalg_readers(node)
    unsupported = sorted(set(wanted) - set(spec))
    if unsupported:
        raise NotImplementedError(f"{target} result {unsupported} is not supported")
    if direct and 0 not in spec:
        raise NotImplementedError(f"{target} has no single-result form")

    operands = getattr(node, "args", ())
    if not operands:
        raise ValueError(f"{target} expects a matrix operand")
    matrix_id = _argument_ids(operands[0], node_ids)
    operations: list[tuple[Operation, Value]] = []
    produced: dict[int, str] = {}
    for index in sorted(wanted):
        output_type = _node_type(wanted[index])
        op_name, attrs, post = spec[index](node)
        if op_name in {"solve", "solve_triangular"}:
            if len(operands) < 2:
                raise ValueError(f"{target} expects a right-hand side")
            inputs = (matrix_id, _argument_ids(operands[1], node_ids))
        else:
            inputs = (matrix_id,)
        output_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op_name, inputs, (output_id,), attrs),
                Value(output_id, output_type),
            )
        )
        if post == "transpose_last_two":
            rank = len(output_type.shape)
            transposed_id = f"v{first_op_index + len(operations)}"
            operations.append(
                (
                    Operation(
                        "transpose",
                        (output_id,),
                        (transposed_id,),
                        {"dim0": rank - 2, "dim1": rank - 1},
                    ),
                    Value(transposed_id, output_type),
                )
            )
            output_id = transposed_id
        produced[index] = output_id
    return tuple(operations), produced


def _lower_linalg(lowering: Lowering, node: Any, target: str) -> bool:
    produced, ids = _linalg_operations(
        node, lowering.node_ids, lowering.next_index(), target
    )
    lowering.emit_all(produced)
    readers, direct = _linalg_readers(node)
    if direct:
        lowering.bind(node, ids[0])
        return True
    for user in readers.values():
        lowering.bind(user, ids[user.args[1]])
    for user in node.users:
        if user not in readers.values():
            lowering.dead_nodes.add(user)
    return True


def _lower_linalg_check(lowering: Lowering, node: Any, target: str) -> bool:
    """``_linalg_check_errors`` raises on a singular operand and returns nothing.

    Torch's decompositions insert it after every factorization to turn the info
    flag into an exception. The flag itself is a diagnostic, not a value the
    graph computes with, so the node is dropped: a singular operand reaches the
    device's own error instead.
    """

    lowering.dead_nodes.add(node)
    return True


def _lower_addbmm(lowering: Lowering, node: Any, target: str) -> bool:
    """``addbmm`` is a batch of products summed into one, then scaled in.

    ATen defines it as ``beta * input + alpha * sum_i(batch1_i @ batch2_i)``, so
    the batch product is an ordinary batched ``matmul``, the sum collapses the
    batch axis, and the two scale factors fold into pointwise multiplies that
    the surrounding kernel absorbs. A factor of one is not emitted at all, and a
    factor of zero drops its term instead of multiplying by it: ATen ignores the
    term entirely, so an inf or NaN hiding in it must not reach the result.
    """

    args = getattr(node, "args", ())
    if len(args) != 3:
        raise ValueError(f"{target} expects three arguments, got {len(args)}")
    input_id = _argument_ids(args[0], lowering.node_ids)
    left_id = _argument_ids(args[1], lowering.node_ids)
    right_id = _argument_ids(args[2], lowering.node_ids)
    left = lowering.values[left_id].type
    right = lowering.values[right_id].type
    if len(left.shape) != 3 or len(right.shape) != 3:
        raise ValueError(
            f"{target} expects rank-3 operands, got {left.shape} and {right.shape}"
        )
    if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[1]:
        raise ValueError(
            f"{target} operands {left.shape} and {right.shape} do not form a batch "
            "of products"
        )
    kwargs = getattr(node, "kwargs", {})
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)
    for name, value in (("beta", beta), ("alpha", alpha)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{target}.{name} must be a number, got {value!r}")

    output_type = _node_type(node)
    rows, columns = left.shape[1], right.shape[2]
    if alpha == 0:
        if beta == 0:
            # Both terms are ignored, so the result is zeros rather than a
            # product of either operand with zero, which would answer NaN for
            # any NaN or inf in them.
            lowering.bind(node, lowering.emit("full", (), {"value": 0.0}, output_type))
            return True
        lowering.bind(node, _scaled_bias(lowering, input_id, beta, output_type))
        return True

    product_id = lowering.emit(
        "matmul",
        (left_id, right_id),
        {},
        TensorType((left.shape[0], rows, columns), left.dtype),
    )
    total_id = lowering.emit(
        "sum",
        (product_id,),
        {"dims": (0,), "keepdim": False},
        TensorType((rows, columns), left.dtype),
    )
    if alpha != 1:
        total_id = lowering.emit(
            "mul",
            (total_id,),
            {"scalar": alpha, "scalar_position": "rhs"},
            TensorType((rows, columns), left.dtype),
        )
    if beta == 0:
        lowering.bind(node, total_id)
        return True
    lowering.bind(
        node,
        lowering.emit(
            "add",
            (_scaled_bias(lowering, input_id, beta, output_type), total_id),
            {},
            output_type,
        ),
    )
    return True


def _lower_addmv(lowering: Lowering, node: Any, target: str) -> bool:
    """``addmv`` is ``beta * input + alpha * mat @ vec``.

    The product reuses the shape ``mv`` already lowers through - the vector as
    one column, the rank-2 ``matmul``, and the column flattened again - so the
    only decisions left are the two scale factors. A zero factor drops its term
    rather than multiplying by it, because ATen ignores that operand and a
    multiply would carry an inf or a NaN out of a term the result does not
    contain.
    """

    args = getattr(node, "args", ())
    if len(args) != 3:
        raise ValueError(f"{target} expects three arguments, got {len(args)}")
    input_id = _argument_ids(args[0], lowering.node_ids)
    matrix_id = _argument_ids(args[1], lowering.node_ids)
    vector_id = _argument_ids(args[2], lowering.node_ids)
    matrix = lowering.values[matrix_id].type
    vector = lowering.values[vector_id].type
    if len(matrix.shape) != 2:
        raise ValueError(f"{target} requires a rank-2 matrix, got {matrix.shape}")
    rows, columns = matrix.shape
    if vector.shape != (columns,):
        raise ValueError(
            f"{target} vector shape {vector.shape} does not match the matrix "
            f"reduction size {columns}"
        )
    kwargs = getattr(node, "kwargs", {})
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)
    for name, value in (("beta", beta), ("alpha", alpha)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{target}.{name} must be a number, got {value!r}")

    output_type = _node_type(node)
    if alpha == 0:
        if beta == 0:
            lowering.bind(node, lowering.emit("full", (), {"value": 0.0}, output_type))
            return True
        lowering.bind(node, _scaled_bias(lowering, input_id, beta, output_type))
        return True

    column_id = lowering.emit(
        "reshape",
        (vector_id,),
        {"shape": (columns, 1)},
        TensorType((columns, 1), vector.dtype),
    )
    product_id = lowering.emit(
        "matmul", (matrix_id, column_id), {}, TensorType((rows, 1), matrix.dtype)
    )
    total_id = lowering.emit(
        "reshape",
        (product_id,),
        {"shape": (rows,)},
        TensorType((rows,), matrix.dtype),
    )
    if alpha != 1:
        total_id = lowering.emit(
            "mul",
            (total_id,),
            {"scalar": alpha, "scalar_position": "rhs"},
            TensorType((rows,), matrix.dtype),
        )
    if beta == 0:
        lowering.bind(node, total_id)
        return True
    lowering.bind(
        node,
        lowering.emit(
            "add",
            (_scaled_bias(lowering, input_id, beta, output_type), total_id),
            {},
            output_type,
        ),
    )
    return True


def _lower_trilinear(lowering: Lowering, node: Any, target: str) -> bool:
    """``_trilinear`` as the batched product ``nn.Bilinear`` is built from.

    Torch traces ``F.bilinear(x, y, weight, bias)`` into
    ``_trilinear(x, weight, y, expand1=[1, 3], expand2=[0], expand3=[1, 2],
    sumdim=[2, 3])`` followed by the bias add, which is a separate node. In the
    ``(batch, out, in1, in2)`` layout the expand lists give each operand the
    axes it owns: ``x`` holds batch and ``in1``, the weight holds all three of
    ``out``/``in1``/``in2``, and ``y`` holds batch and ``in2``.

    Multiplying the three operands together would materialize ``batch * out *
    in1 * in2`` values - 2**42 for a pair of 1024-wide inputs - so the ``in1``
    axis is contracted by a matmul against the weight first, and only the
    ``in2`` axis is left to the pointwise multiply that the sum then reduces.
    """

    args = getattr(node, "args", ())
    if len(args) != 7:
        raise ValueError(f"{target} expects seven arguments, got {len(args)}")
    expand1, expand2, expand3, sumdim = args[3], args[4], args[5], args[6]
    if (expand1, expand2, expand3, sumdim) != ([1, 3], [0], [1, 2], [2, 3]):
        raise NotImplementedError(
            f"{target} with expand {expand1}, {expand2}, {expand3} and sumdim "
            f"{sumdim} is outside what this IR lowers; only the bilinear layout "
            "((batch, out, in1, in2) summed over the last two axes) is supported"
        )

    first_id = _argument_ids(args[0], lowering.node_ids)
    weight_id = _argument_ids(args[1], lowering.node_ids)
    second_id = _argument_ids(args[2], lowering.node_ids)
    first = lowering.values[first_id].type
    weight = lowering.values[weight_id].type
    second = lowering.values[second_id].type
    if len(weight.shape) != 3:
        raise ValueError(f"{target} weight must be rank 3, got {weight.shape}")
    outputs, inner, contracted = weight.shape
    if first.shape[-1] != inner:
        raise ValueError(
            f"{target} first operand {first.shape} does not match the weight's "
            f"contraction size {inner}"
        )
    if second.shape[-1] != contracted:
        raise ValueError(
            f"{target} second operand {second.shape} does not match the weight's "
            f"contraction size {contracted}"
        )
    if first.shape[:-1] != second.shape[:-1]:
        raise ValueError(
            f"{target} operands {first.shape} and {second.shape} do not share "
            "a batch prefix"
        )

    dtype = first.dtype
    output_type = _node_type(node)
    batch = 1
    for dim in first.shape[:-1]:
        batch *= dim

    # The contraction runs as a batched product: the first operand is a batch of
    # row vectors and the weight a batch of ``(in1, in2)`` matrices, both rank 4
    # so the ``out`` axis of the weight and the batch axis of the input
    # broadcast against each other instead of meeting head-on.
    first_id = lowering.emit(
        "reshape",
        (first_id,),
        {"shape": (batch, 1, 1, inner)},
        TensorType((batch, 1, 1, inner), dtype),
    )
    weight_id = lowering.emit(
        "reshape",
        (weight_id,),
        {"shape": (1, outputs, inner, contracted)},
        TensorType((1, outputs, inner, contracted), dtype),
    )
    second_id = lowering.emit(
        "reshape",
        (second_id,),
        {"shape": (batch, 1, 1, contracted)},
        TensorType((batch, 1, 1, contracted), dtype),
    )
    product_id = lowering.emit(
        "matmul",
        (first_id, weight_id),
        {},
        TensorType((batch, outputs, 1, contracted), dtype),
    )
    weighted_id = lowering.emit(
        "mul",
        (product_id, second_id),
        {},
        TensorType((batch, outputs, 1, contracted), dtype),
    )
    total_id = lowering.emit(
        "sum",
        (weighted_id,),
        {"dims": (3,), "keepdim": False},
        TensorType((batch, outputs, 1), dtype),
    )
    lowering.bind(
        node,
        lowering.emit(
            "reshape", (total_id,), {"shape": output_type.shape}, output_type
        ),
    )
    return True


def _scaled_bias(
    lowering: Lowering, input_id: str, beta: float, output_type: TensorType
) -> str:
    """``beta * input``, widened to the output shape the add would have made."""

    source = lowering.values[input_id].type
    if beta != 1:
        input_id = lowering.emit(
            "mul",
            (input_id,),
            {"scalar": beta, "scalar_position": "rhs"},
            source,
        )
    if len(source.shape) == 0 and output_type.shape:
        # `expand` aligns from the right but refuses a rank-0 operand, so the
        # scalar gets the singleton shape every output axis can broadcast from.
        singleton = (1,) * len(output_type.shape)
        input_id = lowering.emit(
            "reshape",
            (input_id,),
            {"shape": singleton},
            TensorType(singleton, source.dtype),
        )
        source = TensorType(singleton, source.dtype)
    if source.shape != output_type.shape:
        input_id = lowering.emit(
            "expand",
            (input_id,),
            {"shape": output_type.shape},
            TensorType(output_type.shape, source.dtype),
        )
    return input_id


HANDLERS: dict[str, tuple[Handler, ...]] = {
    _MV_TARGET: (_lower_mv_or_trace,),
    _TRACE_TARGET: (_lower_mv_or_trace,),
    _DIAGONAL_TARGET: (_lower_diagonal,),
    _DIAGONAL_COPY_TARGET: (_lower_diagonal,),
    _LINALG_CHECK_TARGET: (_lower_linalg_check,),
    _ADDBMM_TARGET: (_lower_addbmm,),
    _ADDMV_TARGET: (_lower_addmv,),
    _TRILINEAR_TARGET: (_lower_trilinear,),
}

for _linalg_target in _LINALG_RESULTS:
    HANDLERS[_linalg_target] = (_lower_linalg,)
