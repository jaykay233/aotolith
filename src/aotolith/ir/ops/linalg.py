"""Linear algebra: the factorizations and solves, and the shapes they promise.

Torch returns most of these as a tuple - ``eigh`` gives values and vectors,
``svd`` gives three matrices, ``det`` answers its determinant next to the
factorization it took to get there - while the graph usually reads one of them.
Each result is therefore its own operator, the same way ``sort`` and ``argsort``
are split: reading only the singular values should not pay for the vectors, and
an unused result needs no value at all.

Every operator here takes a batch of matrices in its trailing two axes and keeps
the leading axes, so the shapes below are the batch axes followed by whatever
the factorization produces. A ``solve`` is the one case a single operand cannot
describe: its batch axes are the broadcast of both operands' batch axes.
"""

from __future__ import annotations

from .. import DType, Graph, Operation, TensorType
from .rules import AttrKind, AttrSpec, DtypeRule, OpContract, ShapeRule
from .shared import _require_output


def _matrix_shape(graph: Graph, op: Operation) -> tuple[int, int]:
    """The trailing matrix shape of the first operand, checked for rank."""

    shape = graph.value(op.inputs[0]).type.shape
    if len(shape) < 2:
        raise ValueError(f"{op.op} requires a matrix, got shape {shape}")
    return shape[-2], shape[-1]


def _check_square(graph: Graph, op: Operation) -> None:
    rows, columns = _matrix_shape(graph, op)
    if rows != columns:
        raise ValueError(f"{op.op} requires a square matrix, got {rows}x{columns}")


def _require_shape(graph: Graph, op: Operation, shape: tuple[int, ...]) -> None:
    input_type = graph.value(op.inputs[0]).type
    _require_output(graph, op, TensorType(shape, input_type.dtype), "output shape")


def _check_determinant(graph: Graph, op: Operation) -> None:
    _check_square(graph, op)
    _require_shape(graph, op, graph.value(op.inputs[0]).type.shape[:-2])


def _check_cholesky(graph: Graph, op: Operation) -> None:
    _check_square(graph, op)
    input_type = graph.value(op.inputs[0]).type
    _require_output(graph, op, input_type, "output type")


def _check_eigenvalues(graph: Graph, op: Operation) -> None:
    _check_square(graph, op)
    _require_shape(graph, op, graph.value(op.inputs[0]).type.shape[:-1])


def _check_singular_values(graph: Graph, op: Operation) -> None:
    rows, columns = _matrix_shape(graph, op)
    batch = graph.value(op.inputs[0]).type.shape[:-2]
    _require_shape(graph, op, batch + (min(rows, columns),))


def _check_singular_vectors(graph: Graph, op: Operation) -> None:
    rows, columns = _matrix_shape(graph, op)
    batch = graph.value(op.inputs[0]).type.shape[:-2]
    rank = min(rows, columns)
    full = op.attrs["full_matrices"]
    if op.op == "svd_u":
        expected = batch + ((rows, rows) if full else (rows, rank))
    else:
        expected = batch + ((columns, columns) if full else (rank, columns))
    _require_shape(graph, op, expected)


def _broadcast_batch(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...] | None:
    """The batch axes two operands agree on, or ``None`` when they cannot."""

    result: list[int] = []
    for position in range(1, max(len(left), len(right)) + 1):
        left_size = left[-position] if position <= len(left) else 1
        right_size = right[-position] if position <= len(right) else 1
        if left_size != right_size and 1 not in {left_size, right_size}:
            return None
        result.append(max(left_size, right_size))
    return tuple(reversed(result))


def solve_vector_rhs(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    """Whether a solve's right-hand side is one vector, not a stack of columns.

    This is Torch's own rule: a 1-D operand, or one whose shape is exactly the
    matrix's batch prefix followed by the equation axis, is a vector and
    answers one dimension less. Anything else is a stack of columns, whose
    second-to-last axis holds the equations. The runtime asks the same question
    of the arrays it is handed, so it imports this rather than restating it.
    """

    return len(right) == 1 or (len(right) == len(left) - 1 and right == left[:-1])


def _check_solve(graph: Graph, op: Operation) -> None:
    _check_square(graph, op)
    left = graph.value(op.inputs[0]).type
    right = graph.value(op.inputs[1]).type
    if not right.shape:
        raise ValueError(f"{op.op} requires a right-hand side, got a scalar")
    size = left.shape[-1]
    vector = solve_vector_rhs(left.shape, right.shape)
    equations = right.shape[-1] if vector else right.shape[-2]
    if equations != size:
        raise ValueError(
            f"{op.op} right-hand side {right.shape} does not match the "
            f"{size}x{size} matrix"
        )
    prefix = right.shape[:-1] if vector else right.shape[:-2]
    batch = _broadcast_batch(left.shape[:-2], prefix)
    if batch is None:
        raise ValueError(
            f"{op.op} batch axes {left.shape[:-2]} and {prefix} cannot broadcast"
        )
    expected = batch + right.shape[-1:] if vector else batch + right.shape[-2:]
    _require_output(graph, op, TensorType(expected, left.dtype), "output shape")


def _check_qr(graph: Graph, op: Operation) -> None:
    rows, columns = _matrix_shape(graph, op)
    batch = graph.value(op.inputs[0]).type.shape[:-2]
    reduced = min(rows, columns)
    expected = batch + ((rows, reduced) if op.op == "qr_q" else (reduced, columns))
    _require_shape(graph, op, expected)


def _check_pseudo_inverse(graph: Graph, op: Operation) -> None:
    rows, columns = _matrix_shape(graph, op)
    batch = graph.value(op.inputs[0]).type.shape[:-2]
    _require_shape(graph, op, batch + (columns, rows))


def _check_pivots(graph: Graph, op: Operation) -> None:
    _check_square(graph, op)
    output_type = graph.value(op.outputs[0]).type
    if output_type.dtype != DType.I32:
        raise ValueError(
            f"{op.op} must return I32 pivots, got {output_type.dtype.value}"
        )
    _require_output(
        graph,
        op,
        TensorType(graph.value(op.inputs[0]).type.shape[:-1], DType.I32),
        "output shape",
    )


_UPPER_ATTRS = (AttrSpec("upper", AttrKind.BOOL),)


CONTRACTS: dict[str, OpContract] = {
    "solve": OpContract(
        arity=(2, 2),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_solve,
    ),
    # ``solve_triangular`` differs from ``solve`` only in which triangle it
    # reads and whether the operand is transposed first; the shapes are the same.
    "solve_triangular": OpContract(
        arity=(2, 2),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("upper", AttrKind.BOOL),
            AttrSpec("unit_diagonal", AttrKind.BOOL),
            AttrSpec("transpose", AttrKind.BOOL),
        ),
        extra=_check_solve,
    ),
    "inv": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.SAME,
        extra=_check_square,
    ),
    "det": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_determinant,
    ),
    "slogdet_sign": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_determinant,
    ),
    "slogdet_logabsdet": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_determinant,
    ),
    "cholesky": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.SAME,
        extra=_check_cholesky,
    ),
    "eigh_values": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        attrs=_UPPER_ATTRS,
        extra=_check_eigenvalues,
    ),
    "eigh_vectors": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.SAME,
        attrs=_UPPER_ATTRS,
        extra=_check_cholesky,
    ),
    "svd_values": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_singular_values,
    ),
    "svd_u": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("full_matrices", AttrKind.BOOL),),
        extra=_check_singular_vectors,
    ),
    "svd_vh": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("full_matrices", AttrKind.BOOL),),
        extra=_check_singular_vectors,
    ),
    "qr_q": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_qr,
    ),
    "qr_r": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_qr,
    ),
    "pinv": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        extra=_check_pseudo_inverse,
    ),
    "lu_factor": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.SAME,
        extra=_check_square,
    ),
    "lu_pivots": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        extra=_check_pivots,
    ),
    "matrix_exp": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.SAME,
        extra=_check_square,
    ),
}
