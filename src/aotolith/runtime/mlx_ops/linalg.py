"""The factorization and solve family, which MLX implements on the CPU only.

Every ``mx.linalg`` decomposition - ``solve``, ``inv``, ``det``, ``cholesky``,
``eigh``, ``svd``, ``qr``, ``pinv``, ``lu_factor`` - raises "not yet supported on
the GPU" on this build, so each call names the CPU stream explicitly. The
operands are ordinary MLX arrays, so MLX stages them across and the result stays
part of the same graph; the cost is the trip, not a change in semantics.

The factorizations themselves are float32 only ("arrays must have type float32
or float64"), so a half operand is promoted and the result cast back to the
dtype the plan declared. Torch does the same internally: it runs the LAPACK path
in float and returns the input dtype.

Two spellings need translating rather than forwarding. MLX reports
``lu_factor`` pivots zero-based where ATen reports them one-based, and it
returns a full pair of singular-vector matrices where a reduced factorization
wants the leading ``min(m, n)`` slices, so the slices are taken here.
"""

from __future__ import annotations

import mlx.core as mx
from math import ceil, log2
from typing import Any
from ...ir import Graph
from ...ir import Operation
from ...ir.ops.linalg import solve_vector_rhs
from ...codegen.dtypes import _MLX_DTYPES

_CPU = mx.cpu

_LINALG_OPS = frozenset(
    {
        "solve",
        "solve_triangular",
        "inv",
        "det",
        "slogdet_sign",
        "slogdet_logabsdet",
        "cholesky",
        "eigh_values",
        "eigh_vectors",
        "svd_values",
        "svd_u",
        "svd_vh",
        "qr_q",
        "qr_r",
        "pinv",
        "lu_factor",
        "lu_pivots",
        "matrix_exp",
    }
)


def _float_operand(value: mx.array) -> mx.array:
    """The float32 operand the factorizations require."""

    if value.dtype == mx.float32:
        return value
    return value.astype(mx.float32)


def _declare(graph: Graph, op: Operation, value: mx.array) -> mx.array:
    """Cast a result to the dtype the plan declared for it."""

    dtype = _MLX_DTYPES[graph.value(op.outputs[0]).type.dtype]
    return value if value.dtype == dtype else value.astype(dtype)


def _solve_argument(left: mx.array, right: mx.array) -> tuple[mx.array, mx.array, bool]:
    """Broadcast only the batch axes, which is what Torch broadcasts.

    Torch reads the right-hand side as one vector when it is 1-D or when it is
    exactly the matrix's own batch shape plus its equation axis; anything else
    is a stack of columns. Broadcasting the operands as they are would line the
    vector up against the matrix's own columns, so the vector form gets the
    trailing axis a column needs, and the batch prefixes are broadcast apart
    from the matrix axes.
    """

    vector = solve_vector_rhs(left.shape, right.shape)
    if vector:
        right = right[..., None]
    batch = mx.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = mx.broadcast_to(left, batch + left.shape[-2:])
    right = mx.broadcast_to(right, batch + right.shape[-2:])
    return left, right, vector


def _matrix_exp(value: mx.array) -> mx.array:
    """``exp`` of a batch of matrices by scaling and squaring.

    The norm picks the scaling: with ``A / 2**s`` small, a truncated Taylor
    series is accurate, and ``s`` squarings of the result undo the scaling. The
    norm is a value, not a shape, so it is read out of the graph - this is the
    one operator here whose structure depends on the operand's contents.
    """

    magnitude = float(mx.max(mx.sum(mx.abs(value), axis=-1)))
    squarings = max(0, ceil(log2(magnitude))) if magnitude > 0.5 else 0
    scaled = value / (2**squarings)
    result = mx.eye(scaled.shape[-1], dtype=scaled.dtype)
    term = result
    for order in range(1, 12):
        # The scaling turns the series into a chain of products, and a float32
        # product on the GPU is far coarser than the CPU's, which is where the
        # rest of this family already runs.
        term = mx.matmul(term, scaled, stream=_CPU) / order
        result = result + term
    for _ in range(squarings):
        result = mx.matmul(result, result, stream=_CPU)
    return result


def _linalg_result(op: Operation, env: dict[str, mx.array]) -> Any:
    """The MLX result of one factorization, or the selection it stands for."""

    value = _float_operand(env[op.inputs[0]])
    name = op.op
    if name in {"solve", "solve_triangular"}:
        left, right, vector = _solve_argument(value, _float_operand(env[op.inputs[1]]))
        if name == "solve":
            result = mx.linalg.solve(left, right, stream=_CPU)
            return result[..., 0] if vector else result
        if op.attrs["unit_diagonal"]:
            # A unitriangular operand reads its diagonal as ones; replacing the
            # diagonal with ones is the same system, and MLX has no flag for it.
            diagonal = mx.eye(left.shape[-1], dtype=left.dtype) == 1
            left = mx.where(diagonal, mx.array(1.0, dtype=left.dtype), left)
        result = mx.linalg.solve_triangular(
            left, right, upper=bool(op.attrs["upper"]), stream=_CPU
        )
        return result[..., 0] if vector else result
    if name == "inv":
        return mx.linalg.inv(value, stream=_CPU)
    if name == "det":
        return mx.linalg.det(value, stream=_CPU)
    if name in {"slogdet_sign", "slogdet_logabsdet"}:
        sign, logabsdet = mx.linalg.slogdet(value, stream=_CPU)
        return sign if name == "slogdet_sign" else logabsdet
    if name == "cholesky":
        return mx.linalg.cholesky(value, stream=_CPU)
    if name in {"eigh_values", "eigh_vectors"}:
        upper = bool(op.attrs["upper"])
        eigenvalues, eigenvectors = mx.linalg.eigh(
            value, UPLO="U" if upper else "L", stream=_CPU
        )
        return eigenvalues if name == "eigh_values" else eigenvectors
    if name in {"svd_values", "svd_u", "svd_vh"}:
        if name == "svd_values":
            return mx.linalg.svd(value, compute_uv=False, stream=_CPU)
        u, _, vh = mx.linalg.svd(value, stream=_CPU)
        rows, columns = value.shape[-2:]
        rank = min(rows, columns)
        if name == "svd_u":
            # MLX always answers the full U; the reduced form is its leading
            # columns, which is what ``full_matrices=False`` asks for.
            return u if op.attrs["full_matrices"] else u[..., :rank]
        return vh if op.attrs["full_matrices"] else vh[..., :rank, :]
    if name in {"qr_q", "qr_r"}:
        q, r = mx.linalg.qr(value, stream=_CPU)
        return q if name == "qr_q" else r
    if name == "pinv":
        return mx.linalg.pinv(value, stream=_CPU)
    if name in {"lu_factor", "lu_pivots"}:
        factors, pivots = mx.linalg.lu_factor(value, stream=_CPU)
        # ATen's pivots are LAPACK's: one-based.
        return factors if name == "lu_factor" else pivots + 1
    if name == "matrix_exp":
        return _matrix_exp(value)
    raise NotImplementedError(f"no MLX implementation for {name}")


def _evaluate_linalg_mlx(graph: Graph, op: Operation, env: dict[str, mx.array]) -> Any:
    return _declare(graph, op, _linalg_result(op, env))
