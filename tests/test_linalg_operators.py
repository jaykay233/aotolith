"""The linear-algebra family: solves, and the factorizations they build on.

MLX runs every one of these on the CPU stream - its ``mx.linalg`` module has no
GPU implementation - so what the tests check is not a kernel but the translation
around it: the dtype the factorization is allowed to run in, the batch axes
Torch broadcasts and the matrix axes it does not, and which of a tuple result's
members the graph actually reads.

The decompositions whose answer is only defined up to a sign or an ordering -
eigenvectors, singular vectors, a QR pair, an LU pair - are checked by what they
reconstruct rather than element by element, because Torch and MLX are free to
pick different signs. The arithmetic of a reconstruction runs in NumPy: MLX's
float32 matrix product on the GPU is far coarser than the CPU factorization it
would be checking, and it would be the product, not the factorization, that
decides the tolerance.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from aotolith import compile_callable
from aotolith.legalize.linalg import _LINALG_RESULTS
from aotolith.legalize.targets import NATIVE_TARGETS


def _plan(call, *values: torch.Tensor):
    """Compile ``call`` and run it over the same inputs Torch was given."""

    compiled = compile_callable(call, values)
    actual = compiled.plan(*(mx.array(value.numpy()) for value in values))
    mx.eval(actual)
    return compiled, actual


def _run(call, *values: torch.Tensor) -> tuple[tuple[np.ndarray, ...], ...]:
    """Compile, run, and hand back both answers as float64 arrays."""

    _, actual = _plan(call, *values)
    expected = call(*values)
    if not isinstance(expected, (tuple, list)):
        expected = (expected,)
    return (
        tuple(np.asarray(value, dtype=np.float64) for value in actual),
        tuple(
            np.asarray(value.detach().numpy(), dtype=np.float64) for value in expected
        ),
    )


def _assert_close(actual, expected, atol: float = 1e-5) -> None:
    reference = np.asarray(expected, dtype=np.float64)
    assert actual.shape == reference.shape
    scale = float(np.abs(reference).max()) or 1.0
    assert float(np.abs(actual - reference).max()) <= atol * scale


@pytest.fixture(scope="module")
def system() -> torch.Tensor:
    """A well-conditioned symmetric system to factorize."""

    torch.manual_seed(0)
    raw = torch.randn(4, 4)
    return (raw @ raw.T + torch.eye(4) * 3).float()


@pytest.mark.parametrize(
    "rhs",
    [
        pytest.param(torch.randn(4), id="vector"),
        pytest.param(torch.randn(4, 1), id="one-column"),
        pytest.param(torch.randn(4, 3), id="three-columns"),
        pytest.param(torch.randn(2, 4, 3), id="batched"),
    ],
)
def test_solve_matches_pytorch(system, rhs) -> None:
    actual, expected = _run(torch.linalg.solve, system, rhs)
    _assert_close(actual[0], expected[0])


@pytest.mark.parametrize(
    "rhs",
    [
        # A single equation axis belongs to the matrix, not to the batch, so a
        # single vector is a valid right-hand side for a stack of matrices.
        pytest.param(torch.randn(4), id="single-vector"),
        pytest.param(torch.randn(4, 2), id="single-matrix"),
        pytest.param(torch.randn(2, 4), id="batched-vectors"),
    ],
)
def test_solve_broadcasts_the_batch_axes_of_one_operand(system, rhs) -> None:
    # Torch broadcasts the batch prefixes, which is why a single right-hand side
    # is a valid operand for a stack of matrices. Broadcasting the operands as
    # MLX would have lined the vector up against the matrices' own columns.
    matrices = torch.stack([system, system * 2])
    actual, expected = _run(torch.linalg.solve, matrices, rhs)
    _assert_close(actual[0], expected[0])
    assert actual[0].shape == expected[0].shape


@pytest.mark.parametrize(
    ("upper", "unitriangular"),
    [
        pytest.param(True, False, id="upper"),
        pytest.param(False, False, id="lower"),
        pytest.param(True, True, id="upper-unit"),
    ],
)
def test_solve_triangular_matches_pytorch(system, upper, unitriangular) -> None:
    triangle = (torch.triu(system) if upper else torch.tril(system)).contiguous()
    rhs = torch.randn(4, 2)
    call = lambda a, b: torch.linalg.solve_triangular(  # noqa: E731
        a, b, upper=upper, unitriangular=unitriangular
    )
    actual, expected = _run(call, triangle, rhs)
    _assert_close(actual[0], expected[0])


def test_inverse_and_determinant_match_pytorch(system) -> None:
    actual, expected = _run(torch.linalg.inv, system)
    _assert_close(actual[0], expected[0])

    actual, expected = _run(torch.linalg.det, system)
    _assert_close(actual[0], expected[0])

    actual, expected = _run(torch.linalg.slogdet, system)
    _assert_close(actual[0], expected[0])
    _assert_close(actual[1], expected[1])


@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_matches_pytorch(system, upper) -> None:
    call = lambda a: torch.linalg.cholesky(a, upper=upper)  # noqa: E731
    actual, expected = _run(call, system)
    _assert_close(actual[0], expected[0])


def test_eigh_matches_as_an_eigen_decomposition(system) -> None:
    # The ascending order is fixed, but the sign of each eigenvector is not, so
    # the check is the values and the reconstruction the vectors promise.
    actual, _ = _run(torch.linalg.eigh, system)
    values, vectors = actual
    reference_values = torch.linalg.eigh(system)[0].numpy()
    _assert_close(values, reference_values, atol=1e-6)
    _assert_close(vectors @ np.diag(values) @ vectors.T, system.numpy(), atol=1e-4)


@pytest.mark.parametrize("full_matrices", [False, True])
def test_svd_matches_as_a_factorization(full_matrices) -> None:
    torch.manual_seed(3)
    value = torch.randn(6, 4)
    call = lambda a: torch.linalg.svd(a, full_matrices=full_matrices)  # noqa: E731
    actual, expected = _run(call, value)
    u, s, vh = actual
    assert u.shape == expected[0].shape
    assert vh.shape == expected[2].shape
    _assert_close(s, expected[1], atol=1e-6)
    rank = s.shape[0]
    _assert_close(u[:, :rank] @ np.diag(s) @ vh[:rank, :], value.numpy(), atol=1e-4)


def test_qr_matches_as_a_factorization() -> None:
    torch.manual_seed(4)
    value = torch.randn(6, 4)
    actual, _ = _run(torch.linalg.qr, value)
    q, r = actual
    _assert_close(q @ r, value.numpy(), atol=1e-4)
    _assert_close(q.T @ q, np.eye(q.shape[1]), atol=1e-4)


def test_pinv_matches_pytorch() -> None:
    # The pseudo-inverse itself is defined, so it compares element by element;
    # the tolerance is the one a truncated factorization implies.
    torch.manual_seed(5)
    value = torch.randn(5, 3)
    actual, expected = _run(torch.linalg.pinv, value)
    _assert_close(actual[0], expected[0], atol=1e-4)


def test_lu_factor_and_pivots_match_pytorch(system) -> None:
    # ATen reports LAPACK's pivots, which are one-based where MLX's are
    # zero-based.
    actual, expected = _run(torch.linalg.lu_factor, system)
    _assert_close(actual[0], expected[0])
    assert actual[1].tolist() == expected[1].tolist()


def test_matrix_exp_matches_pytorch() -> None:
    torch.manual_seed(6)
    value = torch.randn(3, 3) / 4
    actual, expected = _run(torch.linalg.matrix_exp, value)
    _assert_close(actual[0], expected[0], atol=1e-5)


def test_only_the_selected_result_of_a_factorization_is_lowered() -> None:
    # A factorization whose other members are never read costs nothing: each
    # member is its own operator, and legalization emits only the ones the
    # graph selects.
    torch.manual_seed(8)
    value = torch.randn(4, 4)
    value = (value @ value.T + torch.eye(4) * 3).float()

    compiled, _ = _plan(lambda a: torch.linalg.eigh(a)[0], value)
    assert [op.op for op in compiled.lowered_graph.ops] == ["eigh_values"]

    compiled, _ = _plan(lambda a: torch.linalg.svd(a)[1], value)
    assert [op.op for op in compiled.lowered_graph.ops] == ["svd_values"]

    compiled, _ = _plan(lambda a: torch.linalg.slogdet(a)[1], value)
    assert [op.op for op in compiled.lowered_graph.ops] == ["slogdet_logabsdet"]


def test_a_half_precision_factorization_answers_in_half_precision(system) -> None:
    # The factorizations are float32 only, so a half operand is promoted for the
    # call and the result cast back to the dtype the plan declared.
    half = system.half()
    _, actual = _plan(torch.linalg.cholesky, half)
    assert actual[0].dtype == mx.float16
    _assert_close(
        np.asarray(actual[0], dtype=np.float64),
        torch.linalg.cholesky(system).numpy(),
        atol=1e-3,
    )


def test_every_factorization_spelling_is_kept_from_decomposition() -> None:
    # Torch ships a decomposition for ``aten.linalg_solve.default``: if the
    # targets were not kept, which operator the frontend sees would depend on
    # the direction that rule happens to go.
    assert set(_LINALG_RESULTS) <= NATIVE_TARGETS


@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_matches_pytorch(system, upper) -> None:
    # ``cholesky_solve`` answers ``A⁻¹B`` from the factor of ``A``; the lowering
    # runs the two triangular solves that definition spells out, so the plan
    # reads the same operator ``solve_triangular`` already is.
    factor = torch.linalg.cholesky(system)
    if upper:
        factor = factor.T.contiguous()
    rhs = torch.randn(4, 3)
    call = lambda b, l: torch.cholesky_solve(b, l, upper=upper)  # noqa: E731
    compiled, actual = _plan(call, rhs, factor)
    _assert_close(np.asarray(actual[0], dtype=np.float64), call(rhs, factor).numpy())
    assert [op.op for op in compiled.lowered_graph.ops] == [
        "transpose",
        "solve_triangular",
        "solve_triangular",
    ]


def test_cholesky_solve_broadcasts_a_batched_factor(system) -> None:
    factor = torch.linalg.cholesky(system)
    factors = torch.stack([factor, factor * 2.0])
    rhs = torch.randn(2, 4, 3)
    _, actual = _plan(torch.cholesky_solve, rhs, factors)
    _assert_close(
        np.asarray(actual[0], dtype=np.float64),
        torch.cholesky_solve(rhs, factors).numpy(),
    )


@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_inverse_matches_pytorch(system, upper) -> None:
    # ``potri``'s answer, read off the factor: the same two solves with an
    # identity right-hand side.
    factor = torch.linalg.cholesky(system)
    if upper:
        factor = factor.T.contiguous()
    call = lambda l: torch.cholesky_inverse(l, upper=upper)  # noqa: E731
    _, actual = _plan(call, factor)
    _assert_close(np.asarray(actual[0], dtype=np.float64), call(factor).numpy())


def test_cholesky_inverse_builds_the_identity_over_the_batch_axes(system) -> None:
    factor = torch.linalg.cholesky(system)
    factors = torch.stack([factor, factor * 3.0])
    _, actual = _plan(torch.cholesky_inverse, factors)
    _assert_close(
        np.asarray(actual[0], dtype=np.float64),
        torch.cholesky_inverse(factors).numpy(),
    )


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(0.0, id="zero-norm"),
        pytest.param(1.0, id="manhattan"),
        pytest.param(2.0, id="euclidean"),
        pytest.param(3.0, id="general-p"),
        pytest.param(float("inf"), id="maximum"),
    ],
)
def test_cdist_matches_pytorch(order) -> None:
    # ``cdist`` is the p-norm of every pairwise difference, so each order is the
    # same lowering with a different exponent - including the 0-"norm", which
    # counts the entries that differ.
    torch.manual_seed(23)
    left = torch.randn(3, 5)
    right = torch.randn(4, 5)
    call = lambda a, b: torch.cdist(a, b, order)  # noqa: E731
    compiled, actual = _plan(call, left, right)
    expected = call(left, right)
    assert actual[0].shape == expected.shape
    _assert_close(np.asarray(actual[0], dtype=np.float64), expected.numpy())
    assert "vector_norm" in [op.op for op in compiled.lowered_graph.ops]


def test_cdist_broadcasts_the_point_set_batches() -> None:
    torch.manual_seed(25)
    left = torch.randn(2, 3, 5)
    right = torch.randn(4, 5)  # one set of points against a batch of the other
    call = lambda a, b: torch.cdist(a, b)  # noqa: E731
    _, actual = _plan(call, left, right)
    expected = call(left, right)
    assert actual[0].shape == expected.shape
    _assert_close(np.asarray(actual[0], dtype=np.float64), expected.numpy())
