"""The index-bounds contract, enforced where it is cheapest.

An index or take operation must fail the same way PyTorch fails: a clear error
naming the offending axis and the out-of-range value, not a garbage read. The
check is a comparison per element, so instead of testing inside the kernel it is
deferred: the caller records which arrays will be indexed, the run performs the
checks once in one vectorized comparison, and the kernel then reads unchecked.

The deferral is per-context so that a plan invoked from several threads does not
share a pending list, and ``_INDEX_CHECKS_ENABLED`` is the switch tests use to
prove the kernel path is correct with the checks turned off.
"""

from __future__ import annotations

import mlx.core as mx
from contextvars import ContextVar
from math import prod
from ..ir import TensorType
from ..codegen.expr import _DTYPE_ITEMSIZE

_PENDING_INDEX_CHECKS: ContextVar[list[tuple[mx.array, int, bool]] | None] = ContextVar(
    "aotolith_pending_index_checks", default=None
)


_INDEX_CHECKS_ENABLED: ContextVar[bool] = ContextVar(
    "aotolith_index_checks_enabled", default=True
)


def _defer_index_check(
    indices: mx.array, upper_bound: int, *, allow_negative: bool
) -> bool:
    """Queue a bounds check so a plan run pays for at most one device sync.

    Returns ``True`` when the check was queued and the caller must not sync.
    """

    if not _INDEX_CHECKS_ENABLED.get():
        return True
    pending = _PENDING_INDEX_CHECKS.get()
    if pending is None:
        return False
    pending.append((indices, upper_bound, allow_negative))
    return True


def _flush_index_checks(pending: list[tuple[mx.array, int, bool]]) -> None:
    """Collapse every queued bounds check into a single reduction and sync."""

    if not pending:
        return
    if not _INDEX_CHECKS_ENABLED.get():
        pending.clear()
        return
    violation: mx.array | None = None
    for indices, upper_bound, allow_negative in pending:
        low = -upper_bound if allow_negative else 0
        if indices.size == 0:
            continue
        bad = mx.logical_or(indices < low, indices >= upper_bound)
        # One plan can queue index tensors of different shapes, so reduce each
        # one before combining: the masks themselves do not broadcast.
        found = mx.any(bad)
        violation = found if violation is None else mx.logical_or(violation, found)
    if violation is None:
        return
    if bool(violation.item()):
        raise IndexError("index out of range in self")


def _check_take_indices(indices: mx.array, upper_bound: int) -> None:
    """Match PyTorch gather bounds instead of MLX's wrap-and-fill behavior."""

    if _defer_index_check(indices, upper_bound, allow_negative=False):
        return
    _flush_index_checks([(indices, upper_bound, False)])


def _mlx_take_checked(source: mx.array, indices: mx.array, axis: int) -> mx.array:
    _check_take_indices(indices, source.shape[axis])
    return mx.take(source, indices, axis=axis)


def _check_advanced_index_indices(indices: mx.array, upper_bound: int) -> None:
    if _defer_index_check(indices, upper_bound, allow_negative=True):
        return
    _flush_index_checks([(indices, upper_bound, True)])


def _tensor_nbytes(tensor_type: TensorType) -> int:
    itemsize = _DTYPE_ITEMSIZE[tensor_type.dtype]
    return prod(tensor_type.shape) * itemsize
