"""ATen decompositions that widen what the frontend can lower.

``capture_fx`` traces with ``make_fx``, which accepts a decomposition table:
a mapping from an ATen operator to a function that rewrites it into smaller
operators *during* tracing. Torch ships 941 such rules in
``core_aten_decompositions``, which is most of the long tail the frontend
would otherwise have to reject one operator at a time - ``lgamma``,
``digamma``, ``i0``, ``aminmax``, ``celu``, the FFT family, and the ``out=``
and gradient spellings of operators that already have a forward lowering.

Two reasons to use it rather than hand-writing every lowering:

* The rules are maintained upstream, so an operator that PyTorch itself
  redefines keeps working here without a change.
* A decomposition is expressed in operators the frontend already lowers, so
  the result goes through the same layout simplification, pointwise fusion
  and dtype promotion. There is no second code path to keep correct.

The frontend keeps the operators it lowers itself. For those a decomposition
would be a regression rather than a shortcut: it replaces a dedicated Metal
kernel (``attention``, ``layer_norm``, ``softmax``), an exact emulation
(``arange`` rebuilds its float sequence from the fma residual and a two-sum so
it rounds once, the way PyTorch's double expression does), or a lowering whose
attribute normalisation the rest of the compiler relies on (``sum``/``var``
spell their axes and ``keepdim`` differently in each schema). ``_KEEP_NATIVE``
names all of them and is derived from the frontend itself, so it cannot drift:
that module's ``NATIVE_TARGETS`` is the single list of targets it lowers, and a
test asserts the two sets are disjoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A decomposition function takes an FX node's arguments and emits operations
# while tracing, so its signature is only meaningful to ``make_fx``.
Decomposition = Callable[..., Any]

_CACHE: dict[object, Decomposition] | None = None


def native_targets() -> frozenset[str]:
    """Every ATen target the frontend lowers itself."""

    from .frontend import NATIVE_TARGETS

    return frozenset(NATIVE_TARGETS)


def aten_decompositions() -> dict[object, Decomposition]:
    """The decomposition table handed to ``make_fx``.

    Torch's full table minus the targets the frontend lowers natively, so an
    operator with a dedicated kernel keeps it. Cached because building the
    table walks Torch's registries and ``asdict``-copies each rule.
    """

    global _CACHE
    if _CACHE is not None:
        return _CACHE

    from torch._decomp import core_aten_decompositions

    keep_native = native_targets()
    table: dict[object, Decomposition] = {
        target: rule
        for target, rule in core_aten_decompositions().items()
        if str(target) not in keep_native
    }
    _CACHE = table
    return table


def is_kept_native(target: str) -> bool:
    """Whether ``target`` keeps its own lowering instead of decomposing."""

    return target in native_targets()
