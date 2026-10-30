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

Which operators keep their own lowering is the *caller's* decision, passed in
as ``keep``. This module deliberately does not know the frontend's operator
table: the table is a fact about capture, and reaching back for it would make
capture and legalization import each other. The frontend already owns the list
(``frontend.NATIVE_TARGETS``), so it passes it in and the dependency points one
way.

For the operators that are kept, a decomposition would be a regression rather
than a shortcut: it replaces a dedicated Metal kernel (``attention``,
``layer_norm``, ``softmax``), an exact emulation (``arange`` rebuilds its float
sequence from the fma residual and a two-sum so it rounds once, the way
PyTorch's double expression does), or a lowering whose attribute normalisation
the rest of the compiler relies on (``sum``/``var`` spell their axes and
``keepdim`` differently in each schema). A test asserts the two sets stay
disjoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A decomposition function takes an FX node's arguments and emits operations
# while tracing, so its signature is only meaningful to ``make_fx``.
Decomposition = Callable[..., Any]

# Keyed by the keep set, since the table is a function of it. Building the
# table walks Torch's registries and ``asdict``-copies each rule, which is far
# more expensive than the compile that follows it.
_CACHE: dict[frozenset[str], dict[object, Decomposition]] = {}


def aten_decompositions(
    keep: frozenset[str] = frozenset(),
) -> dict[object, Decomposition]:
    """Torch's core decomposition table minus the targets in ``keep``.

    ``keep`` holds the ATen target names, as strings, whose own lowering is
    better than the decomposition: a fused kernel, an exact emulation, or a
    normalization the rest of the compiler depends on.
    """

    cached = _CACHE.get(keep)
    if cached is not None:
        return cached

    from torch._decomp import core_aten_decompositions

    table: dict[object, Decomposition] = {
        target: rule
        for target, rule in core_aten_decompositions().items()
        if str(target) not in keep
    }
    _CACHE[keep] = table
    return table
