"""Capture: turn a callable into the compiler's IR.

``capture_fx`` is the whole public surface, and it is deliberately two calls:
trace, then legalize. The split is not ceremony. Tracing decides *what PyTorch
operators exist* - which means choosing a tracing mode, applying the ATen
decomposition table, and rejecting graphs that cannot be traced at all. Lowering
decides *what those operators mean* - which means recognising fused patterns,
mapping ATen names onto core ops, and emitting one operation per core op.

Because they are separate, either half can be exercised alone: the traced graph
can be inspected to see why an operator was rejected, and the legalizer can be
driven from a hand-written FX graph in a test.

This module is therefore the composition point, not a layer of its own: it sits
above both halves and only wires them together. ``tracer`` stays below
``legalize``; nothing here is imported by either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..ir import Graph
from ..legalize import legalize
from ..legalize.targets import NATIVE_TARGETS
from .tracer import trace


def capture_fx(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    *,
    promoted_constants: list[Any] | None = None,
) -> Graph:
    """Capture a supported callable with make_fx and build the compiler IR.

    ``promoted_constants`` opts into binding a tensor that tracing captured as
    an FX ``get_attr`` node instead of rejecting it. Each such tensor becomes one
    extra graph input, appended after the traced placeholders in the order the
    graph reaches them, and its value is collected in the list the caller
    passed. ``compile_module`` uses this to fold module attributes and tensors
    built inside ``forward`` into the plan's private inputs.
    """

    return legalize(
        trace(fn, example_inputs, frozenset(NATIVE_TARGETS)), promoted_constants
    )


__all__ = ["capture_fx", "trace"]
