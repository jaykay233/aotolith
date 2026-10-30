"""Run a callable under ``make_fx`` and return the traced graph.

Tracing is one job and lowering is another, and they fail for different reasons.
Keeping the trace here means the error message about *what can be traced* - a
tensor that never became a graph input - is written next to the ``make_fx`` call
that provokes it, instead of being buried in the graph-to-IR loop.

The decomposition table is applied during tracing rather than afterwards, so
what the legalizer sees is already the core operator set plus the ATen operators
the backend lowers natively. The caller passes that keep set in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..decompose import aten_decompositions


def trace(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    keep: frozenset[str],
) -> Any:
    """Trace ``fn`` into an ``fx.GraphModule``, decomposed and shape-only.

    ``keep`` is the set of ATen targets the backend lowers natively, which the
    decomposition table must leave alone. It is a parameter rather than an
    import so that tracing stays below the legalizer: the tracer needs to know
    *which targets survive*, and it takes that as given.
    """

    try:
        from torch.fx.experimental.proxy_tensor import make_fx
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the FX frontend") from exc

    # Fake tracing keeps capture shape-only: the default "real" mode executes
    # every operator on the example inputs, so tracing a large matmul costs a
    # full CPU matmul and compile time scales with the problem size.
    try:
        traced = make_fx(
            fn,
            tracing_mode="fake",
            decomposition_table=aten_decompositions(keep),
        )(*example_inputs)
    except AssertionError as exc:
        if "Please convert all Tensors to FakeTensors" not in str(exc):
            raise
        raise NotImplementedError(
            "a tensor reached an operator without being a graph input. Fake "
            "tracing only accepts the traced inputs and registered parameters "
            "and buffers, so a constant held on the module inside a container "
            "(a list, tuple or mapping attribute) or captured in a closure is "
            "invisible to it. Pass it through example_inputs or example_kwargs, "
            "or register it with register_buffer"
        ) from exc

    return traced
