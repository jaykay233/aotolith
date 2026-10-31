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


def _guard_error() -> type[BaseException]:
    """The exception Torch raises when a value cannot be guarded on.

    Imported lazily so the tracer only reaches into Torch inside the functions
    that already need it, and falling back to a class nothing raises when the
    private module moved.
    """

    try:
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
    except ImportError:
        return type("_NoDataDependentGuard", (Exception,), {})
    return GuardOnDataDependentSymNode


def _data_dependent_message(error: BaseException) -> str:
    """Explain a trace that asked the data for a size or an integer.

    ``one_hot`` without ``num_classes`` takes the class count from the largest
    value, and a reshape to a size computed from the values does the same thing
    one layer down, so both surface here as a guard on an expression rather
    than as a named operator. The fix is the same in each case, so the message
    names the pattern and the argument that removes it.
    """

    return (
        "the trace asked for a size or an integer taken from the tensor's "
        f"values ({error}). A plan is compiled for fixed shapes, so an extent "
        "that is only known once the data is has nothing to compile to. The "
        "usual sources are one_hot without num_classes (pass it explicitly), a "
        "reshape or view to a size computed from the values, and an operator "
        "whose result length depends on the data. Give the extent from the "
        "input shape instead, or run this part outside the compiled region. A "
        "check that reduces over the values lands here too - (x > 0).all(), or "
        "a torch.distributions object built with the default validate_args - "
        "and belongs outside the traced function as well: construct the "
        "distribution with validate_args=False if the check is the only thing "
        "in the way"
    )


def _data_dependent_output_error() -> type[BaseException]:
    """The exception fake tracing raises for an output that is not a shape.

    ``torch.equal`` and ``torch.allclose`` reduce over the values, so fake
    tracing cannot make up an example result for them. Imported lazily for the
    same reason as the guard: Torch stays inside the functions that need it.
    """

    try:
        from torch._subclasses.fake_tensor import DataDependentOutputException
    except ImportError:
        return type("_NoDataDependentOutput", (Exception,), {})
    return DataDependentOutputException


def _data_output_message(error: BaseException) -> str:
    return (
        "the traced graph ends in an operator that reduces over the values - "
        f"torch.equal, torch.allclose or a similar comparison ({error}). Fake "
        "tracing keeps capture shape-only, so it has no example result to hand "
        "back for a comparison, and the comparison has no shape to compile to. "
        "Compile the model without it and compare the plan's outputs outside "
        "the compiled region"
    )


def _dynamic_output_shape_error() -> type[BaseException]:
    """The exception fake tracing raises for a result it cannot even size.

    ``nonzero`` and ``unique`` invent a symbolic length and are refused once
    the graph exists, but ``ctc_loss`` has no shape to invent: its fake kernel
    raises outright, before any node carries a name. Imported lazily for the
    same reason as the guard.
    """

    try:
        from torch._subclasses.fake_tensor import DynamicOutputShapeException
    except ImportError:
        return type("_NoDynamicOutputShape", (Exception,), {})
    return DynamicOutputShapeException


# Operators that raise the bare target name, and the name a user knows them by.
_DYNAMIC_OUTPUT_NAMES = {
    "aten._ctc_loss.Tensor": "ctc_loss",
    "aten._ctc_loss.default": "ctc_loss",
}


def _dynamic_output_shape_message(error: BaseException) -> str:
    """Explain an operator whose result length is a function of the values.

    The exception carries nothing but the ATen target, so the refusal has to
    supply the name and the way out - the same advice as the symbolic-length
    refusal in legalization, written here because this operator never reaches
    a graph.
    """

    target = str(error)
    named = _DYNAMIC_OUTPUT_NAMES.get(target)
    subject = f"{named} ({target})" if named else target
    return (
        f"{subject} returns a length computed from its values, and a plan can "
        "only hold static shapes: fake tracing cannot size the result, so "
        "there is no graph to lower. Pass the extent in from the input shape "
        "instead, or run this operator outside the compiled region"
    )


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
    except _guard_error() as exc:
        raise ValueError(_data_dependent_message(exc)) from exc
    except _data_dependent_output_error() as exc:
        raise ValueError(_data_output_message(exc)) from exc
    except _dynamic_output_shape_error() as exc:
        raise ValueError(_dynamic_output_shape_message(exc)) from exc
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
