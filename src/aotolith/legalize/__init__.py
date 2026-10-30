"""Lower a traced FX graph into the compiler's typed SSA IR.

One pass over ``traced.graph.nodes``. Placeholders become graph inputs, tensor
constants become promoted inputs, and every ``call_function`` is handed to the
operator family that claims it. What is left is a short loop: the dispatch
decisions themselves live in the family modules, keyed by ATen target.

Three precedence rules are worth stating outright, because they are the whole
design:

* **A family handler claims a target before anything generic sees it.** Fused
  patterns must win over the positional mapping, or ``rms_norm`` would never be
  recognised - it arrives as a decomposed multiply chain.
* **A handler may decline.** Returning False means "not the shape I expected",
  and the next handler for that target is tried. Constant padding uses this:
  it only claims a negative pad, which is a crop.
* **Recognition is attempted for every node.** An RMS norm has no target name at
  all, so its handler is tried for every node after the target-specific ones
  fail and before the generic path runs. It declines almost always.

Dead code falls out of the same pass: values nothing reads are dropped when the
program is assembled, so a graph that reads only a ``getitem`` of a six-part
decomposition still emits only what is reachable from its outputs.
"""

from __future__ import annotations

from typing import Any

from .fx import (
    _constant_tensor_message,
    _constant_value,
    _node_type,
    _reject_unsupported_attention_dropout,
    _tensor_type,
)
from ..ir import Graph, Value
from . import attention, elementwise, indexing, linalg, padding, reductions, views
from .targets import _EMPTY_COPY_TARGETS
from .state import Lowering

Handler = Any


def _empty_value_nodes(traced: Any) -> set[Any]:
    # Re-exported through capture, but the loop below wants it locally.
    from .fx import _empty_value_nodes as detect

    return detect(traced)


# Target -> handlers, in the order they are tried. A target may appear in more
# than one family when two lowerings apply to the same ATen call, which is why
# the value is a tuple rather than one function.
HANDLERS: dict[str, tuple[Handler, ...]] = {}
# Tried for every node, after the target-specific handlers decline. Recognition
# by shape has to come after recognition by name or a named lowerer for the same
# node would be shadowed.
PRE_HANDLERS: tuple[Handler, ...] = ()

for _family in (views, indexing, attention, reductions, padding, linalg, elementwise):
    for _target, _handlers in _family.HANDLERS.items():
        HANDLERS.setdefault(_target, ())
        HANDLERS[_target] += _handlers
    PRE_HANDLERS += getattr(_family, "PRE_HANDLERS", ())


def _lower_call(lowering: Lowering, node: Any, target: str) -> bool:
    """Try the handlers for one ``call_function`` node."""

    for handler in HANDLERS.get(target, ()):
        if handler(lowering, node, target):
            return True
    for handler in PRE_HANDLERS:
        if handler(lowering, node, target):
            return True
    return elementwise.lower_supported(lowering, node, target)


def legalize(
    traced: Any,
    promoted_constants: list[Any] | None = None,
) -> Graph:
    """Build the IR for one traced graph.

    ``promoted_constants`` opts into binding a tensor that tracing captured as
    an FX ``get_attr`` node instead of rejecting it. Each such tensor becomes
    one extra graph input, appended after the traced placeholders in the order
    the graph reaches them, and its value is collected in the list the caller
    passed.
    """

    _reject_unsupported_attention_dropout(traced.graph)
    lowering = Lowering(traced, promoted_constants)
    node_ids = lowering.node_ids
    values = lowering.values
    input_ids = lowering.input_ids

    for node in lowering.traced.graph.nodes:
        if node in lowering.dead_nodes:
            continue
        if node.op == "placeholder":
            value_id = f"arg{len(input_ids)}"
            node_ids[node] = value_id
            input_ids.append(value_id)
            values[value_id] = Value(value_id, _node_type(node))
        elif node.op == "get_attr":
            if node in lowering.empty_nodes:
                # Folding happens at the concatenation that consumes it; any
                # other use reaches ``_argument_ids`` and fails by name.
                continue
            if promoted_constants is None:
                raise NotImplementedError(
                    _constant_tensor_message(lowering.traced, node)
                )
            constant = _constant_value(lowering.traced, node)
            value_id = f"arg{len(input_ids)}"
            node_ids[node] = value_id
            input_ids.append(value_id)
            values[value_id] = Value(
                value_id, _tensor_type(constant, f"constant {node.target!r}")
            )
            promoted_constants.append(constant)
        elif node.op == "call_function":
            if node in node_ids:
                continue
            target = str(node.target)
            if node in lowering.empty_nodes and target in _EMPTY_COPY_TARGETS:
                # An identity copy of an empty constant carries no arithmetic;
                # it exists only to be concatenated away.
                continue
            if not _lower_call(lowering, node, target):
                raise ValueError(f"unsupported FX target: {target}")
        elif node.op == "output":
            from .refs import _output_ids

            lowering.output_ids = _output_ids(node.args[0], node_ids)
        else:
            raise ValueError(f"unsupported FX node kind: {node.op}")

    return lowering.result(lowering.output_ids)


__all__ = ["Handler", "legalize"]
