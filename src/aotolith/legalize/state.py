"""Lowering state: the value table, the operation list, and the id allocation.

One pass over a traced graph builds a program. That pass needs a few things at
once - the values defined so far, the operations emitted, the mapping from FX
node to value id, and how many ids have been allocated - and every builder needs
to write into it. This object is that shared state.

Two rules keep it usable:

* **Ids count values, not operations.** One operation may produce several
  values, so ``next_index`` counts allocated values rather than operations, and
  a single-output graph still numbers from ``v0``. Every builder takes the same
  index as its first free slot, so a composite can emit as many values as it
  needs without colliding with the enclosing loop.
* **Inputs are not numbered.** Graph inputs take the ids ``arg0``, ``arg1``...
  and are excluded from the count, which is why ``next_index`` subtracts them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..ir import Operation, TensorType, Value
from .fx import _empty_value_nodes


class Lowering:
    """The program being built from one traced FX graph."""

    def __init__(self, traced: Any, promoted_constants: list[Any] | None) -> None:

        self.traced = traced
        self.promoted_constants = promoted_constants
        # Zero-length constants that a KV-cache model opens its cache with.
        # They carry no arithmetic and are folded at the concatenation that
        # consumes them; any other use fails by name in ``_argument_ids``.
        self.empty_nodes = _empty_value_nodes(traced)
        self.values: dict[str, Value] = {}
        self.input_ids: list[str] = []
        self.node_ids: dict[object, str] = {}
        self.ops: list[Operation] = []
        self.output_ids: tuple[str, ...] = ()
        # Results of a multi-output node that nothing reads.
        self.dead_nodes: set[object] = set()

    def next_index(self) -> int:
        """Index of the next value id to hand out."""

        return len(self.values) - len(self.input_ids)

    def allocate(self, type: TensorType, count: int = 1) -> tuple[str, ...]:
        """Reserve ``count`` output ids and their types, in one step."""

        start = self.next_index()
        ids = tuple(f"v{start + position}" for position in range(count))
        for value_id in ids:
            self.values[value_id] = Value(value_id, type)
        return ids

    def emit(
        self,
        op_name: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_type: TensorType,
    ) -> str:
        """Emit one single-result operation and return the id it wrote."""

        output_id = self.allocate(output_type)[0]
        self.ops.append(Operation(op_name, inputs, (output_id,), attrs))
        return output_id

    def emit_multi(
        self,
        op_name: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_types: tuple[TensorType, ...],
    ) -> tuple[str, ...]:
        """Emit one operation with several results, each with its own type."""

        start = self.next_index()
        ids = tuple(f"v{start + position}" for position in range(len(output_types)))
        for value_id, output_type in zip(ids, output_types):
            self.values[value_id] = Value(value_id, output_type)
        self.ops.append(Operation(op_name, inputs, ids, attrs))
        return ids

    def emit_all(self, ops_and_values: Any) -> None:
        """Absorb a multi-operation builder's ``(operations, values)`` result.

        The builders that lower one ATen node into several operations return
        exactly that shape, so the loop does not have to repeat the append.
        """

        for operation, value in ops_and_values:
            self.ops.append(operation)
            self.values[value.id] = value

    def bind(self, node: object, value_id: str) -> None:
        self.node_ids[node] = value_id

    def result(self, output_ids: tuple[str, ...]) -> Any:
        """The finished program, with everything unreachable removed."""

        from ..ir import Graph

        if not output_ids:
            raise ValueError("captured graph has no outputs")
        needed = set(output_ids)
        live_ops: list[Operation] = []
        for op in reversed(self.ops):
            if needed.intersection(op.outputs):
                live_ops.append(op)
                needed.update(op.inputs)
        ops = tuple(reversed(live_ops))
        graph = Graph(tuple(self.input_ids), self.values, ops, output_ids)
        graph.validate()
        return graph


# What a lowering handler looks like: it consumes one FX node and reports
# whether it claimed it. Declining is a normal outcome, not an error - it is how
# a conditional lowering defers to the next one.
Handler = Callable[[Lowering, Any, str], bool]
