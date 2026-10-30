"""A compiled plan: what to run, where its memory lives, and how to call it.

``CompiledPlan`` is the output of compilation and the input of execution. It
holds the graph, the compiled kernels, the selections that chose them, and a
storage plan that assigns every value a slot in one arena; the arena is what
makes the plan a single allocation rather than one per value.

``_compile_plan_runner`` turns that into a Python callable - a straight-line
sequence of kernel launches with the input checks in front. Building it once,
rather than dispatching per operation per call, is what keeps the per-call cost
at the kernel launches themselves.
"""

from __future__ import annotations

import mlx.core as mx
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from ..ir import Graph
from ..selection import KernelSelection
from ..ir import Operation
from ..planning import StoragePlan
from ..codegen.dtypes import _MLX_DTYPES
from ..codegen.msl import KernelArtifact
from .bounds import (
    _INDEX_CHECKS_ENABLED,
    _PENDING_INDEX_CHECKS,
    _flush_index_checks,
    _mlx_take_checked,
    _tensor_nbytes,
)
from .mlx_ops.__init__ import _evaluate_mlx_op
from .mlx_ops.layout import _MLX_LAYOUT_OPS, _evaluate_layout_mlx

_NO_RUNNER = object()


@dataclass(slots=True)
class CompiledPlan:
    graph: Graph
    kernels: tuple[KernelArtifact, ...]
    selections: tuple[KernelSelection, ...]
    _callables: dict[str, Any]
    storage_plan: StoragePlan
    _runner: Any = None

    def __call__(
        self, *inputs: Any, verify_index_bounds: bool = True
    ) -> tuple[mx.array, ...]:
        runner = self._runner
        if runner is None:
            compiled = _compile_plan_runner(self)
            self._runner = _NO_RUNNER if compiled is None else compiled
            runner = compiled
        if runner is None or runner is _NO_RUNNER:
            outputs, _ = self._execute(
                inputs,
                collect_stats=False,
                verify_index_bounds=verify_index_bounds,
            )
            return outputs
        return runner(inputs, verify_index_bounds=verify_index_bounds)

    def call_with_stats(
        self, *inputs: Any
    ) -> tuple[tuple[mx.array, ...], "ExecutionStats"]:
        """Execute the plan and report how the planned arena was used."""

        return self._execute(inputs, collect_stats=True)

    def _execute(
        self,
        inputs: tuple[Any, ...],
        *,
        collect_stats: bool,
        verify_index_bounds: bool = True,
    ) -> tuple[tuple[mx.array, ...], "ExecutionStats"]:
        env = dict(
            zip(
                self.graph.inputs,
                _validate_plan_inputs(self.graph, inputs),
                strict=True,
            )
        )

        slot_by_value = (
            {
                value_id: slot
                for slot in self.storage_plan.slots
                for value_id in slot.value_ids
            }
            if collect_stats
            else {}
        )
        slot_by_id = {slot.slot_id: slot for slot in self.storage_plan.slots}
        arena = (
            mx.zeros((self.storage_plan.arena_size_bytes,), dtype=mx.uint8)
            if collect_stats and self.storage_plan.arena_size_bytes
            else None
        )
        slot_assignments: list[ArenaAssignment] = []
        assigned_values: dict[int, str] = {}
        pending_index_checks: list[tuple[mx.array, int, bool]] = []

        def store_result(value_id: str, result: mx.array) -> None:
            if not collect_stats:
                env[value_id] = result
                return
            slot = slot_by_value.get(value_id)
            if slot is None:
                env[value_id] = result
                return
            if arena is None:
                raise RuntimeError("storage plan assigned a value without an arena")

            value_type = self.graph.value(value_id).type
            nbytes = _tensor_nbytes(value_type)
            if nbytes > slot.size_bytes:
                raise RuntimeError(
                    f"value {value_id} needs {nbytes} bytes but slot "
                    f"{slot.slot_id} has {slot.size_bytes}"
                )
            raw = mx.view(result.reshape((-1,)), mx.uint8)
            start = slot.offset_bytes
            arena[start : start + nbytes] = raw
            storage = mx.view(
                arena[start : start + nbytes], _MLX_DTYPES[value_type.dtype]
            ).reshape(value_type.shape)
            env[value_id] = storage
            if collect_stats:
                previous = assigned_values.get(slot.slot_id)
                slot_assignments.append(
                    ArenaAssignment(
                        value_id=value_id,
                        slot_id=slot.slot_id,
                        offset_bytes=slot.offset_bytes,
                        size_bytes=nbytes,
                        reused=previous is not None and previous != value_id,
                    )
                )
            assigned_values[slot.slot_id] = value_id

        check_token = _PENDING_INDEX_CHECKS.set(pending_index_checks)
        enabled_token = _INDEX_CHECKS_ENABLED.set(verify_index_bounds)
        try:
            for op_index, op in enumerate(self.graph.ops):
                runtime = self._callables.get(op.outputs[0])
                if runtime is not None:
                    result = runtime(*(env[value_id] for value_id in op.inputs))
                else:
                    result = _evaluate_mlx_op(self.graph, op, env)
                if len(op.outputs) == 1:
                    store_result(op.outputs[0], result)
                else:
                    if not isinstance(result, tuple):
                        raise RuntimeError(
                            f"{op.op} produces {len(op.outputs)} values but its "
                            f"runtime returned "
                            f"{type(result).__name__}"
                        )
                    if len(result) != len(op.outputs):
                        raise RuntimeError(
                            f"{op.op} produces {len(op.outputs)} values but its "
                            f"runtime returned {len(result)}"
                        )
                    for value_id, array in zip(op.outputs, result, strict=True):
                        store_result(value_id, array)
                for value_id in self.storage_plan.release_after_op[op_index]:
                    env.pop(value_id, None)
        finally:
            _INDEX_CHECKS_ENABLED.reset(enabled_token)
            _PENDING_INDEX_CHECKS.reset(check_token)

        _flush_index_checks(pending_index_checks)

        stats = ExecutionStats(
            arena_size_bytes=self.storage_plan.arena_size_bytes,
            arena_slot_count=len(slot_by_id),
            assignments=tuple(slot_assignments),
            reuse_count=sum(item.reused for item in slot_assignments),
        )
        return tuple(env[value_id] for value_id in self.graph.outputs), stats


@dataclass(frozen=True, slots=True)
class ArenaAssignment:
    value_id: str
    slot_id: int
    offset_bytes: int
    size_bytes: int
    reused: bool


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    arena_size_bytes: int
    arena_slot_count: int
    assignments: tuple[ArenaAssignment, ...]
    reuse_count: int


def _validate_plan_inputs(graph: Graph, inputs: tuple[Any, ...]) -> list[mx.array]:
    """Validate the caller's arrays against the graph signature."""

    if len(inputs) != len(graph.inputs):
        raise ValueError(f"expected {len(graph.inputs)} inputs, received {len(inputs)}")
    values: list[mx.array] = []
    for value_id, value in zip(graph.inputs, inputs, strict=True):
        array = value if isinstance(value, mx.array) else mx.array(value)
        expected = graph.value(value_id).type
        if tuple(array.shape) != expected.shape:
            raise ValueError(
                f"input {value_id} has shape {tuple(array.shape)}, "
                f"expected {expected.shape}"
            )
        if array.dtype != _MLX_DTYPES[expected.dtype]:
            raise TypeError(
                f"input {value_id} has dtype {array.dtype}, "
                f"expected {expected.dtype.value}"
            )
        values.append(array)
    return values


def _bind_plan_op(
    graph: Graph, op: Operation, runtime: Any
) -> Callable[..., mx.array] | None:
    """Return a positional evaluator for one operation of a compiled plan.

    The generic interpreter resolves operands through a value-id dictionary.
    A compiled runner already knows the operand order, so the frequently used
    fallbacks get a positional closure and only the rare kinds fall back to the
    dictionary adapter.
    """

    if runtime is not None:
        return runtime
    kind = op.op
    if kind in _MLX_LAYOUT_OPS:
        return lambda value, _op=op, _graph=graph: _evaluate_layout_mlx(
            _graph, _op, value
        )
    if kind == "cat":
        rank = len(graph.value(op.inputs[0]).type.shape)
        axis = int(op.attrs["dim"]) % rank
        return lambda *values, _axis=axis: mx.concatenate(values, axis=_axis)
    if kind == "embedding":
        return lambda weight, indices: _mlx_take_checked(weight, indices, axis=0)
    operand_ids = op.inputs

    def generic(*values: mx.array) -> mx.array:
        return _evaluate_mlx_op(graph, op, dict(zip(operand_ids, values)))

    return generic


def _compile_plan_runner(plan: CompiledPlan) -> Callable[..., Any] | None:
    """Turn the plan interpreter into straight-line Python, or return ``None``.

    Interpreting a thousand operations costs more than the kernels they launch:
    a decode step spent about a millisecond per token resolving value ids.
    Unrolling the graph once at first use replaces that bookkeeping with local
    variable assignments, and the kernels keep their original call signatures.
    """

    graph = plan.graph
    ops = graph.ops
    if any(len(op.outputs) != 1 for op in ops):
        return None

    names: dict[str, str] = {
        value_id: f"i{position}" for position, value_id in enumerate(graph.inputs)
    }
    namespace: dict[str, Any] = {}
    body: list[str] = []
    for index, op in enumerate(ops):
        arguments: list[str] = []
        for value_id in op.inputs:
            name = names.get(value_id)
            if name is None:
                return None
            arguments.append(name)
        thunk = _bind_plan_op(graph, op, plan._callables.get(op.outputs[0]))
        if thunk is None:
            return None
        thunk_name = f"t{index}"
        namespace[thunk_name] = thunk
        target = f"v{index}"
        names[op.outputs[0]] = target
        body.append(f"    {target} = {thunk_name}({', '.join(arguments)})")

    returned: list[str] = []
    for value_id in graph.outputs:
        name = names.get(value_id)
        if name is None:
            return None
        returned.append(name)
    body.append(f"    return ({', '.join(returned)},)")

    parameters = ", ".join(f"i{position}" for position in range(len(graph.inputs)))
    source = f"def run({parameters}):\n" + "\n".join(body) + "\n"
    exec(compile(source, f"<aotolith-plan-{len(ops)}>", "exec"), namespace)
    generated = namespace["run"]

    def runner(
        inputs: tuple[Any, ...], *, verify_index_bounds: bool = True
    ) -> tuple[mx.array, ...]:
        values = _validate_plan_inputs(graph, inputs)
        # Unlike the interpreter, the unrolled body keeps every intermediate
        # alive until it returns. That is the lifetime the async executor needs
        # anyway, so the storage plan's early releases only apply to the
        # statistics path, which still runs through ``_execute``.
        pending: list[tuple[mx.array, int, bool]] = []
        check_token = _PENDING_INDEX_CHECKS.set(pending)
        enabled_token = _INDEX_CHECKS_ENABLED.set(verify_index_bounds)
        try:
            outputs = generated(*values)
        finally:
            _INDEX_CHECKS_ENABLED.reset(enabled_token)
            _PENDING_INDEX_CHECKS.reset(check_token)
        _flush_index_checks(pending)
        return outputs

    return runner


def _as_arrays(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (list, tuple)):
        return tuple(outputs)
    return (outputs,)
