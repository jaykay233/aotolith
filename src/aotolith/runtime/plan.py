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
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from ..errors import IRValidationError, PlanDTypeError, PlanInputError
from ..ir import Graph
from ..selection import KernelSelection
from ..ir import Operation
from ..planning import StoragePlan, tensor_nbytes as _tensor_nbytes
from ..codegen.dtypes import _MLX_DTYPES
from ..codegen.msl import KernelArtifact
from .bounds import (
    _INDEX_CHECKS_ENABLED,
    _PENDING_INDEX_CHECKS,
    _flush_index_checks,
    _mlx_take_checked,
)
from .mlx_ops.__init__ import _evaluate_mlx_op
from .mlx_ops.layout import _MLX_LAYOUT_OPS, _evaluate_layout_mlx

class _NoRunner:
    """Sentinel for "this graph cannot be unrolled; interpret it".

    A dedicated class rather than a bare ``object()`` so that the sentinel, the
    unrolled callable and ``None`` form a union a type checker can narrow: with
    ``object()`` the assignment wiped the callable's type out and the call below
    became untyped.
    """


_NO_RUNNER = _NoRunner()

#: How a runner trades peak memory for latency. Peak memory here is the memory
#: held by intermediate values that are dead but whose buffers have not been
#: released yet.
#:
#: ``"latency"`` enqueues the whole graph and synchronizes never, which is the
#: fastest thing to do and the least predictable: MLX allocates an output for
#: every node before the first one has finished, so the peak is the sum of every
#: intermediate, not the largest live set.
#:
#: ``"balanced"`` synchronizes at release points once a threshold of dead bytes
#: has accumulated. Because the threshold scales with the plan's transient
#: traffic, a call pays at most :data:`SYNC_BUDGET` synchronizations, and the
#: extra memory held above the planned peak is bounded by the same threshold
#: instead of by the graph's length.
#:
#: ``"minimal"`` synchronizes at every release point that frees anything, which
#: holds the peak to the storage plan's own figure and costs the most.
#:
#: A synchronization is a barrier: evaluate, wait for the device, then delete.
#: Evaluating alone only schedules the work, and MLX cannot hand a buffer back to
#: the allocator while the GPU may still be writing it, so a runner that only
#: evaluated let the next operator allocate on top of buffers that were dead on
#: paper - the same plan peaked at 134 MB, at 160 MB, and at 134 MB + 256 B
#: across twelve fresh processes until the wait was added.
MEMORY_POLICIES = ("latency", "balanced", "minimal")
DEFAULT_MEMORY = "balanced"
#: Upper bound on the synchronizations a ``"balanced"`` runner performs per
#: call. It is what makes the policy's latency cost a constant rather than a
#: function of the graph's length.
SYNC_BUDGET = 8
#: A floor for the ``"balanced"`` threshold. Synchronizing costs a pipeline
#: flush - measured at about 0.16 ms on a 28-layer prefill whose whole run is
#: 0.8 ms - so it is only worth doing when the bytes it releases are worth
#: more than that. On a machine with tens of gigabytes, holding eight megabytes
#: of dead intermediates is free; holding a gigabyte is not.
MIN_SYNC_BYTES = 8 << 20


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """What one runner policy does, and the peak it can reach.

    ``peak_bound_bytes`` is an upper bound, not a prediction: the runner holds
    at most ``planned_peak_bytes`` of live values plus whatever dead bytes have
    accumulated since the last synchronization. A synchronization is a device
    barrier, so the bytes released at one are reclaimed before the next operator
    allocates - the bound is a physical figure, and ``tests/test_memory.py``
    measures it in a fresh process rather than trusting the plan.
    """

    policy: str
    sync_count: int
    transient_bytes: int
    threshold_bytes: int | None
    planned_peak_bytes: int
    peak_bound_bytes: int


@dataclass(frozen=True, slots=True)
class PlanFootprint:
    """The memory a plan accounts for, split by who is responsible for it.

    ``input_bytes`` is what the caller keeps resident between calls - weights
    for a module plan, plus any tensor the caller passes every time.
    ``activation_bytes`` is the largest amount of memory the call itself creates
    and then gives back, which is the part a memory policy can bound.
    ``arena_bytes`` is what the plan would need if it wrote into one buffer
    instead of one allocation per value; it is the number to compare against a
    cache or a second request's footprint when budgeting.

    All three are derived from shapes, so they are the same on every run of the
    same build, which is what makes them cheap to gate.
    """

    input_bytes: int
    activation_bytes: int
    output_bytes: int
    peak_live_bytes: int
    arena_bytes: int

    def describe(self) -> str:
        return (
            f"inputs {_bytes(self.input_bytes)}, "
            f"activations {_bytes(self.activation_bytes)}, "
            f"outputs {_bytes(self.output_bytes)}, "
            f"peak {_bytes(self.peak_live_bytes)}"
        )


def _bytes(value: int) -> str:
    if value >= 1 << 30:
        return f"{value / (1 << 30):.1f} GiB"
    if value >= 1 << 20:
        return f"{value / (1 << 20):.1f} MiB"
    if value >= 1 << 10:
        return f"{value / (1 << 10):.1f} KiB"
    return f"{value} B"


def _activation_peak_bytes(lifetimes: tuple[Any, ...]) -> tuple[int, int]:
    """Peak of everything the call creates, and of the values it returns."""

    events: dict[int, int] = {}
    for lifetime in lifetimes:
        if lifetime.is_input:
            continue
        events[lifetime.start] = events.get(lifetime.start, 0) + lifetime.size_bytes
        events[lifetime.end] = events.get(lifetime.end, 0) - lifetime.size_bytes
    current = 0
    peak = 0
    for index in sorted(events):
        current += events[index]
        peak = max(peak, current)
    outputs = sum(
        lifetime.size_bytes for lifetime in lifetimes if lifetime.is_output
    )
    return peak, outputs



def _transient_bytes(plan: CompiledPlan) -> int:
    """Bytes of values this plan creates and may release.

    Inputs and outputs are excluded: an input is the caller's buffer and an
    output outlives the call, so neither is memory the plan can give back.
    """

    return sum(
        lifetime.size_bytes
        for lifetime in plan.storage_plan.lifetimes
        if not lifetime.is_input and not lifetime.is_output
    )


def _sync_threshold(plan: CompiledPlan, memory: str) -> int | None:
    """Dead bytes that trigger a synchronization, or ``None`` for never."""

    if memory == "latency":
        return None
    if memory == "minimal":
        return 0
    # Dividing the transient traffic by the sync budget is what turns "at most
    # eight synchronizations" into "at most an eighth of the traffic held in
    # limbo", and it scales with the plan rather than with the GPU: a decode
    # step's intermediates are kilobytes and a prefill's are megabytes, and both
    # stay inside the same fraction.
    return max(MIN_SYNC_BYTES, _transient_bytes(plan) // SYNC_BUDGET)



@dataclass(slots=True)
class CompiledPlan:
    """A plan, its kernels and its storage layout.

    A plan is safe to call from several threads. The lazy part is building the
    unrolled runner on first call, and that is guarded by ``_runner_lock`` so
    two threads cannot each unroll a graph and both compile their kernels; a
    call that arrives while another thread is building waits rather than
    duplicating the work. Everything after that point is per-call state -
    locally bound arrays, and the index checks, which are kept in a
    ``ContextVar`` precisely so one thread's pending checks cannot be flushed by
    another.

    What is *not* shared is the device. MLX compiles and enqueues on one stream
    per device, so concurrency here is concurrency in the host: several threads
    may build and submit work, and the GPU runs it in submission order. A
    caller that wants overlapping device work uses ``mx.async_eval`` (the Qwen2
    adapter does), not another thread.

    ``memory`` chooses how much peak memory the runner may trade for latency;
    see :data:`MEMORY_POLICIES`. A runner is built per policy and cached, so
    switching costs one unroll and not one per call.
    """

    graph: Graph
    kernels: tuple[KernelArtifact, ...]
    selections: tuple[KernelSelection, ...]
    _callables: dict[str, Any]
    storage_plan: StoragePlan
    _runners: dict[str, Any] = field(default_factory=dict, repr=False)
    _reports: dict[str, "MemoryReport"] = field(default_factory=dict, repr=False)
    _runner_lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def _runner(self) -> Any:
        """The default policy's runner, or ``None`` before the first call.

        Kept as a property rather than a field so that "has this plan been
        unrolled" stays a one-line question; the runner itself is per policy.
        """

        return self._runners.get(DEFAULT_MEMORY)

    def __call__(
        self,
        *inputs: Any,
        verify_index_bounds: bool = True,
        memory: str = DEFAULT_MEMORY,
    ) -> tuple[mx.array, ...]:
        runner, _ = self._runner_for(memory)
        if isinstance(runner, _NoRunner):
            outputs, _ = self._execute(
                inputs,
                collect_stats=False,
                verify_index_bounds=verify_index_bounds,
            )
            return outputs
        return runner(inputs, verify_index_bounds=verify_index_bounds)

    def _runner_for(self, memory: str) -> tuple[Any, "MemoryReport | None"]:
        """Build (and cache) the runner for one memory policy."""

        if memory not in MEMORY_POLICIES:
            raise PlanInputError(
                f"unknown memory policy {memory!r}",
                hint="choose one of: " + ", ".join(MEMORY_POLICIES),
            )
        runner = self._runners.get(memory)
        if runner is not None:
            return runner, self._reports.get(memory)
        with self._runner_lock:
            # Re-read inside the lock: another thread may have finished the
            # build between the check above and the acquisition.
            runner = self._runners.get(memory)
            if runner is None:
                built = _compile_plan_runner(self, memory)
                if built is None:
                    runner = _NO_RUNNER
                else:
                    runner, report = built
                    self._reports[memory] = report
                self._runners[memory] = runner
        return runner, self._reports.get(memory)

    def memory_report(self, memory: str = DEFAULT_MEMORY) -> "MemoryReport":
        """What the runner for ``memory`` is expected to do, before running it.

        The report is static: it is derived from the storage plan's release
        points, so it can be asked about a plan that has never been called. It
        exists because "how much memory will this use" is a question a host asks
        while deciding whether to admit a request, not after the OOM.
        """

        _, report = self._runner_for(memory)
        if report is None:  # pragma: no cover - every policy yields a report
            raise IRValidationError(
                f"the {memory!r} memory policy produced no runner report"
            )
        return report

    @property
    def footprint(self) -> "PlanFootprint":
        """Static memory accounting for this plan, split by owner."""

        lifetimes = self.storage_plan.lifetimes
        activation, outputs = _activation_peak_bytes(lifetimes)
        return PlanFootprint(
            input_bytes=sum(
                lifetime.size_bytes for lifetime in lifetimes if lifetime.is_input
            ),
            activation_bytes=activation,
            output_bytes=outputs,
            peak_live_bytes=self.storage_plan.peak_live_bytes,
            arena_bytes=self.storage_plan.arena_size_bytes,
        )

    def prepare(self, memory: str = DEFAULT_MEMORY) -> "MemoryReport":
        """Build the runner now instead of on the first call.

        The first call of a plan pays for unrolling the graph into a Python
        function; a server that wants its first request to look like its second
        calls this at startup. The kernels themselves are compiled by MLX on
        first *use*, which this cannot force without real inputs, so
        ``bench``'s ``first_call_ms`` keeps measuring that part.
        """

        return self.memory_report(memory)

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
                result: Any = (
                    runtime(*(env[value_id] for value_id in op.inputs))
                    if runtime is not None
                    else _evaluate_mlx_op(self.graph, op, env)
                )
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
        raise PlanInputError(
            f"expected {len(graph.inputs)} inputs, received {len(inputs)}",
            hint=(
                "pass the state and tensor keyword inputs the plan reported, in "
                "graph order; plan.graph.inputs names them"
            ),
        )
    values: list[mx.array] = []
    for value_id, value in zip(graph.inputs, inputs, strict=True):
        array = value if isinstance(value, mx.array) else mx.array(value)
        expected = graph.value(value_id).type
        if tuple(array.shape) != expected.shape:
            raise PlanInputError(
                f"input {value_id} has shape {tuple(array.shape)}, "
                f"expected {expected.shape}"
            )
        if array.dtype != _MLX_DTYPES[expected.dtype]:
            raise PlanDTypeError(
                f"input {value_id} has dtype {array.dtype}, "
                f"expected {expected.dtype.value}"
            )
        values.append(array)
    return values


def _int_attr(op: Operation, name: str) -> int:
    """Read an integer attribute, refusing anything else by name.

    Attributes come from the graph or from a decoded artifact, so the type is
    ``object``; a kernel emitted against a non-integer axis would be wrong in a
    way the Metal compiler cannot report, so the check lives here.
    """

    value = op.attrs.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise IRValidationError(f"{op.op}.{name} must be an integer, got {value!r}")
    return value


def _bind_plan_op(
    graph: Graph, op: Operation, runtime: Any
) -> Callable[..., mx.array | tuple[mx.array, ...]] | None:
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
        axis = _int_attr(op, "dim") % rank
        return lambda *values, _axis=axis: mx.concatenate(list(values), axis=_axis)
    if kind == "embedding":
        return lambda weight, indices: _mlx_take_checked(weight, indices, axis=0)
    operand_ids = op.inputs

    def generic(*values: mx.array) -> mx.array | tuple[mx.array, ...]:
        return _evaluate_mlx_op(graph, op, dict(zip(operand_ids, values)))

    return generic


def _compile_plan_runner(
    plan: CompiledPlan, memory: str = DEFAULT_MEMORY
) -> tuple[Callable[..., Any], "MemoryReport"] | None:
    """Turn the plan interpreter into straight-line Python, or return ``None``.

    Interpreting a thousand operations costs more than the kernels they launch:
    a decode step spent about a millisecond per token resolving value ids.
    Unrolling the graph once at first use replaces that bookkeeping with local
    variable assignments, and the kernels keep their original call signatures.

    ``memory`` decides where the unrolled body synchronizes. MLX is lazy: a
    body that only builds the graph leaves every intermediate node alive until
    the caller evaluates the outputs, and the evaluator then allocates an output
    for every node before releasing the ones it is done with - measured on a
    ten-operator chain, the peak was the sum of every intermediate rather than
    the largest live set. A ``"balanced"`` or ``"minimal"`` runner therefore
    evaluates the values that are still live at a release point and then deletes
    the names that died there, which is the only way to make the storage plan's
    releases real. Deleting the names alone does nothing measurable: the graph
    still holds the nodes.
    """

    graph = plan.graph
    ops = graph.ops
    threshold = _sync_threshold(plan, memory)
    input_set = set(graph.inputs)

    names: dict[str, str] = {
        value_id: f"i{position}" for position, value_id in enumerate(graph.inputs)
    }
    namespace: dict[str, Any] = {}
    body: list[str] = []
    if threshold is not None:
        # The generated body synchronizes, so it needs the module that does it.
        namespace["mx"] = mx
    #: Names produced since the last synchronization, still alive. Evaluating
    #: these forces everything they depend on, which is the whole point.
    unevaluated: list[str] = []
    #: Names whose last use has passed and whose buffers are still allocated.
    dead: list[str] = []
    dead_bytes = 0
    sync_count = 0
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
        if len(op.outputs) == 1:
            target = f"v{index}"
            names[op.outputs[0]] = target
            body.append(f"    {target} = {thunk_name}({', '.join(arguments)})")
            produced = (target,)
        else:
            # A multi-output operator is one call that returns a tuple, so the
            # targets are bound by unpacking rather than by a second evaluation.
            # Keeping it unrolled matters: a plan with one such operator used to
            # fall back to the interpreter for the whole graph.
            targets = [f"v{index}_{position}" for position in range(len(op.outputs))]
            for value_id, target in zip(op.outputs, targets, strict=True):
                names[value_id] = target
            joined = ", ".join(targets)
            body.append(f"    {joined} = {thunk_name}({', '.join(arguments)})")
            produced = tuple(targets)
        unevaluated.extend(produced)

        for value_id in plan.storage_plan.release_after_op[index]:
            if value_id in input_set:
                # The storage plan lists inputs at their last use, but an input
                # is the caller's buffer: dropping our name frees nothing, and
                # counting its bytes would make a plan with a large weight look
                # like it had a large transient and synchronize for no reason.
                continue
            name = names.get(value_id)
            if name is None:  # pragma: no cover - plan_storage never names one
                return None
            dead.append(name)
            dead_bytes += _tensor_nbytes(graph.value(value_id).type)
            if name in unevaluated:
                unevaluated.remove(name)

        if dead and threshold is not None and dead_bytes >= threshold:
            if unevaluated:
                body.append(f"    mx.eval({', '.join(unevaluated)})")
                unevaluated.clear()
            # ``mx.eval`` schedules and returns; the buffers it scheduled are
            # still in flight, and MLX cannot hand one back to the allocator
            # while the GPU may still be writing it. Waiting here is what makes
            # the release real - without it the measured peak randomly exceeded
            # the plan's own peak by a whole release group, because the next
            # operator allocated before the previous one's buffers came back.
            body.append("    mx.synchronize()")
            body.append(f"    del {', '.join(dead)}")
            dead.clear()
            dead_bytes = 0
            sync_count += 1

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

    planned_peak = plan.storage_plan.peak_live_bytes
    transient = _transient_bytes(plan)
    # With no threshold nothing is ever released, so every transient byte can be
    # alive at once on top of the planned peak. With one, the excess is what
    # accumulated since the last synchronization - and it can never exceed the
    # whole transient, which is why a threshold above the traffic does not make
    # the bound worse than the policy it was meant to improve on.
    allowance = transient if threshold is None else min(threshold, transient)
    report = MemoryReport(
        policy=memory,
        sync_count=sync_count,
        transient_bytes=transient,
        threshold_bytes=threshold,
        planned_peak_bytes=planned_peak,
        peak_bound_bytes=planned_peak + allowance,
    )
    return runner, report


def _as_arrays(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (list, tuple)):
        return tuple(outputs)
    return (outputs,)
