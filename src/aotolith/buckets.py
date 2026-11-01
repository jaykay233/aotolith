"""Serving variation in one shape axis from a fixed set of compiled plans.

The IR is static: every extent in the graph is a number, which is what makes the
storage plan, the peak bound and the kernel selection answers instead of
guesses. That is a boundary of the design rather than a gap in it - a truly
symbolic extent would push every one of those answers to run time and turn the
memory contract back into a hope - so this module bounds it from the other side.
A caller compiles a handful of widths once, and a request of any width up to the
largest bucket is served by the smallest bucket that holds it.

What dispatch costs is the difference between the requested width and the
bucket's: the padded rows are computed and thrown away. What it buys is that the
only per-request host work is one gather and one slice, instead of a compile that
costs hundreds of milliseconds.

Padding is *tiling*, not a constant fill. ``mx.take(x, arange(width) % n, axis)``
repeats the request's own rows, so the padded region holds real values: a
constant would turn ``softmax`` or a variance into NaN on the padded rows, and
repeating an existing row keeps the arithmetic finite. The padded rows are sliced
off the outputs, so their values never reach the caller - which means the
contract on the varying axis is that it is *independent*: widening it must not
change the values of the rows that were already there. That holds for a batch
axis and for any axis a function treats row-by-row. It does not hold for, say,
the sequence axis of a causal attention block, which is why :func:`compile_bucketed`
takes ``pad`` and ``unpad`` hooks: the model that needs a mask owns the mask, and
the compiler only owns the dispatch.

One consequence of trimming by shape rather than by provenance: an output whose
extent on the axis happens to equal the bucket width is indistinguishable from a
request-shaped one, and is trimmed with the rest. A module that returns a
fixed-width tensor of exactly a bucket's width needs ``unpad`` to say so.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any

import mlx.core as mx

from .compiler import CompiledModulePlan, compile_module
from .errors import (
    ArtifactCompatibilityError,
    BucketRangeError,
    CompilationError,
    MemoryBudgetError,
    PlanInputError,
)
from .runtime.plan import DEFAULT_MEMORY, MEMORY_POLICIES, MemoryReport

_log = logging.getLogger("aotolith.buckets")

#: Version of the ``.buckets.json`` index this module writes and reads.
BUCKETS_INDEX_VERSION = 1

#: How the padded region of an input is filled. Only ``"tile"`` is serializable,
#: because a hook is a Python callable and an artifact is not.
PAD_MODES = ("tile",)


def _width_of(value: Any, axis: int) -> int | None:
    """The extent of ``value`` on ``axis``, or ``None`` if it has no such axis."""

    shape = getattr(value, "shape", None)
    if shape is None or len(shape) <= axis:
        return None
    return int(shape[axis])


def _tile_reference(value: Any, axis: int, width: int) -> Any:
    """Repeat ``value``'s slices on ``axis`` until it is ``width`` wide.

    Used on the torch example inputs, where the values do not matter for tracing
    but do matter for autotuning, so they are the caller's own numbers rather
    than zeros.
    """

    size = value.shape[axis]
    if size == width:
        return value
    if size == 0:  # pragma: no cover - an IR input is never empty
        raise CompilationError(
            "cannot widen an empty axis: the example input has 0 slices on "
            f"axis {axis}",
            hint="pass an example input with at least one slice on the bucket axis",
        )
    import torch

    indices = torch.arange(width) % size
    return torch.index_select(value, axis, indices)


def _tile_request(value: mx.array, axis: int, width: int) -> mx.array:
    """Repeat ``value``'s slices on ``axis`` until it is ``width`` wide."""

    size = int(value.shape[axis])
    if size == width:
        return value
    if size == 0:  # pragma: no cover - dispatch refuses a zero width first
        raise PlanInputError(
            "cannot widen an empty axis: the request has 0 slices on "
            f"axis {axis}",
            hint="send at least one row on the bucket axis",
        )
    indices = mx.arange(width) % size
    return mx.take(value, indices, axis=axis)


def _trim_request(
    output: mx.array, axis: int, kept: int, padded_width: int
) -> mx.array:
    """Cut a bucket-width output back to the requested width.

    Only an axis that is exactly the bucket width is a candidate: an output whose
    extent on that axis came from somewhere else - a weight, a reduction - is
    left as it is.
    """

    if int(output.shape[axis]) != padded_width:
        return output
    keep = [slice(None)] * output.ndim
    keep[axis] = slice(0, kept)
    return output[tuple(keep)]


@dataclass(frozen=True, slots=True)
class ShapeBucket:
    """One compiled width, and the plan that serves it."""

    #: The extent of the varying axis this plan was compiled for.
    width: int
    plan: CompiledModulePlan

    def memory_report(self, memory: str = DEFAULT_MEMORY) -> MemoryReport:
        return self.plan.plan.memory_report(memory)

    @property
    def peak_bound_bytes(self) -> int:
        return self.memory_report().peak_bound_bytes


@dataclass(frozen=True, slots=True)
class BucketFootprint:
    """What a bucket set holds resident, and what one call of each can add.

    ``state_bytes`` counts each distinct state array once, by identity: the
    buckets are the same module compiled at several widths, so their weights are
    the same arrays after :func:`share_plan_state` rebinds them, and only the
    arrays that genuinely depend on the width are extra. ``planned_peak_bytes``
    is the largest planned live set any bucket holds, and ``peak_bound_bytes``
    is the largest bound across buckets - the number a memory budget has to
    clear for every request, not just for the average one.
    """

    bucket_count: int
    state_bytes: int
    planned_peak_bytes: int
    peak_bound_bytes: int

    def describe(self) -> str:
        return (
            f"{self.bucket_count} buckets, state {_bytes(self.state_bytes)}, "
            f"live set up to {_bytes(self.planned_peak_bytes)}, "
            f"peak bound {_bytes(self.peak_bound_bytes)}"
        )


def _bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def share_plan_state(
    plans: Sequence[CompiledModulePlan],
) -> list[CompiledModulePlan]:
    """Rebind identical state entries to one array across plans.

    Every bucket traces the same module at a different width, so a naive bucket
    list holds one copy of the weights per plan. The weights are identical, and
    the rotary tables (or any other state that depends on the width) are not:
    rebinding by ``(name, shape, dtype)`` leaves exactly one copy of each weight
    for the whole set and keeps the arrays that genuinely differ.
    """

    if len(plans) < 2:
        return list(plans)
    canonical: dict[str, mx.array] = {}
    for name, value in zip(plans[0].state_names, plans[0]._state_values):
        canonical.setdefault(name, value)

    shared: list[CompiledModulePlan] = [plans[0]]
    for plan in plans[1:]:
        values = list(plan._state_values)
        changed = False
        for index, name in enumerate(plan.state_names):
            reference = canonical.get(name)
            if reference is None:
                continue
            value = values[index]
            if value.shape == reference.shape and value.dtype == reference.dtype:
                canonical.setdefault(name, value)
                if values[index] is not reference:
                    values[index] = reference
                    changed = True
        shared.append(replace(plan, _state_values=tuple(values)) if changed else plan)
    return shared


@dataclass(slots=True)
class BucketStats:
    """How often each bucket served a request, and how much work that wasted.

    A serving loop needs both numbers. The counts say which widths the traffic
    actually clusters on, which is how the bucket list gets tuned; the slack says
    how much of the compute was padding, which is the cost of that tuning.
    """

    counts: dict[int, int] = field(default_factory=dict)
    slack_tokens: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, bucket_width: int, requested: int) -> None:
        with self._lock:
            self.counts[bucket_width] = self.counts.get(bucket_width, 0) + 1
            self.slack_tokens += max(0, bucket_width - requested)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counts": dict(sorted(self.counts.items())),
                "slack_tokens": self.slack_tokens,
            }


class BucketedPlans:
    """A plan per width, dispatched by the request's width on one axis.

    The class is immutable apart from the statistics counter, so it is safe to
    share between threads: each call reads the bucket list, picks one by shape,
    and hands the arrays to a plan that already documents its own thread
    contract.
    """

    def __init__(
        self,
        buckets: Sequence[ShapeBucket],
        *,
        axis: int,
        widening: Sequence[int],
        input_count: int,
        widening_kwargs: Sequence[str] = (),
        pad: Callable[[tuple[Any, ...], int], tuple[Any, ...]] | None = None,
        unpad: Callable[[tuple[Any, ...], int], tuple[Any, ...]] | None = None,
    ) -> None:
        ordered = tuple(sorted(buckets, key=lambda bucket: bucket.width))
        if not ordered:
            raise ValueError("a bucket set needs at least one bucket")
        widths = [bucket.width for bucket in ordered]
        if widths != sorted(set(widths)):
            raise ValueError(f"bucket widths must be strictly increasing, got {widths}")
        self.buckets = ordered
        self.axis = axis
        #: Positions in the argument tuple whose extent on ``axis`` decides the
        #: bucket. Everything else is passed through untouched.
        self.widening = tuple(widening)
        #: Tensor keyword inputs that are shaped like the request, so they are
        #: tiled with it. A keyword whose extent did not follow the example's
        #: width at compile time is a fixed table and is passed through.
        self.widening_kwargs = tuple(widening_kwargs)
        self.input_count = input_count
        self._pad = pad
        self._unpad = unpad
        self.stats = BucketStats()

    # -- inspection ------------------------------------------------------

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(bucket.width for bucket in self.buckets)

    @property
    def max_width(self) -> int:
        return self.buckets[-1].width

    @property
    def min_width(self) -> int:
        return self.buckets[0].width

    @property
    def pad_mode(self) -> str:
        """``"tile"`` for the built-in padding, ``"custom"`` for hooks."""

        return "tile" if self._pad is None else "custom"

    def bucket_for(self, width: int) -> ShapeBucket:
        """The smallest bucket that holds ``width``."""

        if width <= 0:
            raise PlanInputError(
                f"cannot serve a request of width {width}",
                hint="the bucket axis must have at least one slice",
            )
        for bucket in self.buckets:
            if width <= bucket.width:
                return bucket
        raise BucketRangeError(
            f"requested width {width} exceeds the largest bucket ({self.max_width})",
            hint=(
                "compile a wider bucket, or split the request; the plans are "
                "static, so nothing is recompiled at run time"
            ),
        )

    def requested_width(self, inputs: Sequence[Any]) -> int:
        """The width the caller asked for, read off the widening inputs."""

        observed: int | None = None
        for position in self.widening:
            if position >= len(inputs):
                raise PlanInputError(
                    f"expected at least {self.input_count} inputs, received "
                    f"{len(inputs)}",
                )
            width = _width_of(inputs[position], self.axis)
            if width is None:
                raise PlanInputError(
                    f"input {position} has no axis {self.axis}, and the bucket "
                    "dispatch reads the request's width from it"
                )
            if observed is None:
                observed = width
            elif width != observed:
                raise PlanInputError(
                    f"inputs disagree about the bucket axis: input "
                    f"{self.widening[0]} is {observed}, input {position} is "
                    f"{width}",
                    hint=(
                        "every widened input must have the same extent on axis "
                        f"{self.axis}"
                    ),
                )
        if observed is None:  # pragma: no cover - widening is never empty
            raise PlanInputError("no input carries the bucket axis")
        return observed

    def footprint(self) -> BucketFootprint:
        state: dict[int, int] = {}
        planned = 0
        bound = 0
        for bucket in self.buckets:
            for value in bucket.plan._state_values:
                state[id(value)] = value.nbytes
            report = bucket.plan.plan.memory_report()
            planned = max(planned, report.planned_peak_bytes)
            bound = max(bound, report.peak_bound_bytes)
        return BucketFootprint(
            bucket_count=len(self.buckets),
            state_bytes=sum(state.values()),
            planned_peak_bytes=planned,
            peak_bound_bytes=bound,
        )

    def memory_report(self, memory: str = DEFAULT_MEMORY) -> MemoryReport:
        """The widest bucket's report: the worst case across the set."""

        if memory not in MEMORY_POLICIES:
            raise PlanInputError(
                f"unknown memory policy {memory!r}",
                hint="choose one of: " + ", ".join(MEMORY_POLICIES),
            )
        return self.buckets[-1].plan.plan.memory_report(memory)

    def check_memory_budget(
        self, budget_bytes: int, *, memory: str = DEFAULT_MEMORY
    ) -> BucketFootprint:
        """Refuse the set before a request if any bucket cannot fit."""

        footprint = self.footprint()
        report = self.memory_report(memory)
        needed = footprint.state_bytes + report.peak_bound_bytes
        if needed > budget_bytes:
            raise MemoryBudgetError(
                f"the bucket set needs {needed} bytes "
                f"({footprint.state_bytes} of state + {report.peak_bound_bytes} "
                f"peak) and the budget is {budget_bytes}",
                hint=(
                    "drop the widest bucket, lower the memory policy, or raise "
                    "the budget; the bound is per request, not per set"
                ),
            )
        return footprint

    # -- execution -------------------------------------------------------

    def __call__(
        self,
        *inputs: Any,
        verify_index_bounds: bool = True,
        memory: str = DEFAULT_MEMORY,
        **tensor_kwargs: Any,
    ) -> tuple[mx.array, ...]:
        if len(inputs) != self.input_count:
            raise PlanInputError(
                f"expected {self.input_count} inputs, received {len(inputs)}",
                hint="pass the positional inputs the plan was compiled for",
            )
        requested = self.requested_width(inputs)
        bucket = self.bucket_for(requested)
        self.stats.record(bucket.width, requested)
        if bucket.width != requested:
            _log.debug(
                "bucket dispatch: request %d served by %d (%.0f%% padding)",
                requested,
                bucket.width,
                100.0 * (bucket.width - requested) / requested,
            )
        widened, widened_kwargs = self._widen(inputs, tensor_kwargs, requested, bucket.width)
        outputs = bucket.plan(
            *widened,
            verify_index_bounds=verify_index_bounds,
            memory=memory,
            **widened_kwargs,
        )
        return self._trim(outputs, bucket.width, requested)

    def _widen(
        self,
        inputs: tuple[Any, ...],
        tensor_kwargs: Mapping[str, Any],
        requested: int,
        width: int,
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        if width == requested:
            return inputs, tensor_kwargs
        kwargs = dict(tensor_kwargs)
        for name in self.widening_kwargs:
            value = kwargs.get(name)
            if value is not None and _width_of(value, self.axis) == requested:
                kwargs[name] = _tile_request(value, self.axis, width)
        if self._pad is not None:
            return self._pad(inputs, width), kwargs
        widened = list(inputs)
        for position in self.widening:
            widened[position] = _tile_request(widened[position], self.axis, width)
        return tuple(widened), kwargs

    def _trim(
        self, outputs: tuple[mx.array, ...], width: int, requested: int
    ) -> tuple[mx.array, ...]:
        if width == requested:
            return outputs
        if self._unpad is not None:
            return self._unpad(outputs, requested)
        return tuple(
            _trim_request(output, self.axis, requested, width)
            if _width_of(output, self.axis) == width
            else output
            for output in outputs
        )

    def prepare(self, memory: str = DEFAULT_MEMORY) -> tuple[MemoryReport, ...]:
        """Unroll every bucket's runner now, so no first request pays for it."""

        return tuple(bucket.plan.prepare(memory) for bucket in self.buckets)

    # -- artifacts -------------------------------------------------------

    def save(self, prefix: str | Path) -> tuple[Path, ...]:
        """Write one artifact per bucket plus an index that names them.

        Only the built-in padding can be saved: a hook is a Python callable, and
        an index that claims otherwise would dispatch differently after a
        reload. A custom-shaped set raises rather than writing an artifact that
        would change meaning.
        """

        if self._pad is not None or self._unpad is not None:
            raise CompilationError(
                "a bucket set with custom pad/unpad hooks cannot be saved",
                hint=(
                    "the hooks are host code, not part of the artifact; save the "
                    "underlying plans with bucket.plan.save(), or use the "
                    "built-in tiling padding"
                ),
            )
        path = Path(prefix)
        suffix = path.suffix or ".aoto"
        stem = path.name[: -len(suffix)] if path.suffix else path.name
        parent = path.parent
        written: list[Path] = []
        entries: list[dict[str, Any]] = []
        targets = {bucket.plan.target for bucket in self.buckets}
        if len(targets) != 1:
            # Each archive records its own target, so a mixed set would load as
            # whichever bucket happened to be reached first and fail much later
            # with a message about that archive rather than about the set.
            raise ArtifactCompatibilityError(
                f"a bucket set must come from one backend, found {sorted(targets)}",
                hint="recompile the set with a single target=",
            )
        backend = next(iter(targets))
        for bucket in self.buckets:
            artifact = parent / f"{stem}.{bucket.width}{suffix}"
            written.append(bucket.plan.save(artifact))
            entries.append(
                {
                    "width": bucket.width,
                    "artifact": artifact.name,
                    "peak_bound_bytes": bucket.peak_bound_bytes,
                    "planned_peak_bytes": bucket.memory_report().planned_peak_bytes,
                }
            )
        index_path = parent / f"{stem}.buckets.json"
        index = {
            "buckets_index_version": BUCKETS_INDEX_VERSION,
            "target": backend,
            "axis": self.axis,
            "pad": self.pad_mode,
            "input_count": self.input_count,
            "widening": list(self.widening),
            "widening_kwargs": list(self.widening_kwargs),
            "widths": list(self.widths),
            "buckets": entries,
        }
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        _log.info(
            "wrote %d buckets to %s",
            len(entries),
            index_path,
        )
        return tuple(written) + (index_path,)

    def describe(self) -> dict[str, Any]:
        """A JSON-ready summary, the shape ``aoto inspect`` prints."""

        footprint = self.footprint()
        return {
            "buckets": len(self.buckets),
            "axis": self.axis,
            "pad": self.pad_mode,
            "widths": list(self.widths),
            "widening_inputs": list(self.widening),
            "widening_kwargs": list(self.widening_kwargs),
            "state_bytes": footprint.state_bytes,
            "planned_peak_bytes": footprint.planned_peak_bytes,
            "peak_bound_bytes": footprint.peak_bound_bytes,
            "per_bucket": [
                {
                    "width": bucket.width,
                    "peak_bound_bytes": bucket.peak_bound_bytes,
                    "planned_peak_bytes": bucket.memory_report().planned_peak_bytes,
                    "syncs": bucket.memory_report().sync_count,
                    "kernels": len(bucket.plan.plan.kernels),
                }
                for bucket in self.buckets
            ],
        }


def compile_bucketed(
    module: Any,
    example_inputs: Sequence[Any],
    *,
    axis: int = 0,
    sizes: Sequence[int],
    pad: Callable[[tuple[Any, ...], int], tuple[Any, ...]] | None = None,
    unpad: Callable[[tuple[Any, ...], int], tuple[Any, ...]] | None = None,
    **compile_options: Any,
) -> BucketedPlans:
    """Compile ``module`` once per width in ``sizes`` along ``axis``.

    The example inputs are the shape the module is traced at; every input whose
    extent on ``axis`` equals the example's is re-widened per bucket, and the
    rest are left alone, so a weight of a different rank is never touched. The
    width in ``sizes`` may be the example's own; it is compiled like any other,
    except that no padding is ever applied to it.

    ``compile_options`` are :func:`aotolith.compile_module`'s, so
    ``codegen_matmul``, ``autotune`` and the rest keep working; the state is
    shared across buckets afterwards with :func:`share_plan_state`.
    """

    widths = [int(width) for width in sizes]
    if not widths:
        raise ValueError("sizes must name at least one bucket width")
    if any(width <= 0 for width in widths):
        raise ValueError(f"bucket widths must be positive, got {widths}")
    if widths != sorted(set(widths)):
        raise ValueError(f"bucket widths must be strictly increasing, got {widths}")
    if axis < 0:
        raise ValueError(f"bucket axis must be non-negative, got {axis}")

    reference = [_width_of(value, axis) for value in example_inputs]
    widening = [
        position for position, width in enumerate(reference) if width is not None
    ]
    if not widening:
        raise CompilationError(
            f"no example input has axis {axis}, so there is nothing to bucket",
            hint=(
                "pass the axis the request's shape varies on, which is 0 for a "
                "batch axis"
            ),
        )
    # Only the inputs that share the extent decide the width; an input of a
    # different extent on the same axis (a fixed table, say) is not a request
    # dimension and must not be widened.
    observed = {reference[position] for position in widening}
    if len(observed) != 1:
        first = reference[widening[0]]
        widening = [
            position for position in widening if reference[position] == first
        ]
        if not widening:  # pragma: no cover - first is in the set by construction
            raise CompilationError("no consistent bucket axis")

    # A keyword whose extent followed the example's width is another view of
    # the request - a position table, a mask - and has to be re-widened with it,
    # or the plan traced at width 4 would demand a keyword still 1 wide. A
    # keyword whose extent did not follow is a fixed table: widening it would
    # change what the module reads.
    example_kwargs = dict(compile_options.pop("example_kwargs", None) or {})
    example_width = reference[widening[0]]
    widening_kwargs = [
        name
        for name, value in example_kwargs.items()
        if _width_of(value, axis) == example_width
    ]

    plans: list[CompiledModulePlan] = []
    for width in widths:
        resized = tuple(
            _tile_reference(value, axis, width)
            if position in widening
            else value
            for position, value in enumerate(example_inputs)
        )
        options = dict(compile_options)
        if example_kwargs:
            options["example_kwargs"] = {
                name: (
                    _tile_reference(value, axis, width)
                    if name in widening_kwargs
                    else value
                )
                for name, value in example_kwargs.items()
            }
        plans.append(compile_module(module, resized, **options))

    shared = share_plan_state(plans)
    plans_object = BucketedPlans(
        [
            ShapeBucket(width=width, plan=plan)
            for width, plan in zip(widths, shared, strict=True)
        ],
        axis=axis,
        widening=widening,
        widening_kwargs=widening_kwargs,
        input_count=len(example_inputs),
        pad=pad,
        unpad=unpad,
    )
    _log.info(
        "compiled %d buckets on axis %d: %s",
        len(widths),
        axis,
        ", ".join(str(width) for width in widths),
    )
    return plans_object


def load_bucketed(
    index: str | Path, *, require_device_match: bool = False
) -> BucketedPlans:
    """Load the set written by :meth:`BucketedPlans.save`.

    The index names each bucket's artifact; every one is loaded through
    :func:`aotolith.load_compiled_module`, so the ABI, the device profile and the
    per-artifact digest checks all still apply. The weights are then shared
    across the loaded plans, because an artifact carries its own copy of the
    state and a serving process must not hold one per width.
    """

    from .artifact import load_compiled_module

    index_path = Path(index)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    version = payload.get("buckets_index_version")
    if version != BUCKETS_INDEX_VERSION:
        raise CompilationError(
            f"bucket index version {version} is not {BUCKETS_INDEX_VERSION}",
            hint="re-record the bucket set with `aoto compile --bucket-sizes`",
        )
    if payload.get("pad") != "tile":
        raise CompilationError(
            f"bucket index names pad mode {payload.get('pad')!r}, which this "
            "loader cannot reproduce",
            hint="only the built-in tiling padding survives an artifact",
        )
    # The target is compared before any archive is loaded. Every artifact checks
    # its own manifest too, so this is not the only guard - it is the one that
    # fails with "this set was built for another backend" instead of a message
    # about one bucket, and it costs one dictionary lookup.
    recorded_target = payload.get("target", "apple-metal")
    plans: list[CompiledModulePlan] = []
    buckets: list[ShapeBucket] = []
    for entry in payload["buckets"]:
        artifact = index_path.parent / entry["artifact"]
        plan = load_compiled_module(artifact, require_device_match=require_device_match)
        if plan.target != recorded_target:
            raise ArtifactCompatibilityError(
                f"{artifact.name} was built for {plan.target!r} and the index "
                f"records {recorded_target!r}",
                hint="re-record the set; its archives and its index disagree",
            )
        bucket = ShapeBucket(width=int(entry["width"]), plan=plan)
        _check_bucket_against_the_index(bucket, entry, artifact)
        plans.append(plan)
        buckets.append(bucket)
    recorded = [int(width) for width in payload["widths"]]
    if recorded != [bucket.width for bucket in buckets]:
        raise ArtifactCompatibilityError(
            f"the index lists widths {recorded} and its entries are "
            f"{[bucket.width for bucket in buckets]}",
            hint="re-record the bucket set; the index and its artifacts are mixed",
        )
    shared = share_plan_state(plans)
    loaded = BucketedPlans(
        [
            ShapeBucket(width=bucket.width, plan=plan)
            for bucket, plan in zip(buckets, shared, strict=True)
        ],
        axis=int(payload["axis"]),
        widening=[int(position) for position in payload["widening"]],
        widening_kwargs=[str(name) for name in payload.get("widening_kwargs", ())],
        input_count=int(payload["input_count"]),
    )
    _log.info("loaded %d buckets from %s", len(loaded.buckets), index_path)
    return loaded


def _check_bucket_against_the_index(
    bucket: ShapeBucket, entry: Mapping[str, Any], artifact: Path
) -> None:
    """Refuse a bucket whose plan does not match what the index recorded.

    ``planned_peak_bytes`` is derived from the graph's shapes, so it is a
    fingerprint: the same artifact always rebuilds to the same number. A mismatch
    means the entry and the file describe different graphs - a copied artifact, a
    hand-edited index, a stale file after a recompile - and the index's width
    would then dispatch to a plan compiled for a different one.
    """

    recorded = entry.get("planned_peak_bytes")
    if recorded is None:
        return
    actual = bucket.memory_report().planned_peak_bytes
    if int(recorded) != actual:
        raise ArtifactCompatibilityError(
            f"{artifact.name} plans a {actual}-byte peak and the index recorded "
            f"{recorded} for width {bucket.width}",
            hint="re-record the bucket set; this artifact is not the one the index names",
        )


__all__ = [
    "BUCKETS_INDEX_VERSION",
    "PAD_MODES",
    "BucketFootprint",
    "BucketStats",
    "BucketedPlans",
    "ShapeBucket",
    "compile_bucketed",
    "load_bucketed",
    "share_plan_state",
]
