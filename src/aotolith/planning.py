from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .ir import DType, Graph, TensorType

_DTYPE_BYTES = {
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.BOOL: 1,
    DType.I32: 4,
    DType.I64: 8,
}
_BUFFER_ALIGNMENT = 16


@dataclass(frozen=True, slots=True)
class ValueLifetime:
    value_id: str
    tensor_type: TensorType
    start: int
    end: int
    size_bytes: int
    is_input: bool
    is_output: bool

    @property
    def overlaps(self) -> bool:
        return self.end > self.start


@dataclass(frozen=True, slots=True)
class BufferSlot:
    slot_id: int
    offset_bytes: int
    size_bytes: int
    alignment_bytes: int
    value_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StoragePlan:
    lifetimes: tuple[ValueLifetime, ...]
    slots: tuple[BufferSlot, ...]
    release_after_op: tuple[tuple[str, ...], ...]
    peak_live_bytes: int
    arena_size_bytes: int


def plan_storage(graph: Graph) -> StoragePlan:
    """Build a conservative lifetime and reusable-slot plan for ``graph``."""

    graph.validate()
    output_set = set(graph.outputs)
    input_set = set(graph.inputs)
    producers = {
        op.outputs[0]: index for index, op in enumerate(graph.ops)
    }
    active_values = input_set | set(producers)
    if active_values != set(graph.values):
        missing = active_values - set(graph.values)
        if missing:
            raise ValueError(f"active values have no type: {sorted(missing)}")
    last_use = {value_id: -1 for value_id in active_values}
    for index, op in enumerate(graph.ops):
        for value_id in op.inputs:
            last_use[value_id] = index
    terminal = len(graph.ops)
    for value_id in graph.outputs:
        last_use[value_id] = terminal

    lifetimes = tuple(
        ValueLifetime(
            value_id=value_id,
            tensor_type=graph.value(value_id).type,
            start=producers.get(value_id, -1),
            end=max(last_use[value_id], producers.get(value_id, -1)) + 1,
            size_bytes=_tensor_size_bytes(graph.value(value_id).type),
            is_input=value_id in input_set,
            is_output=value_id in output_set,
        )
        for value_id in sorted(active_values)
    )

    release: list[list[str]] = [[] for _ in graph.ops]
    for value_id, last_index in last_use.items():
        effective_last_use = max(last_index, producers.get(value_id, -1))
        if (
            value_id not in output_set
            and effective_last_use >= 0
            and effective_last_use < len(graph.ops)
        ):
            release[effective_last_use].append(value_id)
    release_after_op = tuple(tuple(sorted(values)) for values in release)

    reusable = [
        lifetime
        for lifetime in lifetimes
        if not lifetime.is_input and not lifetime.is_output and lifetime.overlaps
    ]
    slot_values: list[list[ValueLifetime]] = []
    slot_sizes: list[int] = []
    for lifetime in reusable:
        for slot_index, size_bytes in enumerate(slot_sizes):
            if size_bytes < lifetime.size_bytes:
                continue
            if all(
                not _intervals_overlap(lifetime, existing)
                for existing in slot_values[slot_index]
            ):
                slot_values[slot_index].append(lifetime)
                break
        else:
            slot_values.append([lifetime])
            slot_sizes.append(lifetime.size_bytes)

    slots: list[BufferSlot] = []
    offset = 0
    for slot_id, (size_bytes, values) in enumerate(
        zip(slot_sizes, slot_values, strict=True)
    ):
        offset = _align_up(offset, _BUFFER_ALIGNMENT)
        slots.append(
            BufferSlot(
                slot_id=slot_id,
                offset_bytes=offset,
                size_bytes=size_bytes,
                alignment_bytes=_BUFFER_ALIGNMENT,
                value_ids=tuple(value.value_id for value in values),
            )
        )
        offset += size_bytes

    return StoragePlan(
        lifetimes=lifetimes,
        slots=tuple(slots),
        release_after_op=release_after_op,
        peak_live_bytes=_peak_live_bytes(lifetimes),
        arena_size_bytes=_align_up(offset, _BUFFER_ALIGNMENT),
    )


def _tensor_size_bytes(tensor_type: TensorType) -> int:
    return prod(tensor_type.shape) * _DTYPE_BYTES[tensor_type.dtype]


def _intervals_overlap(left: ValueLifetime, right: ValueLifetime) -> bool:
    return left.start < right.end and right.start < left.end


def _peak_live_bytes(lifetimes: tuple[ValueLifetime, ...]) -> int:
    events: dict[int, int] = {}
    for lifetime in lifetimes:
        events[lifetime.start] = events.get(lifetime.start, 0) + lifetime.size_bytes
        events[lifetime.end] = events.get(lifetime.end, 0) - lifetime.size_bytes
    current = 0
    peak = 0
    for index in sorted(events):
        current += events[index]
        peak = max(peak, current)
    return peak


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
