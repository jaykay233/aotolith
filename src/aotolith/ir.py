from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DType(str, Enum):
    F16 = "float16"
    BF16 = "bfloat16"
    F32 = "float32"
    BOOL = "bool"
    I32 = "int32"
    I64 = "int64"


@dataclass(frozen=True, slots=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"tensor dimensions must be static and positive: {self.shape}")


@dataclass(frozen=True, slots=True)
class Value:
    id: str
    type: TensorType


@dataclass(frozen=True, slots=True)
class Operation:
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.outputs) != 1:
            raise ValueError(f"{self.op} must produce exactly one value")


@dataclass(frozen=True, slots=True)
class Graph:
    inputs: tuple[str, ...]
    values: dict[str, Value]
    ops: tuple[Operation, ...]
    outputs: tuple[str, ...]

    def validate(self) -> None:
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("graph inputs must be unique")
        if not set(self.inputs).issubset(self.values):
            raise ValueError("every graph input must have a value type")
        if not set(self.outputs).issubset(self.values):
            raise ValueError("every graph output must have a value type")

        defined = set(self.inputs)
        for op in self.ops:
            missing = set(op.inputs) - defined
            if missing:
                raise ValueError(f"{op.op} uses undefined values: {sorted(missing)}")
            untyped_outputs = set(op.outputs) - set(self.values)
            if untyped_outputs:
                raise ValueError(
                    f"{op.op} produces values without types: {sorted(untyped_outputs)}"
                )
            duplicate = defined.intersection(op.outputs)
            if duplicate:
                raise ValueError(f"values defined more than once: {sorted(duplicate)}")
            defined.update(op.outputs)

        missing_outputs = set(self.outputs) - defined
        if missing_outputs:
            raise ValueError(f"undefined graph outputs: {sorted(missing_outputs)}")

    def value(self, value_id: str) -> Value:
        return self.values[value_id]

    def replace_ops(self, ops: tuple[Operation, ...]) -> Graph:
        graph = Graph(self.inputs, self.values, ops, self.outputs)
        graph.validate()
        return graph
