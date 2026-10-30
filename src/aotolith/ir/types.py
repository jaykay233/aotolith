"""The typed SSA types a graph is built from.

``DType``, ``TensorType`` and ``Value`` are the vocabulary the rest of the
compiler speaks: an operator says what it consumes and produces in terms of
them, the contract table declares rules over them, and the backend computes
sizes from them. Nothing here knows about operations or graphs, which is what
lets the contract table import it without pulling in the whole IR.
"""

from __future__ import annotations

from dataclasses import dataclass
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
            raise ValueError(
                f"tensor dimensions must be static and positive: {self.shape}"
            )


@dataclass(frozen=True, slots=True)
class Value:
    id: str
    type: TensorType
