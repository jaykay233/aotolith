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


#: Bytes one element of each dtype occupies.
#:
#: It lives here rather than next to the kernel generator because everything
#: above this layer sizes something: the storage plan sizes buffers, the memory
#: accounting reports them, and the codegen sizes a kernel. Two tables would
#: drift, and the drift would show up as a buffer that is too small.
DTYPE_ITEMSIZE: dict[DType, int] = {
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.BOOL: 1,
    DType.I32: 4,
    DType.I64: 8,
}


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
