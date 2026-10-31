"""Metal source emission: one module per operator family.

Every module here writes MSL text and nothing else - no MLX calls, no plan, no
device probing. The `_compile_*` method that turns an IR operation into a
`KernelArtifact` lives beside the emitters it calls, so "how is a matmul
compiled" has exactly one file to read.

The one thing the families share is `msl`, which owns the preamble every kernel
starts from and the compile-and-cache step. That is also why `expr` and
`dtypes` sit underneath: an expression emitter needs the dtype aliases and the
literal syntax, and neither needs to know which family is asking.
"""

from __future__ import annotations

from .attention import AttentionLowering
from .index import IndexLowering
from .matmul import MatmulLowering
from .msl import KernelArtifact
from .norm import NormLowering
from .pointwise import PointwiseLowering
from .quantize import QuantizationLowering
from .reduction import ReductionLowering

__all__ = [
    "AttentionLowering",
    "IndexLowering",
    "KernelArtifact",
    "MatmulLowering",
    "NormLowering",
    "PointwiseLowering",
    "QuantizationLowering",
    "ReductionLowering",
]
