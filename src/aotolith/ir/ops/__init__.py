"""Declarative definitions for the compiler's IR operations.

Every IR operation the backend executes is described once: how many operands it
takes, which attributes it must carry, what its output shape and dtype must be,
and how many values it writes. Whatever those fields cannot say - a scatter's
index/source compatibility, an embedding's float weight and integer indices -
lives in an ``extra`` hook beside the row instead of in a separate branch of the
backend.

The table exists so a new operator is one row plus, at most, one hook, rather
than another ``if op.op == ...`` arm with its own hand-written rank, dtype and
shape preamble. The generic checks in ``check_operation`` are the ones the
operators were each repeating; the table says which apply and with what
parameters.

Layout:

* ``rules`` - the rule vocabulary (``OpContract`` and friends) and the one
  generic checker. No operator knowledge.
* ``shared`` - helpers the family hooks share.
* one module per family - ``reduction``, ``elementwise``, ``kernels``,
  ``indexing``, ``view``, ``constants`` - each exposing its own ``CONTRACTS``.
* ``table`` - concatenates the families and adds the generated pointwise and
  constant rows.

Families do not import each other, so a change to one cannot affect another.
"""

from __future__ import annotations

from .constants import check_integer_constant, integer_constant_value
from .rules import (
    FLOATING_DTYPES,
    INDEX_DTYPES,
    INTEGRAL_DTYPES,
    SCATTER_REDUCES,
    AttrKind,
    AttrSpec,
    DtypeRule,
    OpContract,
    ShapeRule,
    check_contract,
    keepdim,
    normalize_axes,
    normalized_axes,
    reduced_shape,
)
from .table import OP_CONTRACTS, check_operation, contract_for

__all__ = [
    "INDEX_DTYPES",
    "INTEGRAL_DTYPES",
    "FLOATING_DTYPES",
    "OP_CONTRACTS",
    "SCATTER_REDUCES",
    "AttrKind",
    "AttrSpec",
    "DtypeRule",
    "OpContract",
    "ShapeRule",
    "check_contract",
    "check_integer_constant",
    "check_operation",
    "contract_for",
    "integer_constant_value",
    "keepdim",
    "normalize_axes",
    "normalized_axes",
    "reduced_shape",
]
