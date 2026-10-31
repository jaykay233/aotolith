"""Assemble the contract table from the operator families.

Each family module owns its rows and exposes them as ``CONTRACTS``. This module
concatenates them and adds the two groups that are generated rather than
written out:

* every pointwise operator shares one contract, because they all take broadcast
  operands and produce the output dtype their own emitter decides;
* every constant producer shares the attribute rule that makes its ``value`` a
  literal.

Keeping the assembly in one place is what lets the families stay independent:
none of them imports another, so a change to the attention contract cannot
affect the view table, and the one place where the whole set must be consistent
is this file.
"""

from __future__ import annotations

from ...op_sets import CONSTANT_OPS, POINTWISE_OPS
from .. import Graph, Operation
from . import constants, elementwise, indexing, kernels, linalg, reduction, view
from .rules import AttrKind, AttrSpec, DtypeRule, OpContract, ShapeRule, check_contract

_FAMILIES = (
    reduction.CONTRACTS,
    elementwise.CONTRACTS,
    kernels.CONTRACTS,
    indexing.CONTRACTS,
    view.CONTRACTS,
    constants.CONTRACTS,
    linalg.CONTRACTS,
)

OP_CONTRACTS: dict[str, OpContract] = {}
for _family in _FAMILIES:
    for _name, _contract in _family.items():
        if _name in OP_CONTRACTS:
            raise ValueError(
                f"{_name} is declared in two operator families; a contract "
                "must have exactly one owner"
            )
        OP_CONTRACTS[_name] = _contract

# A pointwise operator is an elementwise pass over broadcast operands. The
# per-operator dtype restrictions stay with the emitter that has to express
# them, because they are about what MSL can compile, not about the operator.
for _pointwise_op in POINTWISE_OPS:
    OP_CONTRACTS.setdefault(
        _pointwise_op,
        OpContract(
            arity=(1, 3),
            rank=(0, 8),
            dtype=DtypeRule.ANY,
            shape=ShapeRule.FREE,
            native=True,
        ),
    )

for _constant_op in CONSTANT_OPS:
    OP_CONTRACTS[_constant_op] = OpContract(
        arity=(0, 1),
        rank=(0, 8),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("value", AttrKind.LITERAL),),
        native=True,
    )


def contract_for(op_name: str) -> OpContract:
    """The contract for ``op_name``, or ``KeyError`` naming the operator."""

    try:
        return OP_CONTRACTS[op_name]
    except KeyError:
        raise KeyError(f"no contract declared for operator {op_name!r}") from None


def check_operation(graph: Graph, op: Operation) -> None:
    """Validate one operation against the contract the table holds for it.

    Raises ``ValueError`` for a malformed operation and ``KeyError`` for one
    this build has no contract for, so an unexpected operator fails by name
    instead of reaching a runtime that cannot execute it.
    """

    check_contract(graph, op, contract_for(op.op))
