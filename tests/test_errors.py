"""The error contract: one class per decision, and a code that does not drift.

These tests are about the promise, not about any particular message. A message
may be rewritten for clarity at any time; the class a caller catches and the
code a caller logs must not change, so both are asserted here.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
import torch

import aotolith
from aotolith import compile_callable
from aotolith.errors import (
    AotolithError,
    ArtifactCompatibilityError,
    ArtifactError,
    CompilationError,
    DeviceError,
    DynamicShapeError,
    IRValidationError,
    IndexBoundsError,
    PlanDTypeError,
    PlanInputError,
    PlatformError,
    UnsupportedDTypeError,
    UnsupportedOperationError,
    error_codes,
    format_error,
    iter_error_classes,
)

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_every_error_is_an_aotolith_error() -> None:
    for cls in iter_error_classes():
        assert issubclass(cls, AotolithError), cls.__name__


def test_codes_are_unique_and_well_formed() -> None:
    codes = [cls.code for cls in iter_error_classes()]
    assert len(codes) == len(set(codes)), "two error classes share a code"
    for code in codes:
        area, _, number = code.partition("-")
        assert area == "AOTO", code
        assert len(number) == 4 and number.isdigit(), code


def test_the_builtin_bases_keep_old_handlers_working() -> None:
    # These four are compatibility promises: the previous release raised the
    # builtin, and a caller that caught it must keep catching the new class.
    assert issubclass(UnsupportedOperationError, NotImplementedError)
    assert issubclass(UnsupportedOperationError, ValueError)
    assert issubclass(CompilationError, ValueError)
    assert issubclass(PlanInputError, ValueError)
    assert issubclass(PlanDTypeError, TypeError)
    assert issubclass(IndexBoundsError, IndexError)
    assert issubclass(ArtifactError, RuntimeError)
    assert issubclass(ArtifactCompatibilityError, ArtifactError)
    assert issubclass(DeviceError, RuntimeError)
    assert issubclass(PlatformError, ImportError)
    assert issubclass(IRValidationError, ValueError)


def test_a_hint_is_rendered_apart_from_the_message() -> None:
    error = PlanInputError("expected 2 inputs", hint="pass both")
    assert str(error) == "expected 2 inputs"
    assert error.hint == "pass both"
    assert format_error(error) == f"[{PlanInputError.code}] expected 2 inputs\nhint: pass both"
    assert format_error(ValueError("plain")) == "plain"


def test_the_public_package_exposes_the_hierarchy() -> None:
    for name in (
        "AotolithError",
        "CompilationError",
        "UnsupportedOperationError",
        "DynamicShapeError",
        "PlanInputError",
        "IndexBoundsError",
        "ArtifactError",
        "ArtifactCompatibilityError",
        "DeviceError",
        "PlatformError",
    ):
        assert hasattr(aotolith, name), name
        assert name in aotolith.__all__


def test_the_docs_catalog_names_every_code() -> None:
    """``docs/errors.md`` is the catalog; a code missing from it is a bug."""

    catalog = (DOCS / "errors.md").read_text(encoding="utf-8")
    for code in error_codes():
        assert code in catalog, f"{code} is not in docs/errors.md"
    translated = (DOCS / "errors.zh-CN.md").read_text(encoding="utf-8")
    for code in error_codes():
        assert code in translated, f"{code} is not in docs/errors.zh-CN.md"


def test_an_unsupported_operator_is_one_error_class() -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        # ``mode`` has no lowering, and the refusal is the one a user meets
        # most often: a model that uses an operator outside the covered set.
        return torch.mode(value).values

    with pytest.raises(UnsupportedOperationError, match="unsupported FX target") as caught:
        compile_callable(fn, (torch.randn(8),))
    assert caught.value.code == UnsupportedOperationError.code
    assert caught.value.hint


def test_a_data_dependent_shape_is_a_compilation_error() -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return torch.unique(value)

    with pytest.raises(DynamicShapeError, match="static shapes"):
        compile_callable(fn, (torch.tensor([1.0, 2.0, 2.0]),))


def test_a_dtype_the_backend_does_not_speak_is_named() -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return value * 2.0

    with pytest.raises(UnsupportedDTypeError, match="unsupported dtype"):
        compile_callable(fn, (torch.randn(4, dtype=torch.float64),))


def test_out_of_range_indices_name_the_value_and_the_range() -> None:
    def fn(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(indices, source)

    table = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    result = compile_callable(fn, (table, torch.tensor([0, 5], dtype=torch.int64)))
    with pytest.raises(IndexBoundsError, match=r"index out of range: 7 is") as caught:
        result.plan(
            mx.array(table.numpy()), mx.array([0, 7], dtype=mx.int64)
        )
    # The old message said "index out of range in self" and nothing else.
    assert "[0, 5]" in str(caught.value)
    assert caught.value.hint


def test_advanced_indexing_reports_the_range_it_accepts() -> None:
    # Advanced indexing accepts negative indices, so the reported range has to
    # say so - a caller reading "[0, 5]" would think -1 was a bug in the plan.
    def fn(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return source[indices]

    table = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    result = compile_callable(fn, (table, torch.tensor([0, 5], dtype=torch.int64)))
    with pytest.raises(IndexBoundsError, match=r"\[-6, 5\]"):
        result.plan(mx.array(table.numpy()), mx.array([-9, 0], dtype=mx.int64))
    # -1 is legal here and must not be refused.
    negative = result.plan(
        mx.array(table.numpy()), mx.array([-1, 0], dtype=mx.int64)
    )[0]
    mx.eval(negative)
    assert negative.tolist() == [[10.0, 11.0], [0.0, 1.0]]


def test_embedding_reports_its_non_negative_range() -> None:
    # ``F.embedding`` refuses negative indices in eager mode too, so the plan
    # must not advertise a range it does not accept.
    def fn(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(indices, source)

    table = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    result = compile_callable(fn, (table, torch.tensor([0, 5], dtype=torch.int64)))
    with pytest.raises(IndexBoundsError, match=r"\[0, 5\]"):
        result.plan(mx.array(table.numpy()), mx.array([-7, 0], dtype=mx.int64))


def test_a_plan_reports_a_bad_call_as_a_plan_input_error() -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return value + 1.0

    result = compile_callable(fn, (torch.randn(4),))
    with pytest.raises(PlanInputError, match="expected 1 inputs"):
        result.plan(mx.zeros((4,), dtype=mx.float32), mx.zeros((4,), dtype=mx.float32))
    with pytest.raises(PlanDTypeError, match="dtype"):
        result.plan(mx.zeros((4,), dtype=mx.float16))


def test_a_missing_torch_is_a_platform_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from aotolith.capture import tracer

    # ``make_fx`` is imported inside ``trace``; hiding the module is what a
    # runtime without the ``fx`` extra looks like.
    monkeypatch.setitem(sys.modules, "torch.fx.experimental.proxy_tensor", None)
    with pytest.raises(PlatformError, match="PyTorch is required"):
        tracer.trace(lambda value: value, (torch.randn(2),), frozenset())
