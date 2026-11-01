"""The error contract: what a failure means, and what a program can branch on.

Every refusal the compiler makes used to be a bare ``ValueError``,
``NotImplementedError`` or ``RuntimeError`` with an English sentence in it. That
is readable to a person and useless to a program: a serving layer that wants to
fall back to eager PyTorch when a model uses an operator the compiler does not
lower could only match on message text, and a retry policy could not tell a bad
input apart from a bad device.

So the classes here name the four things a caller actually decides between:

* :class:`CompilationError` - this model, on this target, cannot be compiled as
  written. The caller's options are changing the model, changing the compile
  options, or not using the compiler for this model.
* :class:`PlanInputError` - the plan is fine, the call was not. Raising a
  different-shaped tensor, or the wrong count of them, is the caller's bug.
* :class:`DeviceError` - the plan is fine and so is the call, but this machine
  cannot run it (no Metal, unreachable GPU, a kernel that will not load).
* :class:`ArtifactError` - the artifact is fine and so is this machine, but the
  file is damaged or was written by a compiler whose ABI this build does not
  speak.

:class:`PlatformError` is the fifth, and it exists because a missing optional
dependency is not any of the above: nothing about the model or the machine is
wrong, the installation is.

Each class carries a stable ``code``. The code is the part that goes in a log
line, a metric label, or a user-facing support ticket, and it does not change
when the message is rewritten for clarity. The catalog is
``docs/errors.md``; a code that appears in a message and not in that file is a
bug.

The builtin bases are deliberate. ``CompilationError`` subclasses
``ValueError`` and ``UnsupportedOperationError`` also subclasses
``NotImplementedError``, so every ``except ValueError`` written against the
previous release keeps working while new code can catch the narrower class.
That is a compatibility promise, not an accident of the hierarchy: a caller
that already handles these failures keeps handling them.
"""

from __future__ import annotations

from collections.abc import Iterator


class AotolithError(Exception):
    """Base class for every error this package raises on purpose.

    Catching this class is how a caller says "anything the compiler decided",
    as opposed to a bug in this package (an ``AttributeError`` from a typo) or
    an error from MLX, Torch or the filesystem underneath it.
    """

    #: Stable identifier, in ``AOTO-<area><number>`` form. See ``docs/errors.md``.
    code = "AOTO-0000"

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        #: An actionable next step, when there is one. Kept apart from the
        #: message so a UI can show the diagnosis and fold the advice away.
        self.hint = hint
        if code is not None:
            self.code = code

    def __str__(self) -> str:
        return self.message

    def describe(self) -> str:
        """Render the code, the message, and the hint as one line block."""

        parts = [f"[{self.code}] {self.message}"]
        if self.hint:
            parts.append(f"hint: {self.hint}")
        return "\n".join(parts)


class CompilationError(AotolithError, ValueError):
    """The model cannot be compiled as written on this target."""

    code = "AOTO-1000"


class UnsupportedOperationError(CompilationError, NotImplementedError):
    """An operator, pattern or tracing behaviour the compiler does not lower.

    Also a ``NotImplementedError`` because that is what the previous release
    raised for these, and callers may already be catching it.
    """

    code = "AOTO-1001"


class DynamicShapeError(CompilationError):
    """A shape, size or result length that is only known once the data is.

    The compiler plans static shapes, so this is a boundary of the design
    rather than a gap in coverage. The hint names the usual sources.
    """

    code = "AOTO-1002"


class UnsupportedDTypeError(CompilationError):
    """A dtype the backend does not speak, or one it speaks only per operator."""

    code = "AOTO-1003"


class UnsupportedModelError(CompilationError, NotImplementedError):
    """A checkpoint this build does not compile, though the compiler works.

    Distinct from :class:`UnsupportedOperationError` in what the caller can do
    about it. An operator with no lowering is a hole in coverage that may close
    in the next release; an unsupported *checkpoint* is usually a layout the
    adapter was not written for - a model family without a frontend, a RoPE
    scaling mode that would need a different rotary table, a sliding-window
    attention the mask cannot express. The hint says which of those it is,
    because "compile a different model" and "wait for a lowering" are not the
    same answer.

    Also a ``NotImplementedError``, which is what the previous release raised
    for these, so a caller that already catches it keeps working.
    """

    code = "AOTO-1004"


class IRValidationError(AotolithError, ValueError):
    """A graph that violates an operator's contract.

    Raised while the IR is built, never at run time: shape, dtype, attribute
    and rank checks all happen here so a malformed graph cannot reach a kernel.
    """

    code = "AOTO-1010"


class PlanInputError(AotolithError, ValueError):
    """A call whose arguments do not match the plan's signature."""

    code = "AOTO-2001"


class PlanDTypeError(PlanInputError, TypeError):
    """An argument whose dtype the plan does not accept for that slot.

    Also a ``TypeError``, because that is both what the previous release raised
    and what a Python caller expects from a wrong-typed argument.
    """

    code = "AOTO-2003"


class IndexBoundsError(AotolithError, IndexError):
    """An index tensor that would read outside the tensor it indexes.

    Matches PyTorch's ``IndexError`` so a caller that already handles eager
    failures handles these too, but names the axis and the value, which the
    eager message does not.
    """

    code = "AOTO-2002"


class MemoryBudgetError(AotolithError, RuntimeError):
    """A configuration whose memory does not fit the budget it was given.

    Raised before any allocation happens, from the static footprint rather than
    from a failed ``malloc``: a host admitting a request wants "this sequence
    needs 3.2 GiB and the budget is 2.0 GiB" while it can still answer with a
    shorter sequence, not an unrecoverable device allocation failure.
    """

    code = "AOTO-2004"


class BucketRangeError(PlanInputError):
    """A request wider than the widest compiled bucket.

    A plan set serves every width up to its largest bucket and refuses the rest,
    because the alternative - compiling during the request - costs hundreds of
    milliseconds and cannot be bounded. A caller that admits requests catches
    this to answer "too long" while it can still reply with a shorter sequence.
    """

    code = "AOTO-2005"


class DeviceError(AotolithError, RuntimeError):
    """This machine cannot run the plan, though the plan is valid."""

    code = "AOTO-3001"


class ArtifactError(AotolithError, RuntimeError):
    """An artifact that is damaged, truncated, or structurally invalid."""

    code = "AOTO-4001"


class ArtifactCompatibilityError(ArtifactError):
    """An artifact this build of the compiler cannot read.

    Separate from :class:`ArtifactError` because the two need different
    responses: a damaged file is retried from its source, an incompatible one
    needs the compiler that wrote it or a recompile.
    """

    code = "AOTO-4002"


class PlatformError(AotolithError, ImportError):
    """The installation cannot work on this machine.

    A missing ``mlx``, a non-Apple platform, or a Torch version the frontend
    needs and does not have. Nothing about the model is wrong.
    """

    code = "AOTO-5001"


class BenchmarkError(AotolithError, ValueError):
    """The benchmark gate cannot run, or cannot use what it was pointed at.

    Not a compiler failure and not a model failure: the code compiled, and the
    machine is capable. What is wrong is the gate's own input - a baseline file
    or directory that cannot be trusted to describe a machine, which is a
    configuration problem with a configuration fix.

    Subclasses ``ValueError`` because that is what the gate raised before it
    had a class, so a caller that already catches the builtin keeps catching
    this.
    """

    code = "AOTO-6000"


class BaselineError(BenchmarkError):
    """A baseline file or matrix directory this run cannot use.

    The response differs from a regression: nothing about the compiler is wrong,
    and rerunning changes nothing. The file has to be re-recorded, renamed to
    the chip it records, or rewritten at the right schema.
    """

    code = "AOTO-6001"


#: Every code this module defines, for the catalog test and for ``docs``.
_ERROR_CLASSES: tuple[type[AotolithError], ...] = (
    AotolithError,
    CompilationError,
    UnsupportedOperationError,
    DynamicShapeError,
    UnsupportedDTypeError,
    UnsupportedModelError,
    IRValidationError,
    PlanInputError,
    PlanDTypeError,
    IndexBoundsError,
    MemoryBudgetError,
    BucketRangeError,
    DeviceError,
    ArtifactError,
    ArtifactCompatibilityError,
    PlatformError,
    BenchmarkError,
    BaselineError,
)


def error_codes() -> dict[str, type[AotolithError]]:
    """Return ``code -> class`` for every error this package defines."""

    return {cls.code: cls for cls in _ERROR_CLASSES}


def iter_error_classes() -> Iterator[type[AotolithError]]:
    for cls in _ERROR_CLASSES:
        yield cls


def format_error(error: BaseException) -> str:
    """Render any exception the way the ``aoto`` command prints it."""

    if isinstance(error, AotolithError):
        return error.describe()
    return str(error)


__all__: list[str] = [
    "AotolithError",
    "ArtifactCompatibilityError",
    "ArtifactError",
    "BaselineError",
    "BenchmarkError",
    "BucketRangeError",
    "CompilationError",
    "DeviceError",
    "DynamicShapeError",
    "IRValidationError",
    "IndexBoundsError",
    "MemoryBudgetError",
    "PlanInputError",
    "PlanDTypeError",
    "PlatformError",
    "UnsupportedDTypeError",
    "UnsupportedModelError",
    "UnsupportedOperationError",
    "error_codes",
    "format_error",
    "iter_error_classes",
]
