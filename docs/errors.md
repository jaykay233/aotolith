# Errors

Every failure this package raises on purpose is an `AotolithError` with a stable
code. The class says what a caller should *do*; the code is what belongs in a log
line, a metric label, or a bug report, and it does not change when the message is
rewritten.

```python
import aotolith as aoto
from aotolith import errors

try:
    compiled = aoto.compile_module(model, (input_ids,))
except errors.UnsupportedOperationError as exc:
    print(exc.code, exc)          # AOTO-1001 unsupported FX target: aten.mode.default
    print(exc.hint)               # what to do about it
    fall_back_to_eager()
except errors.CompilationError:
    # Any refusal decided by the front end or the mid-end.
    report_unsupported_model()
```

`format_error(exc)` renders an `AotolithError` as one block — `[code] message`
followed by `hint: …` — which is exactly what the `aoto` command prints.

## Codes

| code | class | means | caller's move |
| --- | --- | --- | --- |
| `AOTO-0000` | `AotolithError` | base class only; never raised directly | catch this to mean "a deliberate refusal" |
| `AOTO-1000` | `CompilationError` | the model cannot be compiled as written | change the model, the compile options, or stop using the compiler for it |
| `AOTO-1001` | `UnsupportedOperationError` | an operator, tracing behaviour or pattern with no lowering | run it outside the compiled region, or add a lowering |
| `AOTO-1002` | `DynamicShapeError` | a size or result length that is only known once the data is | pass the extent in from an input shape, or move the operator out |
| `AOTO-1003` | `UnsupportedDTypeError` | a dtype the backend does not speak for that operator | cast before the boundary, or leave the operator out |
| `AOTO-1004` | `UnsupportedModelError` | a checkpoint this build does not compile: another model family, a RoPE scaling mode, sliding-window attention | compile the family whose frontend handles it, or wait for the adapter; the hint names which |
| `AOTO-1010` | `IRValidationError` | a graph that violates an operator's contract | a compiler bug, or a graph built by hand — report it |
| `AOTO-2001` | `PlanInputError` | the call does not match the plan's signature | fix the call: input count and shapes come from `plan.graph.inputs` |
| `AOTO-2002` | `IndexBoundsError` | an index that would read outside the tensor it indexes | fix the indices, or pass `verify_index_bounds=False` when they are known good |
| `AOTO-2003` | `PlanDTypeError` | an argument whose dtype the plan does not accept | cast the argument, or recompile with that dtype |
| `AOTO-2004` | `MemoryBudgetError` | a configuration whose memory does not fit the budget it was given | shorten the sequence, compile with `quantize=4`, or raise the budget |
| `AOTO-2005` | `BucketRangeError` | a request wider than the widest compiled bucket | compile a wider bucket, or split the request; nothing is recompiled at run time |
| `AOTO-3001` | `DeviceError` | the plan is valid, this machine cannot run it | run on a supported Apple GPU, outside a sandbox that hides Metal |
| `AOTO-4001` | `ArtifactError` | a damaged or structurally invalid artifact | recompile, or restore the file |
| `AOTO-4002` | `ArtifactCompatibilityError` | an artifact this build cannot read | use the compiler that wrote it, or recompile |
| `AOTO-5001` | `PlatformError` | the installation cannot work here | fix the installation: platform, or the missing extra |
| `AOTO-6000` | `BenchmarkError` | the benchmark gate cannot run, or cannot use what it was pointed at | fix the gate's input, not the compiler: nothing about the model is wrong |
| `AOTO-6001` | `BaselineError` | a baseline file or matrix directory that cannot be trusted to describe a machine | re-record it, rename it to the chip it records, or re-record at the current schema |

The codes are grouped by area: `1xxx` is decided at compile time, `2xxx` at call
time, `3xxx` by the machine, `4xxx` by a stored artifact, `5xxx` by the
installation, `6xxx` by the gate's own inputs.

`6xxx` is deliberately outside the other five: a failed gate is not a bad model,
a bad call, a bad device, a bad artifact or a bad installation. The usual
`BaselineError` is a committed file that stopped describing its chip — it was
renamed, recorded by an older schema, or its device name was edited — and the
fix is to the file. It subclasses `ValueError` because that is what the gate
raised before it had a class.

## Compatibility with the builtin exceptions

Each class also subclasses the builtin the previous release raised, so code
written against that release keeps working:

| class | also |
| --- | --- |
| `CompilationError` | `ValueError` |
| `UnsupportedOperationError` | `ValueError`, `NotImplementedError` |
| `UnsupportedModelError` | `ValueError`, `NotImplementedError` |
| `PlanInputError` | `ValueError` |
| `PlanDTypeError` | `TypeError` |
| `IndexBoundsError` | `IndexError` |
| `ArtifactError`, `DeviceError` | `RuntimeError` |
| `PlatformError` | `ImportError` |
| `BenchmarkError`, `BaselineError` | `ValueError` |

This is a promise, not a coincidence. `except ValueError` around a compile call
still catches every refusal it used to.

## What is *not* an `AotolithError`

MLX, Torch and the filesystem raise their own exceptions and they are passed
through unchanged — an `OSError` from a full disk is not a compiler decision, and
wrapping it would hide that. The one exception is a missing optional dependency:
`from aotolith.capture import capture_fx` without Torch installed raises
`PlatformError` (`AOTO-5001`), because "the installation is incomplete" is more
useful than "No module named torch".

## Logging

The package logs to `logging.getLogger("aotolith")`, with a child per module. It
installs a `NullHandler` and nothing else, so importing it never prints and never
touches the root logger. A host configures it like any other library:

```python
import logging

logging.getLogger("aotolith").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)   # or its own handler on that name
```

`aotolith.configure_logging(level)` is the shortcut for scripts: it attaches one
stderr handler and sets the level, and does nothing else. The `aoto` command
calls it, which is why `aoto --log-level INFO compile …` works; `AOTOLITH_LOG=DEBUG`
has the same effect without a flag.

What the levels mean here:

| level | what is written |
| --- | --- |
| `DEBUG` | per-operation decisions: which implementation each value took and why, families with no native candidate |
| `INFO` | one line per session event: the compile summary, artifact written, artifact read |
| `WARNING` | something was downgraded: tensor cores refused, autotune found no winner, an older artifact format was read |
| `ERROR` | a plan was produced or a run failed in a way that is expected to fail |

The same information is available in-process on `plan.selections`, which carries
the reason for every operation. Logging exists for the case where you do not have
the plan in hand yet.
