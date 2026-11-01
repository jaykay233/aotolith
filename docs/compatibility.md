# Compatibility and deprecation

What a caller can rely on between releases, and what happens when something has
to change. The short version: the compiler's output format and its Python API
move at different speeds, and neither moves silently.

## The four surfaces

| surface | promise | how it changes |
| --- | --- | --- |
| **Artifact format** (`.aoto`, `format_version` / `abi_version`) | a file written by 0.x is readable by every later 0.x in the same major line, for as long as the reader keeps a compatibility branch | a new format version is only written when the reader can also read the previous one; dropping a reader branch requires a minor version and a `CHANGELOG` entry |
| **Python API** (`aotolith.__all__`: `compile_module`, `compile_callable`, `compile_graph`, `CompiledPlan`, the error classes, …) | present and working for the whole 0.x line | deprecation, then removal no sooner than two minor versions after the deprecation, and never in a patch release |
| **Error contract** (`AotolithError.code`) | a code, once published in `docs/errors.md`, keeps its meaning | a code is never reused for a different failure; a code that stops being raised is marked deprecated in the catalog before it is removed |
| **CLI** (`aoto compile/inspect/verify/run/bench`, flag names, exit codes) | flags keep their meaning; `0` means success, `1` means a failure was reported, `2` means the command line was wrong | a flag is removed only after a release that warns when it is used |

Anything not exported from `aotolith/__init__.py` — the passes, the codegen
families, the internal helpers prefixed with `_` — is implementation. It may
change in any release. The architecture test pins the import direction for those
modules, not their contents.

A bucket index (`*.buckets.json`) is part of the artifact surface, and carries
its own `buckets_index_version` for the same reason `.aoto` carries
`format_version`: the reader refuses a version it does not speak, and a new
version is only written when the previous one can still be read. The archives it
names are ordinary `.aoto` files and move under the ordinary rules, so an index
whose entries no longer load fails the way a single artifact would.

## What counts as breaking

These are breaking, and need a minor version and a `CHANGELOG` entry under
**Changed** or **Removed**:

- a function or class in `aotolith.__all__` changing its signature in a way that
  rejects calls it used to accept;
- an artifact the previous release wrote no longer loading;
- an error moving between the five areas in `docs/errors.md` (`1xxx` → `2xxx`),
  which would change what a caller branches on;
- a plan computing different numbers for the same artifact, inputs and device.

These are not breaking:

- an error message being rewritten (the class and the code stay);
- a refusal becoming a different, more specific class in the same area;
- a new `check` in capture or legalization that refuses a graph the backend
  could never have executed correctly — the old behaviour was a wrong answer or
  a crash inside a kernel, not a working plan;
- a new operator gaining a native kernel, which changes speed, not results;
- the heuristic selection preferring a different implementation, which is what
  `plan.selections` records per compile.

The third item is the one that needs care: the boundary is "the previous release
produced a correct plan". If a graph used to compile and run, a new refusal for
it is breaking; if it used to fail somewhere less useful, it is not.

## Deprecating something

1. Add the marker where a reader will see it — `DeprecationWarning` for a Python
   symbol, a `WARNING` log line for a runtime behaviour, a note in the help text
   for a CLI flag. Silence is not a deprecation.
2. Add an entry to `CHANGELOG.md` under **Deprecated**, naming the replacement
   and the release that will remove it.
3. Keep it working. A deprecation that changes behaviour is a removal.
4. Remove it no sooner than two minor versions later, and move the entry to
   **Removed** in the release that does.

`DeprecationWarning` is not shown by default, which is why the same fact is also
logged: a user who never runs with `-W error::DeprecationWarning` still gets one
line in the record.

## Version numbers

`0.x` is pre-1.0, so the second number is where everything happens:

- `0.63.0 → 0.63.1` — fixes only. No API change, no artifact change, no change
  to any number a plan computes.
- `0.63.0 → 0.64.0` — anything else: new features, new refusals, changed
  heuristics, a deprecation, a removal.
- `1.0.0` — the point at which the artifact format and the API above become
  promises across a major line instead of a minor one.

## Artifacts across devices

An artifact records the GPU it was compiled on (`profile.device_name`,
`profile.architecture`). Loading does **not** require a match by default: the
Metal kernels are rebuilt from the validated graph and the stored MSL at load
time, so a plan compiled on an M3 Pro runs on an M5 Pro. What can differ is
speed — the heuristic choices were made for the other GPU — and the record says
so. `load_compiled_module(..., require_device_match=True)` is the strict mode for
a deployment that would rather fail than run an unmeasured plan.

## The current known limits

Stated here because they are the parts of the API most likely to move:

- **Static shapes.** Every plan is compiled for fixed shapes, and a prompt length
  or KV-cache size is part of the artifact. Prompt buckets exist in
  `aotolith.models.qwen2` and `aotolith.models.llama`; symbolic shapes do not
  exist anywhere.
- **Two targets, both MLX-shaped.** `apple-metal` and `mlx-reference` are the
  only backends; `compile_graph(target=…)` refuses anything else rather than
  pretending. Neither is reachable from a machine without MLX, so "another
  backend" means "another device", which is still open.
- **Two model families.** `qwen2` and `llama` are the decoder families the
  static-cache adapter has been checked against. A third `model_type` is refused
  with `UnsupportedModelError` (AOTO-1004) rather than guessed at, so adding one
  is a compat-relevant change to what compiles, not a silent widening.
- **Batch size one in the decoder adapter.** The IR supports a batch dimension;
  the adapter's cache and generation loop assume one sequence.
- **Format 4 read support** exists because it costs one branch. It will be
  removed in a minor release after 1.0, with a deprecation log line first.

## Notes for 0.64.0

- `memory=` is a new keyword-only argument on `CompiledPlan.__call__` and
  `CompiledModulePlan.__call__`. It has a default, so existing calls are
  unaffected; the name is reserved, so a model cannot promote a tensor keyword by
  that name (it is refused at compile time, next to `verify_index_bounds`).
- `Qwen2StaticCacheRuntime` lost its `_decode_masks` array and gained
  `decode_mask(valid_length)`. The field was underscore-prefixed and therefore
  implementation by the rule above; the method returns exactly the row the table
  used to hold.
- The benchmark report schema is now 2. `bench.compare` refuses a baseline with a
  different schema rather than comparing mismatched numbers, so a recorded
  baseline must be re-recorded with `aoto bench --update-baseline`.
- `tensor_nbytes` is new in `aotolith.planning`, and `DTYPE_ITEMSIZE` moved from
  `codegen.expr` to `ir.types` (both implementation surfaces).

## Notes for 0.67.0

- **A second model family and a second backend, neither of which changes an
  existing one.** `aotolith.models.llama` is new; `aotolith.models.qwen2` keeps
  every name it had, because the shared engine in `aotolith.models.decoder` is
  what the Qwen2 entry points now call. `Qwen2PrefillModule` and the rest are
  aliases of the decoder classes, so `isinstance` and `type(...).__name__` are
  the only things a caller could notice, and nothing in the public surface
  promises either.
- **`target=` is new on `compile_graph`, `compile_callable`, `compile_module`,
  and both adapters.** It has a default, so existing calls are unaffected. An
  artifact now records the target it was built for and refuses to load as the
  other one; artifacts written by 0.66.0 have no `target` field and are read as
  `apple-metal`, which is what they were.
- **`AOTO-1004` / `UnsupportedModelError` is new** and sits in the same `1xxx`
  area as `AOTO-1001`. Everything it now raises used to be a bare
  `NotImplementedError`; the class also subclasses `NotImplementedError`, so a
  handler written against 0.66.0 keeps working. This is the "more specific class
  in the same area" case, which is not breaking by the rule above.
- **`prefill_mask` bounds are split.** A non-positive length still raises
  `PlanInputError`; a length wider than the bucket now raises `BucketRangeError`
  (`AOTO-2005`), because that is a request that does not fit rather than a bad
  argument. Both subclass `ValueError`, and `BucketRangeError` also subclasses
  `PlanInputError`, so no existing `except` changes meaning.
- **`AOTO-6000` / `BenchmarkError` and `AOTO-6001` / `BaselineError` are new**,
  in a `6xxx` area of their own: a gate whose baseline file or directory cannot
  be trusted is not a compiler failure. Both subclass `ValueError`, which is
  what `bench.resolve_baseline` and `bench.validate_baselines` raised before
  they had classes, so no existing handler changes meaning.
- **`benchmarks/baselines` became a directory and `--baseline` accepts it.**
  The old single file still works when given by path, and a directory resolves
  by device name, so a workflow that names `benchmarks/baselines/apple-m5-pro.json`
  keeps the behaviour it had. `--update-baseline` given a directory names the
  file after the chip instead of writing the path it was given; given a path it
  behaves exactly as before.
- **The benchmark gate has one workload per model family and one for the
  reference backend.** A baseline recorded before 0.67.0 has no entries for
  them; `bench.compare` reports those as `no baseline entry, not compared`
  rather than failing, so an old baseline still gates what it covered. Re-record
  with `aoto bench --rounds 3 --update-baseline` to gate the new ones.
