# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The policy that decides what counts as a breaking change lives in
[`docs/compatibility.md`](docs/compatibility.md).

## [Unreleased]

### Added

- **`examples/qwen_logit_distance.py`**, which measures each execution path's
  last-token logits against PyTorch for several prompt lengths. It exists because
  the figure it replaces was a single measurement quoted as if it described the
  model: the MLX fallback's mean distance moves by 3.4x between a 7-token and a
  16-token prompt, so the prompt length has to be part of the claim.

### Fixed

- **The README and `docs/performance.md` quoted 0.043 / 0.38 against 0.081 /
  1.5** for how far the native and fallback paths land from PyTorch's logits.
  Neither pair reproduces at 0.68.0, and the commit that added them recorded a
  7-token prompt in its message that never reached the prose. Two claims were
  wrong, not just the numbers: the fallback's distance is not a property of the
  model (it depends on the prompt length by 3.4x), and "the native path is
  closer" overstates a gap that is 5% at a 128-token prefill and reverses across
  32 decode steps. Both languages now carry the measured table and the narrower
  conclusion that code generation does not cost accuracy.

## [0.68.0] - 2026-11-01

A gate that scales past one machine. The performance baseline was a single file
recorded on whichever Mac ran last, which meant a second Mac either overwrote it
or was compared against numbers it could never match; the matrix the gap analysis
asked for - M1 through M4, one file each - needs the file to be the chip's
identity before any of it can work.

### Added

- **`benchmarks/baselines` is a chip matrix.** One file per chip, named by
  `bench.device_slug` of the device name the report records (`Apple M5 Pro` ->
  `apple-m5-pro.json`). `aoto bench --baseline DIR` resolves to this machine's
  file; `aoto bench --update-baseline DIR` writes this chip's file instead of
  replacing a neighbour's. A chip with no file of its own still gates
  `input_ops`, `lowered_ops`, `planned_peak_bytes` and `peak_bound_bytes`, which
  come from the graph and not the GPU, and prints a note naming the file it used
  and why the milliseconds were skipped. Silent passes are the failure mode a
  gate cannot have, so an empty directory is an error rather than a pass.
- **`aoto bench --check-baselines DIR`**, which validates the committed files
  without measuring anything: schema version, a file renamed away from the chip
  it records (the loader would never select it), a chip recorded twice, a report
  with no repeats (its floor cannot be compared). This runs on any macOS runner.
- **`BenchmarkError` (`AOTO-6000`) and `BaselineError` (`AOTO-6001`)**, a sixth
  error area for failures that are neither the model's, the call's, the device's,
  the artifact's nor the installation's - the gate's own inputs. Both subclass
  `ValueError`, so every `except ValueError` written against the gate keeps
  working. `docs/errors.md` and `docs/errors.zh-CN.md` name the area.
- **`real-device-gate` CI job**, which runs the strict timing half on a
  self-hosted M-series runner and is enabled by the repository variable
  `AOTOLITH_SELF_HOSTED_RUNNER=true`. The `bench` job now validates the matrix
  and gates through the directory, so the same workflow picks up a newly
  recorded chip with no edit.

### Changed

- A baseline file that is named for one chip but records another is no longer
  trusted: `resolve_baseline` treats it as "no file for this chip" and says so.
  Copying a baseline and renaming it is how an M4 ends up gated against M5
  milliseconds.
- `compare()`'s note on a device mismatch names both devices and the file, so a
  reader can tell "not recorded here" from "recorded and slower".

### Documented

- **The MLX fallback's precision.** With both codegens on, a two-layer Llama
  lands `1.043e-07` from PyTorch; on the MLX fallback the same graph lands
  `1.3e-3`, because MLX's own float32 matmul is `8e-4` relative to float64 and
  flat in K - a reduced-precision matmul rather than an accumulation error. The
  fallback is correctness-safe (greedy tokens match) but not bit-accurate, which
  decides how a logit-pinning test has to be written. In `docs/performance.md`
  and its translation.

### Fixed

- **`examples/tiny_llama_compile.py` exists.** Both READMEs and a CI step ran
  it, and the file was never written, so that CI step could only fail. It now
  compiles a two-layer `LlamaForCausalLM`, compares last-token logits and the
  greedy continuation against PyTorch, and checks that the plan still runs after
  the module is released. The README's numbers were not reproducible either
  ("195 FX operations to 108 lowered"); they are now the ones the script prints
  (131 -> 79 in prefill, 151 -> 62 in decode, `1.043e-07` drift).
- **A test that every `python examples/...` in the docs or the workflows is a
  file in the tree**, so the class of failure above is caught by the suite and
  not by a red CI step.
- `docs/architecture.md` described the reference matmul as measured "before and
  after each workload", which was the behaviour two releases ago. It is sampled
  between samples, and the section now says why (the 0.177 -> 0.237 drift the
  interleaving removed).
- Section numbering: `13.5` and `13.6` were out of order after the bucket
  section was inserted; buckets is `13.5`, the gate `13.6`, the matrix `13.7`.

## [0.67.0] - 2026-11-01

A scope release for the two boundaries the compiler had only claimed. The
static-cache adapter was written for one model family and the backend was one
class; both are now interfaces, each with a second implementation that proves the
seam is where the design said it was.

### Added

- **`aotolith.models.llama`**, the second decoder family, with
  `compile_llama_static_cache` and `compile_llama_bucketed`. It shares one engine
  with Qwen2 - `aotolith.models.decoder` holds the graph, the cache ABI, the int4
  packing and the greedy loop - so what makes it a family is what it insists on:
  Llama's q/k/v/o projections carry no bias, `num_key_value_heads` may equal
  `num_attention_heads` as well as divide it, and the rotary table comes from
  `rope_theta` rather than a constant. A set `rope_scaling`, a sliding-window
  config, or a hand-built attention with a bias on some of q/k/v and not others
  is refused by name.
- **`FAMILIES`, `family_for` and the pinning rule.** A family is a validated
  `model_type` set (`qwen2`, `llama`), not a class hierarchy, and each frontend
  refuses the other's checkpoint: compiling a Qwen2 layout through the Llama
  entry point would drop or invent a bias term that the logits only show in the
  last decimals.
- **`target="mlx-reference"`**, a second backend that emits nothing. Every node
  runs on the MLX interpreter, every selection says `mlx`, and the kernel table
  is empty - which is exactly what a plan whose kernels all fell back would
  record. It is a portable plan (no metallib, no per-device kernel compile), a
  structural check on the same lowered graph the Metal plan runs, and the proof
  that the compiler, the artifact format and the runtime take a plan without
  asking who produced it.
- **`Backend` is a protocol** - `compile(graph, *, tune_inputs=None)` - with
  `AppleMetalBackend` and `ReferenceBackend` behind it, and `TARGETS` naming
  both. `target=`, and `aoto compile --target`, reach it from every entry point,
  including both adapters.
- **The target is in the artifact.** `manifest["target"]` is written and checked
  on load, so a reference artifact cannot be loaded as a Metal one, and a bucket
  index records it once so a mixed set is refused before any archive is opened.
  An artifact or index without the field reads as `apple-metal`, which is what it
  was.
- **`REFERENCE_TARGET` and `TARGETS` are exported from the package root**, so a
  host that passes `target=` as a value can check the name instead of catching
  the compiler's error.
- **`runtime.family`** names the decoder family a plan was compiled from, on both
  the single-bucket and bucketed runtimes.
- **MLX fallbacks for the normalization family**: `layer_norm`, `rms_norm`,
  `softmax`, `safe_softmax`, `sum` and `mean` previously existed only as
  generated Metal kernels; a reference plan needs them to have an interpreter
  path, and having one is also what lets autotune compare a kernel against MLX
  for those ops.
- **`UnsupportedModelError` (AOTO-1004)** gives "this checkout cannot be
  compiled" its own class, separate from "this operator has no lowering". It is
  also a `NotImplementedError`, so handlers written against the previous release
  keep working. The catalog in `docs/errors.md` names it.
- **Two adapter workloads and one reference workload in `aoto bench`**:
  `llama_prefill_128`, `llama_decode_128`, and `reference_linear_1024x1024`. The
  adapter pairs make each family a gated object rather than a claim; the
  reference one gates the interpreter path, whose op counts and memory bound are
  the same numbers as the Metal plan's and whose kernel count is zero by
  construction.

### Changed

- `models/qwen2.py` is now a frontend over `models/decoder.py` (1128 lines became
  a few dozen). Its public names are aliases of the decoder classes, so every
  import keeps working; what changed is `type(...).__name__`.
- `prefill_mask` splits its refusal: a non-positive length is a `PlanInputError`,
  a length wider than the bucket is a `BucketRangeError` (AOTO-2005). Runtime
  argument checks across the adapter moved from bare `ValueError` to
  `PlanInputError`, and compile-time option checks to `CompilationError`, so a
  caller can branch on the class instead of the message.
- `models.decoder` sits above `buckets.py` in the layer order, because it calls
  `share_plan_state`; the three frontends sit above it. The architecture test
  gained those edges.
- `benchmarks/baselines/apple-m5-pro.json` has the three new workloads.

### Fixed

- **The gate's reference matmul was timed beside the workload, not inside it.**
  A floor divides one sample by a reference sample, and the two were taken at
  different moments: one round measured the same 0.2 ms attention kernel at
  0.203 ms against a 1.15 ms reference, the next at 0.203 ms against 0.78 ms.
  The first pair recorded a `min_ratio` of 0.177 that no later run could reach,
  so unchanged code failed the gate with a 1.35x "regression" that was the
  reference clocking up on its own. The reference is now sampled between a
  workload's samples, so both sides of the division share a window; four
  consecutive runs put the same workload at 0.2425-0.2543.
- `autotune` on the reference target raises instead of silently losing the
  measurements a caller expected, and `target=` on a bucket set is recorded once
  in the index, so a set compiled for one target is refused as a set when it is
  loaded as the other.

## [0.66.0] - 2026-11-01

A scope release for the one thing that made a serving loop recompile. The IR is
still static - a symbolic extent would put the storage plan, the peak bound and
the kernel selection back at run time - so the variation is bounded from the
other side instead: a set of compiled widths, and a dispatcher that serves any
width up to the largest. The Qwen2 adapter had this; now any module does.

### Added

- **`compile_bucketed(module, example_inputs, axis=..., sizes=...)`** compiles
  one plan per width along an axis and returns a `BucketedPlans` that dispatches
  a request to the smallest bucket that holds it. Inputs are widened per bucket
  when their extent on the axis matched the traced width, including tensor
  keyword arguments, so a plan traced at width 4 really expects a width-4 mask;
  an input of a different extent on the same axis - a table, a weight - is left
  alone.
- **Padding is tiling, not a constant fill.** The padded region repeats the
  request's own slices, so a `var` or `softmax` over it stays finite, and the
  padded slices are trimmed off the outputs. The contract the default relies on
  is that widening the axis does not change the rows that were already there,
  which holds for a batch axis and not for the sequence axis of a causal block;
  that case takes `pad`/`unpad` hooks, so the caller owns the mask and the
  compiler owns the dispatch.
- **`BucketedPlans.footprint()`, `check_memory_budget()` and `memory_report()`**
  account for a set rather than a plan: one copy of the shared state (by
  identity), the worst bucket's activations, and the worst bucket's peak bound,
  which is the number an admission decision needs. Over budget raises
  `MemoryBudgetError` (AOTO-2004) before a request runs.
- **`BucketRangeError` (AOTO-2005)** for a request wider than every bucket. It is
  a `PlanInputError`, so existing handlers keep working, and the separate class
  is what lets a serving layer answer "too long" instead of "bad call".
- **Bucket sets are artifacts.** `save(prefix)` writes one archive per bucket
  plus `prefix.buckets.json`, and `load_bucketed` reads the index back, loads
  each archive through the normal ABI and device checks, and re-shares the
  weights. The index records each bucket's planned peak as a fingerprint, so a
  swapped-in artifact is refused with `ArtifactCompatibilityError` instead of
  dispatching a width to a plan compiled for another one. A set with custom
  `pad` hooks refuses to save, because the hooks are host code.
- **`aoto compile --bucket-sizes 1,4,8 [--bucket-axis N]`**, and `inspect`,
  `verify`, and `run` accept the index wherever they accept an artifact. A
  request past the widest bucket exits 1 with `AOTO-2005`.
- **`BucketStats`** counts requests per bucket and the padding they cost, and
  every non-exact dispatch logs its bucket, request width, and padding share at
  `DEBUG`. `share_plan_state` is now public; the Qwen2 adapter calls it too.
- **A `bucket_dispatch_96_into_128` workload in `aoto bench`**, so the tile, the
  trim and the per-bucket lowering are behind the gate rather than only the
  matmul inside one plan. A bucket set reports the sum of its buckets' op and
  kernel counts (an extra one anywhere is a regression) and the widest bucket's
  memory bound.

### Changed

- `docs/performance.md` measures what a bucket boundary costs, and the answer is
  not one number: an exact hit through a set is inside the noise of calling the
  plan directly (0.95-1.03x), 96 rows served by a 128 bucket on a 1024-wide
  projection is 0.87-1.06x because the weight read dominates, and 4096 rows
  served by 8192 is 2.16-2.18x because past a few thousand rows the projection is
  activation-bound. Dispatch is not bit-exact across widths, and neither is the
  backend: this MLX build put one row of a 1-row and a 4-row `float32` matmul
  6.8e-3 apart, and the bucket set is asserted against that measured noise rather
  than a hand-picked tolerance.

## [0.65.0] - 2026-11-01

A correctness release for the promises the last one made. The memory policies
now hold the peak they report, and the benchmark gate compares a number that
does not move when the machine does.

### Fixed

- **A memory policy's release was not a release.** `mx.eval` schedules work and
  returns, so a buffer the storage plan had finished with was still in flight
  when the next operator allocated. The measured peak therefore exceeded
  `peak_bound_bytes` at random: the same 134 MB plan measured 134 MB, 160 MB, or
  134 MB + 256 B across twelve fresh processes. Synchronization points now
  evaluate, `mx.synchronize()`, and only then delete, which makes the planned
  peak the physical one - twelve of twelve fresh processes measured exactly
  `peak_bound_bytes`, and the shipped workloads still emit zero synchronizations
  under the default policy.

### Changed

- **The benchmark gate compares `min_ratio`**, a workload's best sample divided
  by a reference matmul measured at the same moment, instead of `min_ms`. The
  reference is taken around every workload rather than once per run: on a
  machine whose speed drifts, one global reference was enough to report a 60%
  projection "regression" on unchanged code. `min_ms` slower than the baseline
  with `min_ratio` unchanged is now a note, not a failure.
- **`aoto bench --rounds N`** repeats the timed half and keeps every sample. The
  default tolerance is 25%, above the 15% swing two best-of-33 floors showed on
  the machine the baseline came from; the exact half of the gate (op counts,
  planned peak, memory bound) still has none.
- `first_call_ms` and `compile_ms` get a factor of three instead of two: both are
  one-shot host measurements dominated by Metal's shader compiler, and one
  loaded run put the first decode call at 14.2 ms against a baseline of 5.1 ms
  while every timing in the same run moved by less than 2%.
- `benchmarks/baselines/apple-m5-pro.json` re-recorded with `--rounds 3`, the
  same flag CI now uses.

## [0.64.0] - 2026-11-01

A memory release. It adds a knob where a plan previously had none, makes the
storage plan's releases real, and turns peak memory and cold start into numbers
the benchmark gate can fail on.

### Added

- **`memory=` on every plan call**, with three policies. `"latency"` builds one
  lazy graph and never synchronizes; `"balanced"` (the default) synchronizes at
  storage-plan release points once `max(8 MiB, transient/8)` of dead bytes has
  accumulated; `"minimal"` synchronizes at every release point. On a chain whose
  live set peaks at 134 MB, `latency` measured 201 MB, `balanced` 134.5 MB with
  four synchronizations and `minimal` 134.5 MB with nine. The default costs
  nothing on the shapes this repository ships — all of them hold less than 8 MiB
  of transient, so it emits no synchronization at all.
- **`plan.memory_report(policy)`** returns the planned peak, the transient
  traffic, the synchronization count, and `peak_bound_bytes`, the most the runner
  can hold. `plan.prepare()` (and `compiled.prepare()`) unrolls the runner at
  startup so the first request pays only for kernel compilation.
- **`plan.footprint`** splits a plan's memory into the inputs the caller keeps
  resident, the activations the call allocates and gives back, the outputs, and
  the arena the storage plan would need.
- **`Qwen2StaticCacheRuntime.memory_footprint()`** reports weights, the packed KV
  cache, the prefill mask, one step's scratch, and the step peak (which holds two
  caches, because a step allocates the cache it returns before dropping the one
  it was given). **`check_memory_budget(budget_bytes)`** refuses a bucket before
  it runs, with the new `MemoryBudgetError` (`AOTO-2004`).
- **`aoto bench` records peak memory and cold start**: `planned_peak_bytes` and
  `peak_bound_bytes` (deterministic, gated exactly on any machine),
  `peak_extra_bytes` and `first_call_ms` (measured, gated on the baseline's own
  device). Report schema is now 2; the recorded baseline is re-recorded.
- **`aoto run --memory`** exposes the policy on the command line.

### Changed

- **The unrolled runner releases intermediates.** It evaluates the values still
  live at a release point and deletes the names that died there. Without the
  synchronization, deleting a name frees nothing: the node is still in the lazy
  graph, and `mx.async_eval` allocates the buffer when it encodes the command.
  Measurements for both are in `docs/performance.md`.
- **`compile_qwen2_static_cache` shares weights between its prefill and decode
  plans.** Only `compile_qwen2_bucketed` shared them before, so a single bucket
  held two copies of the checkpoint.
- **The decode mask is built per step** instead of indexed out of a
  `capacity x capacity` boolean table: 16 MiB at a 4096-token capacity and 1 GiB
  at 32k, to save a few microseconds. `Qwen2StaticCacheRuntime` no longer carries
  a `_decode_masks` array; `decode_mask(valid_length)` replaces it and returns
  the same window.
- **`tensor_nbytes()` is public** in `aotolith.planning`, and `DTYPE_ITEMSIZE`
  moved to `aotolith.ir.types` so the storage plan, the runtime bounds and the
  memory accounting size a tensor the same way.

### Notes

- No artifact incompatibility: the format version, the ABI, and every number the
  compiler computes for a graph are unchanged. `memory=` is an execution policy,
  not part of the artifact.
- The benchmark report schema changed, so a baseline recorded by 0.63.0 is
  refused with "re-record the baseline"; `aoto bench --update-baseline` does it.

## [0.63.0] - 2026-11-01

A contract release. Nothing here changes a number the compiler computes; what it
changes is what a caller can rely on when it fails, when it is slow, and when it
runs on a machine that is not the one it was compiled on.

### Added

- **An error hierarchy with stable codes.** Every deliberate refusal is now an
  `AotolithError` with a `code` and an optional `hint`. The classes are grouped
  by the decision a caller makes — `CompilationError` (`AOTO-1xxx`),
  `PlanInputError` (`AOTO-2xxx`), `DeviceError` (`AOTO-3xxx`), `ArtifactError`
  (`AOTO-4xxx`), `PlatformError` (`AOTO-5xxx`) — and each also subclasses the
  builtin the previous release raised, so existing `except ValueError` /
  `except NotImplementedError` / `except IndexError` handlers keep working.
  Catalog: [`docs/errors.md`](docs/errors.md), [`docs/errors.zh-CN.md`](docs/errors.zh-CN.md).
- **`format_error()`** renders `[code] message` plus a `hint:` line, and the
  `aoto` command prints failures that way.
- **Package logging** on `logging.getLogger("aotolith")`, with a `NullHandler`
  and no other handler: the compile summary at `INFO`, per-operation decisions at
  `DEBUG`, and every downgrade (tensor cores refused, autotune with no winner,
  an older artifact format) at `WARNING`. `aotolith.configure_logging()` and
  `aoto --log-level` / `AOTOLITH_LOG` are the opt-ins.
- **A platform guard.** `import aotolith` on a non-arm64-darwin host now raises
  `PlatformError` naming the host instead of failing later inside MLX, and the
  `mlx` dependency carries a PEP 508 marker so the requirement is visible in the
  metadata rather than only at resolution time.
- **Thread-safety contract for `CompiledPlan`.** The lazy unrolled runner is
  built under a lock, so two threads arriving at a fresh plan build it once.
  Concurrent calls are supported and documented on the class.
- **Readable indices for out-of-range plans.** `IndexBoundsError` reports the
  offending value and the range that was accepted (`[0, 5]`, or `[-6, 5]` where
  negative indices are legal) instead of "index out of range in self".

### Changed

- **Format 4 artifacts load again.** The reader had always accepted format 4's
  singular `output_id`/`tensor_type` kernel encoding, but a strict
  `format_version == 5` check in the manifest rejected the file before it got
  there. Format 4 is now an accepted compatibility path, logged at `WARNING` so
  a deployment notices it is reading a stale artifact. A future format is still
  refused, and a damaged payload is still refused.
- **`type: ignore`-free type checking of the public surface.** `pyright` runs in
  CI over the entry points, errors, artifact, CLI and plan with
  `pyrightconfig.json`; the numerical kernels are excluded for now and the
  config says so.
- **Version is single-sourced.** `pyproject.toml` reads the version from
  `src/aotolith/_version.py` via `hatch`, so the module and the metadata cannot
  disagree.
- **Coverage floor in CI.** `[tool.coverage.report]` sets `fail_under = 84` and
  CI enforces it, so coverage ratchets up instead of drifting down.
- **Python 3.13 and 3.14 in the CI matrix**, alongside 3.11 and 3.12 — both
  interpreters MLX 0.32 and Torch 2.15 ship wheels for.
- `pyproject.toml` declares `coverage` and `pyright` in the `dev` extra, so
  `pip install -e ".[dev]"` is enough to run every gate CI runs.

### Notes

- No artifact incompatibility: 0.63.0 writes format 5 / ABI 1.3 exactly as
  0.62.0 did, and reads what 0.62.0 wrote.
- `aotolith.models.qwen2` still compiles one fixed prompt length and one fixed
  KV-cache size per plan. Prompt buckets exist; true dynamic shapes do not.

[Unreleased]: https://github.com/jaykay233/aotolith/compare/v0.67.0...HEAD
[0.68.0]: https://github.com/jaykay233/aotolith/compare/v0.67.0...v0.68.0
[0.67.0]: https://github.com/jaykay233/aotolith/compare/v0.66.0...v0.67.0
[0.66.0]: https://github.com/jaykay233/aotolith/compare/v0.65.0...v0.66.0
[0.65.0]: https://github.com/jaykay233/aotolith/compare/v0.64.0...v0.65.0
[0.64.0]: https://github.com/jaykay233/aotolith/compare/v0.63.0...v0.64.0
[0.63.0]: https://github.com/jaykay233/aotolith/compare/v0.62.0...v0.63.0
