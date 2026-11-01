# Usage

The README shows the two calls that matter. This page covers the rest of the
interface: tensor keyword arguments, prompt-length buckets, the `aoto` command,
artifacts, and the index check.

## Tensor keyword arguments

A tensor keyword argument becomes a graph input, so a caller can change it per
call instead of compiling it into the graph:

```python
position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"position_ids": position_ids},
)
logits = compiled(input_ids, position_ids=position_ids)[0]
```

Promotion is recursive, so a keyword that holds tensors inside a list, tuple or
mapping produces one graph input per tensor and the caller passes the same
nesting back:

```python
cos, sin = position_embeddings  # a pair of tensors, as newer HF layers take them
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"position_embeddings": (cos, sin)},
)
logits = compiled(input_ids, position_embeddings=(cos, sin))[0]
```

Non-tensor keyword arguments stay fixed at trace time. Leaves that are not
tensors — `use_cache=False`, or a float inside the same mapping — remain
tracing-time constants, and passing a different value for one raises instead of
being silently ignored. `past_key_values=[(k, v), ...]` promotes the same way.

A model compiled with `use_cache=True` is fine even though a plan is a single
forward pass: the cache starts empty, a zero-length concatenation operand
contributes nothing, and the frontend folds it away, so the graph is the one
`use_cache=False` produces. Cached decoding needs the static-cache adapter
below.

## Prompt-length buckets

A fixed prompt length means a serving loop recompiles for every new one.
`compile_qwen2_bucketed` compiles one plan pair per bucket width instead, and
serves any shorter prompt by left-padding it into the smallest bucket that fits:

```python
from aotolith.models.qwen2 import compile_qwen2_bucketed

runtime = compile_qwen2_bucketed(
    model,
    prompt_lengths=[128, 512, 2048],
    max_new_tokens=64,
)
tokens = runtime.greedy_generate(prompt_ids)  # 128, 512, or 2048
```

Left padding is what keeps the decode layout: the decode masks select the
trailing window of the cache, so the real tokens have to end there.
`is_causal=True` cannot tell a pad from a prompt token, though, so the prefill
graph takes an explicit rank-4 mask that carries causality and padding in one
tensor. At the full bucket width that mask is exactly the causal mask, so one
plan serves both the padded and the exact case.

Only the bucket widths vary a shape. The weights are shared by identity across
every bucket's plan, so a three-bucket runtime holds one copy of the model rather
than three. `save(prefix)` still writes one archive per bucket plan
(`prefix.128.prefill.aoto` and so on), and each archive carries its own copy of
the state.

## Shape buckets

The Qwen2 adapter has a bucket model the compiler can also apply to any module:
`compile_bucketed` compiles one plan per width along a chosen axis, and the
returned set dispatches a request of any width up to the largest bucket to the
smallest bucket that holds it.

```python
from aotolith import compile_bucketed

plans = compile_bucketed(
    model,                        # any nn.Module in eval mode
    (torch.randn(1, 16),),        # the shape it is traced at
    axis=0,                       # the axis that varies per request
    sizes=(1, 4, 8),              # the widths to compile
)
plans(mx.array(rows_3_by_16))     # 3 rows, served by the width-4 plan
```

Three things make that safe to serve from. Inputs whose extent on `axis` matches
the traced width are the request dimension and are re-widened per bucket; an
input of a different extent on the same axis (a table, a weight) is left alone,
and so is a tensor keyword whose extent did not follow the traced width.
Padding is *tiling* — the request's own rows repeat — rather than a constant
fill, so a reduction over the padded region cannot become NaN, and the padded
rows are sliced off the outputs. The last thing is the contract that makes it
correct: widening the axis must not change the values of the rows that were
already there. That holds for a batch axis, and it is why the sequence axis of a
causal block needs the caller's own padding:

```python
plans = compile_bucketed(
    model, (torch.randn(1, 8),), axis=1, sizes=(8, 32),
    pad=lambda inputs, width: (left_pad(inputs[0], width), mask_for(width)),
    unpad=lambda outputs, requested: (outputs[0][:, -requested:],),
)
```

Outputs are trimmed by shape: an output whose extent on the axis equals the
bucket width is treated as request-shaped and sliced back. A module that returns
a fixed-width tensor of exactly a bucket's width is the one case that needs the
`unpad` hook to say otherwise.

Dispatch is not bit-exact across widths, because the backend's matmul is not
row-exact either: this MLX build put the same row of a one-row and a four-row
`float32` matmul 6.8e-3 apart, since the tile shape decides the accumulation
order. The bucket set is within that noise and no more, which is what
`tests/test_buckets.py` asserts rather than a tolerance chosen by hand.

Ask before you admit a request:

```python
plans.widths                   # (1, 4, 8)
plans.footprint()              # state bytes, worst live set, worst bound
plans.check_memory_budget(budget_bytes)   # MemoryBudgetError (AOTO-2004)
plans.memory_report("minimal")            # the widest bucket's report
plans.stats.snapshot()         # requests per bucket, and the padding they cost
```

A request wider than every bucket raises `BucketRangeError` (`AOTO-2005`) rather
than compiling during the request: a compile costs hundreds of milliseconds and
cannot be bounded, which is the opposite of what an admission decision needs.

The set is also an artifact. `plans.save("model.aoto")` writes `model.1.aoto`,
`model.4.aoto`, and `model.buckets.json`; `load_bucketed` reads the index back,
loads each archive through the normal ABI and device checks, and re-shares the
weights. The index records each bucket's planned peak as a fingerprint, so an
artifact swapped in from another build is refused instead of dispatching a width
to a plan built for a different one. Custom `pad` hooks cannot be serialized, so
a set that uses them refuses to save.

## Model families

The static-cache adapter is one engine with a family attached, not one adapter
per model. `aotolith.models.decoder` holds the graph, the cache ABI and the
bucket dispatch; `aotolith.models.qwen2` and `aotolith.models.llama` are thin
frontends that pin a family and check the checkpoint against it.

| family | entry points | what it insists on |
| --- | --- | --- |
| `qwen2` | `compile_qwen2_static_cache`, `compile_qwen2_bucketed` | q/k/v/o projections carry a bias, which the packed projection kernel folds into its epilogue |
| `llama` | `compile_llama_static_cache`, `compile_llama_bucketed` | attention is bias-free; `num_key_value_heads` equals or divides `num_attention_heads`; RoPE base comes from `rope_theta`, and a set `rope_scaling` is refused by name |

```python
from aotolith.models.llama import compile_llama_static_cache

runtime = compile_llama_static_cache(
    model, prompt_length=128, max_new_tokens=64, quantize=64
)
tokens = runtime.greedy_generate(prompt_ids)
```

`quantize=` is the **group size** that shares a scale, and the width is a
separate knob: `quantize_bits=4` (the default) packs eight nibbles to a word and
`quantize_bits=8` packs four bytes. Both fold into the same streaming kernel, so
asking for eight bits is a precision choice rather than a different code path —
it costs about 12% of the decode rate over four bits and leaves roughly a
hundredth of the rounding error. Leaving `quantize=None` compiles the bf16
plans, and `quantize_bits` is refused in that case rather than ignored.

Everything after the checks is shared: the same cache layout, the same
`prefill_mask`, the same weight packing, the same `greedy_generate`. That is why
the refusal matters — `check_qwen2_model` and `check_llama_model` raise on the
other family's checkpoint (`UnsupportedModelError`, AOTO-2006) instead of
compiling a projection that is shaped for a different one. The family is read
from `model.config.model_type`, so a wrapper class is fine and a mismatched
`config` is not.

A family is also recorded where it can be read later: `runtime.family` names it,
and the benchmark gate compiles one prefill and one decode workload per family,
so a regression in the bias-free projection kernel is a failure rather than a
silent trade.

## Backends

`target=` chooses the producer of the plan, on both the compiler entry points and
the two adapters:

| target | what it emits | where it runs |
| --- | --- | --- |
| `apple-metal` (default) | Metal kernels per selected operation, MLX for the rest | any Apple GPU; kernels are rebuilt from MSL at load time |
| `mlx-reference` | no kernels at all — every node runs on the MLX interpreter | anywhere MLX runs, including a host with no Metal device |

```python
compiled = aoto.compile_module(model, (input_ids,), target=aoto.REFERENCE_TARGET)
assert aoto.TARGETS == ("apple-metal", "mlx-reference")
```
```bash
aoto compile ... --target mlx-reference
```

A reference plan is a real plan: same IR, same storage plan, same manifest, same
selections table — each entry just says `mlx`, which is exactly what a plan whose
kernels all fell back would record. Two things follow from that:

- **It is the seam made testable.** The compiler, the artifact format and the
  runtime take a plan without asking which backend made it. A reference artifact
  saves, loads and verifies through the same code path as the Metal one.
- **It is a second opinion, not a numerically tighter oracle.** MLX on Apple GPUs
  does not evaluate a batched `matmul` in IEEE float32: on an M5 with MLX 0.32 the
  same product differs from a float32 reference by about 1e-3 relative, while the
  single-row path is exact. The generated kernels accumulate in float32 and are
  the tighter of the two. Compare structure, shapes, dtypes, cache contents and
  token decisions between the targets; compare precision against the kernels.

Autotune is refused (`ValueError`) rather than ignored, because there is nothing
to measure when nothing is generated. `target=` on an adapter passes through to
every plan in the set, and a bucket index records it once, so a set compiled for
one target is refused as a set when it is loaded as the other — each archive
would refuse itself anyway, but the index is the place that can say so before
loading anything.

## Command line

The installed `aoto` command compiles a callable, module, or HuggingFace model.
It can also inspect or execute the result without importing the original model.

```bash
aoto compile \
  --module 'my_module:build_model' \
  --input-file inputs.json \
  --output model.aoto \
  --codegen-matmul \
  --autotune

aoto inspect model.aoto
aoto verify model.aoto --require-device-match
aoto run model.aoto --input-file inputs.json --output outputs.json
aoto --log-level INFO compile ...   # or AOTOLITH_LOG=INFO
```

`--target {apple-metal,mlx-reference}` picks the backend and is recorded in the
artifact; it combines with `--bucket-sizes`, whose index records it once.
`--codegen-matmul` and `--codegen-attention` are refused under
`--target mlx-reference`, because that backend has nothing to emit.

`--log-level` (and `AOTOLITH_LOG`) turns on the package log described under
[Logging](#logging); the flag goes before the command. Failures print as
`aoto: error: [AOTO-4001] …`, so the code is the first thing a wrapper can match
on and exit status is `1`. A wrong command line exits `2`.

`inputs.json` contains positional tensor specifications. Scalars use an empty
shape and a one-element data list:

```json
[
  {
    "shape": [1, 4],
    "dtype": "int64",
    "data": [[1, 2, 3, 4]]
  },
  {
    "shape": [],
    "dtype": "float32",
    "data": [1.0]
  }
]
```

Use repeated `--input '<json>'` arguments instead of `--input-file` when
convenient. `--callable MODULE:ATTR` traces a function, `--module MODULE:ATTR`
accepts an `nn.Module`, class, or factory, and `--hf-model` loads a HuggingFace
causal language model. `python -m aotolith` provides the same commands when the
console script is not on `PATH`.

A tensor keyword argument becomes a graph input, written here as `NAME=<json
spec>`: `aoto compile ... --forward-kwarg
position_ids='{"shape":[1,4],"dtype":"int64","data":[[0,1,2,3]]}'` and `aoto run
... --kwarg position_ids=...`. `aoto inspect` lists them under `entrypoint.kwargs`.

One compile can produce a bucket set instead of one artifact:

```bash
aoto compile --module 'my_module:build_model' \
  --input '{"shape":[1,16],"dtype":"float32","data":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}' \
  --output model.aoto --bucket-sizes 1,4,8
aoto inspect model.buckets.json
aoto run model.buckets.json --input '{"shape":[3,16],...}'   # served by the 4 bucket
```

It writes `model.1.aoto`, `model.4.aoto`, `model.8.aoto`, and the index
`model.buckets.json`; `inspect`, `verify`, and `run` accept the index wherever
they accept an artifact. `--bucket-axis` picks the axis (default `0`), and the
widths must be strictly increasing. A request wider than every bucket exits `1`
with `AOTO-2005`, so a wrapper can branch on the code instead of parsing text.

`aoto bench` is the regression gate for the compiler itself. It measures a fixed
set of shapes — a projection, a bucket dispatch, a gated MLP, an RMSNorm, causal
attention, an int4 GEMV, and the Qwen2 prefill and decode adapters — and can
compare the result against a recorded baseline:

```bash
aoto bench --baseline benchmarks/baselines --output perf-report.json
aoto bench --update-baseline benchmarks/baselines        # written as this chip's file
aoto bench --check-baselines benchmarks/baselines        # no GPU needed
```

`benchmarks/baselines` is a chip matrix, not a single file: each file is named
after the chip it records (`apple-m5-pro.json` for `Apple M5 Pro`). Pointing
`--baseline` at the directory picks this machine's file, so recording on a
second Mac adds a file instead of replacing the first. On a chip with no file
yet, the gate still compares op counts and the planned peak — those come from
the graph, not the GPU — and says which file it used and why the milliseconds
were skipped. `--check-baselines` validates the committed files without
measuring anything: schema version, a name that no longer matches the device it
records, a chip recorded twice. An explicit file path is still accepted and
used as given.

Its table has a row per workload: median, ratio against one raw MLX matmul,
first-call milliseconds, the peak memory the call added, the bound the memory
policy promises, the synchronizations that policy emits, compile time, op counts,
and native kernel count.

Which numbers it compares exactly and which it only compares on the same GPU is
in [the performance notes](performance.md#how-the-numbers-were-taken).
`aoto run --memory latency|balanced|minimal` picks the peak-memory policy for that
call; the same knob is `memory=` in Python, described under [Memory](#memory).

## Index bounds

PyTorch raises `IndexError` for out-of-range embedding and gather indices, while
MLX silently wraps or fills them, so the backend reproduces that check. The
values live on the device, so validating them costs a sync: checking every
operation up front cost six device synchronizations per decode step, and queuing
the checks and collapsing them into one reduction brought that to one.

A caller that already knows the indices are in range can drop that last one —
`plan(..., verify_index_bounds=False)`, or
`runtime.decode_token(..., verify_token=False)` — which is what the greedy loop
does, because every token it feeds back came from an `argmax` over the
vocabulary. The strict default is unchanged and still raises `IndexError`.

The exception is an `IndexBoundsError` (`AOTO-2002`), which is an `IndexError`
and names the offending value and the range that was accepted:

```
index out of range: 7 is outside the valid range [0, 5] for a dimension of size 6
```

Advanced indexing accepts negative indices, so its range is reported as
`[-6, 5]`; `F.embedding` does not, so its range is `[0, 5]`. The message follows
what the operation actually accepts rather than what a shape suggests.

## Errors

Every deliberate refusal is an `AotolithError` with a stable code, grouped by the
decision a caller makes: `CompilationError` for "this model cannot be compiled as
written", `PlanInputError` for "this call does not match the plan",
`DeviceError` for "this machine cannot run it", `ArtifactError` for a stored
file, `PlatformError` for an installation that cannot work here. Each class also
subclasses the builtin the previous release raised, so existing handlers keep
working:

```python
from aotolith import errors

try:
    compiled = aoto.compile_module(model, (input_ids,))
except errors.UnsupportedOperationError as exc:
    print(exc.code)   # AOTO-1001
    print(exc.hint)   # what to do about it
```

The full catalog, including which code belongs to which response, is
[docs/errors.md](errors.md).

## Logging

The package logs to `logging.getLogger("aotolith")` and installs only a
`NullHandler`, so importing it prints nothing. To see what a compile decided:

```python
import logging
logging.getLogger("aotolith").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
```

or, for a script, `aotolith.configure_logging("INFO")`. The `aoto` command takes
`--log-level`, and `AOTOLITH_LOG=DEBUG` does the same without a flag. `INFO` is
one line per compile (native-versus-MLX counts and the GPU), `DEBUG` is one line
per operation, and `WARNING` is every downgrade — tensor cores refused,
autotune with no winner, a stale artifact format.

## Calling a plan from several threads

A `CompiledPlan` is safe to call concurrently. The one lazy step — unrolling the
graph into a straight-line Python function on first call — is built under a lock,
so two threads arriving at a fresh plan build it once, and everything after that
is per-call state. The device is still a single stream in submission order;
overlapping GPU work means `mx.async_eval`, which the Qwen2 adapter uses, not
another thread.

## Memory

MLX is lazy. A plan call that only builds the graph leaves every intermediate
alive until the caller evaluates the outputs, and the evaluator then allocates an
output for every node before releasing the ones it is done with. On a chain of
eight concatenations whose live set peaks at 134 MB, the naive peak was 201 MB —
the sum of every intermediate, not the largest live set. The storage plan already
knew which values die at which operation; the runner had no way to act on it.

`memory=` is that way. Three policies:

| policy | what it does | peak |
| --- | --- | --- |
| `"latency"` | builds one graph, never synchronizes | the sum of every intermediate |
| `"balanced"` (**default** for `compile_module`, not for decoders) | evaluates and waits at release points once `max(8 MiB, transient/8)` of dead bytes has accumulated | planned peak + that threshold |
| `"minimal"` | evaluates and waits at every release point that frees anything | the planned peak |

A synchronization point is a real barrier, not just an `mx.eval`. Evaluating
schedules the work and returns, and a buffer the storage plan has finished with
cannot go back to the allocator while the GPU may still be writing it — so an
`mx.eval` alone left the next operator allocating on top of buffers that were
only nominally dead. Measured across twelve fresh processes, that made the same
134 MB plan peak at 134 MB, at 160 MB, or at 134 MB + 256 B. The runner now
evaluates, waits, and then deletes, and all twelve processes measured exactly
`peak_bound_bytes`.

An earlier revision of this section said the default costs nothing on the
workloads this repository ships. That was wrong, and the two LLM plans are where
it breaks: a 0.5B model's prefill holds 254 MiB of transient and synchronizes
seven times under `"balanced"`, and a decode step holds 8.46 MiB and synchronizes
once, just over the 8 MiB floor. One barrier per step is enough to cost a fifth
of the step rate, because it flushes the pipeline the caller is using to overlap
the next step's host-side build. Measured in one session on Qwen2.5-0.5B bf16
with `codegen_matmul`, `"latency"` decodes at 240.9 tok/s and `"balanced"` at 196,
and prefill reads 11 660 against 8 707 tok/s.

So the decoder runtimes in `aotolith.models` default to `"latency"`:
they exist to be overlapped, and their peak is weights plus KV cache rather than
dead intermediates, which is not the case the policy was built for.
`compile_module` keeps `"balanced"`. Pass `memory="balanced"` to a decoder when
the bound matters more than the rate — `memory_footprint()` and
`check_memory_budget()` read shapes rather than this policy, so budgeting works
under either. The policy's benefit is still real where the trade is worth it: on
the 134 MB chain above it held the peak with four synchronizations, where
`"minimal"` needed nine for the same peak.

```python
compiled.plan(inputs, memory="balanced")   # or "latency" / "minimal"
compiled(inputs, memory="minimal")          # the module entry point takes it too

compile_qwen2_static_cache(model, prompt_length=128, max_new_tokens=64,
                           memory="balanced")   # decoders default to "latency"
```

Ask before you run:

```python
report = compiled.plan.memory_report("balanced")
report.planned_peak_bytes   # every live byte at the busiest operation
report.peak_bound_bytes     # the most it can hold: planned + the threshold
report.sync_count           # synchronizations the unrolled body will emit
report.transient_bytes      # bytes the plan creates and may give back
```

`peak_bound_bytes` is an upper bound, and it is enforced in CI by `aoto bench`
(a per-workload column) and by `tests/test_memory.py`, which measures the peak in
a subprocess where the allocator is clean.

Separately, `plan.footprint` splits the plan's memory by who owns it — the
inputs the caller keeps resident (weights), the activations the call allocates
and gives back, the outputs, and the arena the storage plan would need:

```python
print(compiled.plan.footprint.describe())
# inputs 16.0 MiB, activations 256.0 KiB, outputs 128.0 KiB, peak 16.2 MiB
```

A server that admits requests uses those numbers to decide before allocating.
For the static-cache runtime there is a budget check as well:

```python
runtime.check_memory_budget(budget_bytes)   # raises MemoryBudgetError (AOTO-2004)
```

It compares the budget against one decode step, which holds two caches: a step
allocates the cache it returns before the one it was given is dropped. That
doubling is why a bucket that "fits" can still fail partway through a sequence.
`runtime.memory_footprint()` reports the split — weights, the packed KV cache,
the prefill mask, one step's scratch, and the step peak.

### Cold start

The first call of a plan pays for unrolling the graph into a Python function and
for Metal compiling the kernels it uses. `compiled.prepare()` (or
`plan.prepare()`) does the unrolling at startup, so the first request pays only
for the kernels:

```python
compiled.prepare()   # builds the runner; kernels still compile on first use
```

`aoto bench` reports both parts: `first_call_ms` is the first call including
kernel compilation, `compile_ms` is the compile. Both are gated against the
recorded baseline with a factor-of-two allowance, because Metal's shader compiler
varied by 1.5x between runs of unchanged code.

## Artifacts

Artifacts use format version 5 and ABI version 1.3. Format 4 still loads, because
the reader always accepted its kernel encoding and one manifest check was
rejecting it; loading a format 4 file logs a warning. A newer format is refused.
Each graph, kernel table, selection table, storage plan, and state tensor has a
SHA-256 digest. Loading validates the ABI, target, graph, entrypoint layout, state
shape/dtype, serialized kernels, rebuilt selections, and storage plan. Kernel
objects are rebuilt from the validated graph and persisted selection, so the
artifact does not need the original Python model and loading does not repeat
autotuning.

Strict GPU matching is optional. `--require-device-match` or
`require_device_match=True` rejects an artifact compiled for another Apple GPU
model; the default allows execution on another Apple Silicon device after
rebuilding kernels.

## Compile a callable

The example in the README compiles a small `addmm + SiLU` function. `compile_module`
traces a model with its parameters and buffers as explicit state, and the
returned plan does not retain the original `nn.Module`:

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import aotolith as aoto

model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
).eval()

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
compiled = aoto.compile_module(
    model,
    (input_ids,),
    codegen_matmul=True,
    codegen_attention=True,
)
logits = compiled(input_ids)[0]
mx.eval(logits)
```
