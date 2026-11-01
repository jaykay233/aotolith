# Host runtime

**Status: proposal, with one gap marked as a blocker.** The compiler side of this
page exists and works; the token loop does not. There is no command that accepts a
text prompt, no stop condition, and no sampler, so "run a model end to end" is
still something only `examples/qwen_static_cache_benchmark.py` does. That is
[item T2](#adoption-path) and it comes before everything else on the Roadmap.

Aotolith compiles a model into static plans. Something has to decide which plan a
request runs on and what happens to the tokens around it. This page records where
that line falls, what the compiler promises across it, and the order the pieces
should arrive in.

## The decision

**End-to-end usability is mandatory, and it is not the same thing as an engine.**
Those two get conflated, and the conflation is why this page exists. The token
loop, the sampler, the stop condition and a text-in/text-out command are cheap,
they are what a user means by "does this work", and the project does not have
them. Request scheduling, continuous batching, paged KV and a serving API are
expensive, they are a product, and they are separable.

So the line is drawn at the token loop, not at `src/`:

| | in aotolith | in the embedding engine |
| --- | --- | --- |
| Text in, tokens out, one request | **yes** | — |
| Sampling, stop conditions, streaming | **yes**, host-side over logits | — |
| Which bucket a request uses | **yes**, one request picks one | — |
| Concurrency above one | — | queue, admission, cache slots |
| Continuous batching, preemption | — | needs a compiler change first |
| Paged KV, prefix cache | — | — |
| Serving API, multi-model residency | — | — |

The left column is the reason someone tries the project at all. The right column
is a second product whose success criteria are model breadth, quantization
formats and behaviour under concurrent load — none of which the compiler is
measured on.

The module docstring already draws the same line, in the same place:

> The host owns the greedy loop and the one-time left padding from prompt length
> to cache capacity. Scheduling, paged allocation, batching, and serving remain
> outside Aotolith.
>
> — the module docstring of `models/qwen2.py`

## What end-to-end actually needs

Not a scheduler. Five things, and they are all small. Measured on Qwen2.5-0.5B,
`prompt_length=128`, `max_new_tokens=64`:

| gap | today | size |
| --- | --- | --- |
| **Text in / text out** | `aoto run` takes tensor specs as JSON and returns JSON arrays. There is no command that accepts a prompt. | one CLI command |
| **Stop condition** | `_greedy_tokens` loops `range(steps - 1)` and never reads an EOS token, so every call generates exactly `max_new_tokens`. Chat is not expressible. | a few lines |
| **Sampling** | greedy only. Plans hold no RNG state by design, and random draws are refused at capture, so the sampler has to be host-side over logits. | ~50 lines |
| **Streaming** | `mx.async_eval` is issued per step but the loop batches all tokens and syncs once at the end. | ~30 lines, with a cost, below |
| **Tokenizer / chat template** | only in `examples/`, via the `hf` extra. | wiring |

Two things that look like blockers and are not:

**Compile latency.** A full 0.5B compile is 3.4 s bf16 and 7.5 s with int4, for a
683-op decode plan and 146 kernels. That is a startup cost, not a serving-path
cost, and it does not argue for an offline artifact step.

**The artifact round trip.** `save_compiled` and `load_compiled_module` already
work, so "compile on a build machine, ship a `.aoto`, load it in the host" is
available whenever it is wanted. It is an optimisation, not a requirement.

One real tension, worth stating because it is not obvious:

**Streaming costs throughput.** Decode reads 232 tok/s asynchronous against 167
synchronous, and emitting a token to the caller is a sync — so streaming per
token gives back about 28% of decode. That is inherent to a host that must hand
over each token, not a defect in the compiler. It also means a "streaming" mode
has to be chosen, not assumed, and that a multi-token step (below) is the way to
get some of it back: emitting a verified chunk of *m* tokens costs one sync per
*m* rows instead of one per row.

## Why the engine beyond that is still the wrong product

**Static shapes and continuous batching are opposite requirements.** Every
dimension in a plan is a compile-time constant: batch size, sequence length, cache
capacity. A continuous-batching scheduler changes the batch composition on every
step, so the alternatives are recompiling per step or compiling for every
composition it might see. Neither is a scheduling problem that can be worked
around at the serving layer. It is a compiler problem, and it has to be solved
first, in that order.

**The comparison moves to a surface the project does not lead on.** The 1.21x on
decode is batch 1, one request, asynchronous submission, 128 prompt tokens plus 64
generated, against `mlx_lm` in the same process. An engine is compared on model
breadth, quantization formats, behavior under concurrent load, and API
compatibility. Under real arrival rates the scheduler takes back a large share of
that 1.21x, and the features that would replace it do not exist yet.

**The expensive tail is genuinely out of scope.** Multi-model residency and an
LRU that does not thrash unified memory. Paged KV, prefix cache, preemption.
Cancellation and timeouts under load. Concurrency limits and error semantics.
Those share no test with the compiler's regression gate, and each one spends
budget from symbolic shapes, shape-aware policy and native coverage.

**The boundary erodes quietly.** The Roadmap's core items are about compiler
quality: symbolic shapes, shape-aware backend policy, and wider native coverage.
Every serving feature that lands will look like a small addition at the time, and
each one spends budget from those three. That is the argument for drawing the line
at the token loop and holding it there — not for leaving the token loop
unimplemented.

## What the compiler already gives a host

These are the guarantees an embedding host may rely on today.

- **A plan pair per prompt-length bucket.** `compile_qwen2_bucketed` produces one
  prefill plan and one decode plan per bucket width. A shorter prompt is
  left-padded into the smallest bucket that holds it, and the padded prefill
  produces the same greedy tokens as a plan compiled for the exact length.
- **One copy of the weights per bucket set.** `_share_plan_state` rebinds state
  entries with the same `(name, shape, dtype)` to one array, so a bucket list does
  not multiply resident weights — and since this release the single-bucket entry
  point shares them between its prefill and decode plan too, which used to leave
  two copies of the checkpoint.
- **Plans that reload in a fresh process.** `save_compiled` and
  `load_compiled_module` round-trip the graph, kernels, selections, storage plan
  and state (format 5, ABI 1.3), and the loader verifies every digest before a
  kernel is rebuilt.
- **Thread safety at the host level, not the device level.** A plan may be called
  from several threads; first-call unrolling is guarded, and index checks live in
  a `ContextVar`. MLX enqueues on one stream per device, so the GPU runs work in
  submission order. Overlap comes from `mx.async_eval`, which the Qwen2 adapter
  already uses, not from another thread.
- **Optional index checks.** `verify_index_bounds=True` is the strict default and
  costs a sync per call. The decode adapter turns it off for tokens it fed from an
  `argmax`, where the index cannot be out of range.
- **Inspectable choices.** `plan.selections` records, per operator, whether it ran
  as a generated Metal kernel or on MLX, with the reason.
- **Memory you can budget before allocating.** `plan.footprint` splits a plan's
  bytes into the inputs the caller keeps resident and the activations the call
  gives back; `plan.memory_report(policy)` states the peak a call can reach and
  `memory=` bounds it. For the static-cache runtime,
  `runtime.memory_footprint()` reports weights, KV cache, prefill mask and one
  step's scratch, and `runtime.check_memory_budget(budget_bytes)` refuses a
  bucket that cannot fit before it runs (`MemoryBudgetError`, `AOTO-2004`). That
  check is an admission decision a host can make in microseconds instead of
  discovering the same thing from a failed allocation mid-sequence.

What that buys, on Qwen2.5-0.5B-Instruct at batch 1: a decode step of 634
operators after the cache and normalization folds, 991 MB of weights streamed per
token, and 232 tok/s asynchronous against `mlx_lm`'s 195 in the same process.
Weight-only int4 moves 312 MB per token and reads 272.8 tok/s, and its small
launches are bounded by a 14 µs fixed cost per dispatch rather than by bandwidth.

## What the host owns

The split is at the token loop, so this list is short by construction.

- **Admission and concurrency.** Batch size is compiled in, so concurrency is a
  policy decision made before a request reaches a plan. A host that wants *n*
  concurrent sequences needs *n* plan slots, each with its own cache, and a queue
  in front of them. The alternative is one plan compiled at batch *n*, which is
  the same trade with fewer host objects and more adapter work.
- **Cache lifetime across requests.** A plan returns a new cache; nothing is
  mutated in place. Reuse between unrelated requests, and when a cache is dropped,
  belong to the host.
- **Bucket widths.** A bucket that is too wide pays prefill for the padding. The
  widths should sit on the boundaries the workload clusters on, and a prompt longer
  than the widest bucket is refused rather than truncated.

And inside the loop, but still aotolith's job — these are the items in the table
above, not host responsibilities:

- **Mask construction**, in a public helper rather than the adapter's private
  field. Prefill needs an explicit rank-4 mask, because `is_causal` cannot
  distinguish a pad from a prompt token; decode needs one row per valid length. The
  precomputed table costs `capacity²` bytes per bucket and should not be duplicated
  per caller.
- **Sampling and stop conditions**, host-side over logits but shipped here: plans
  hold no RNG state, so nothing else can do it.
- **Tokenizer, chat template, detokenization, streaming**.

Explicitly outside, on both sides: request scheduling, continuous batching,
preemption, paged KV allocation, prefix cache, PD disaggregation, EP/MoE dispatch,
and a serving API.

## The compiler work that comes before any engine

Multi-row decode. One plan that consumes *m* new tokens per step and writes *m*
logits.

The reason is measured, not stylistic. Each launch costs about 14 µs before it
moves a byte (`t ≈ 14 µs + bytes / 373 GB/s`), and a decode step is a chain of
about 96 such launches (see [performance.md](performance.md#why-the-small-buckets-cannot-be-fixed)).
Making a single-row GEMV faster has been tried and does not move the total.
Verifying several tokens per launch does: it amortizes the fixed cost across rows
that share the same weight stream, which is exactly what speculative decoding
needs. With weight-only int4 the arithmetic is explicit: four rows read four times
the weight bytes and pay the fixed cost once.

This is compiler work because *m* is another compile-time constant. It fits the
existing contract instead of fighting it: an *m*-row plan is one more plan shape,
not a new dynamic dimension. It needs a decode plan whose token and position
inputs are `[m]`-wide, an adapter that builds the mask for *m* rows, and a host
that verifies and accepts. The first two are in this repository; the third is
host-side, which is the split this page argues for.

Until that exists, an engine embedding aotolith is limited to batch-1 decoding,
and the throughput argument for adopting it stays at 1.21x on one shape.

## Adoption path

**T0, embed the compiler.** `compile_module` or `compile_callable` on a model, run
the plan in your own loop. Available today. Proves the numerics and the kernel
quality; costs the adopter a host loop.

**T1, consume artifacts.** `aoto compile` writes a versioned archive, and any
process can `load_compiled_module` it. Available today. This is the integration
point for a host written in something other than Python, once an ABI for that
exists.

**T2, a text-in/text-out command.** `aoto generate` (and a `serve` variant over
the same code), taking a prompt, a bucket set and a sampling policy, printing or
streaming tokens. It belongs in `src/`, because a pip user who cannot run a model
does not get as far as evaluating the compiler. It stays deliberately narrow: one
request at a time, greedy plus one sampler, refuse a prompt past the widest bucket.
The moment it grows admission policy or a queue, the line has moved and that code
belongs in an engine.

Done means: `aoto generate --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..."` prints
text; it stops at EOS instead of always filling `max_new_tokens`; a prompt longer
than the widest bucket is refused with a clear error; `--stream` hands over tokens
as they decode; and it reports time-to-first-token and steady-state tok/s against
`mlx_lm` on the same prompt, so the 232 tok/s claim is reproducible from the
installed package rather than from a benchmark script.

**T3, an engine adopts aotolith as a backend.** The engine keeps scheduling,
batching and serving; aotolith supplies plans and kernels for the shapes the engine
declares. This is what the contract is for. It depends on multi-row decode, and on
a stable way to ask the compiler "can you serve these shapes?" before paying a
compile.

## When to revisit this

- **An issue says "I just want to run a model".** That is a T2 gap, not a compiler
  gap, and it is the most likely signal that the front door is missing.
- An external engine asks for a shape the compiler cannot express. That is compiler
  work, and it moves up the Roadmap.
- The T2 command starts growing scheduling policy, queues, or multi-model
  residency. That is the signal that the boundary moved by accident, and the code
  belongs in the embedding engine.

## Open questions

- Whether the mask tables should be built by the compiler, carried in the
  artifact, or produced by a public helper. They are currently adapter-private.
- Whether a host should be able to select a bucket without knowing the plan
  widths, i.e. whether `Qwen2BucketedRuntime` needs a shape-query API.
- What an artifact-level entry point for non-Python hosts looks like, given that
  ABI 1.3 describes a graph and a state layout rather than a token interface.
- Whether `--stream` should be the default. It costs about 28% of decode, so the
  honest defaults are: print at the end unless asked, and say why in `--help`.
