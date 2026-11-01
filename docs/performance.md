# Performance notes

What a decode step costs, what the last few rewrites bought, and which ideas
were measured and then dropped. The README keeps the summary; this page keeps the
measurements.

All numbers are from one machine: Apple M5 Pro, MLX 0.32, Qwen2.5-0.5B-Instruct,
batch 1, BF16 unless a row says otherwise.

## How the numbers were taken

Every Qwen2 measurement uses the same prompt, the same precision, and the same
`128 + 64` token settings, and `--mlx-baseline` times `mlx_lm`'s own greedy
decode on that prompt in the same process, so the baseline is measured where the
number is quoted rather than taken from elsewhere.

Two reporting choices matter more than they look. `decode_async` amortizes the
host loop across tokens, which is what a serving loop does; `decode_sync_each_step`
forces a device sync every token, which no serving loop needs, and it exists
because it isolates the device from the host. Whenever a rewrite is being
judged, the pass is ablated inside one session — off, on, off, on — so every pair
shares the machine's thermal and load state. Absolute rates move several percent
between sessions, so rows are comparable to each other and not to a session
recorded weeks earlier.

`aoto bench` is the regression gate, and it compares two different kinds of
number differently. `input_ops`, `lowered_ops`, `planned_peak_bytes` and
`peak_bound_bytes` are deterministic, so they are compared exactly on any
machine and a pass that adds a copy or drops a fusion fails with no timing noise.
Kernel counts and milliseconds depend on the GPU, so they are compared only when
the baseline names the same device and both reports have at least three repeats.
The threshold compares `min_ratio`, not `min_ms` or the median: contention only
ever adds time, so the best sample is the right statistic, and dividing it by a
reference matmul measured at the same moment is what removes the machine from
the number. Two runs of unchanged code here gave the same 0.34 ms projection
floors of 0.344 and 0.431 ms while the reference matmul moved from 0.89 to
1.56 ms; in ratio terms both were 0.39x. A run that is slower in milliseconds
but not in ratio is printed as a note instead of failing the gate. The default
tolerance is 25%, above the 15% swing two best-of-33 floors of an unchanged
kernel showed on this machine — and only the timing half of the gate has a
tolerance at all.

The reference matmul is sampled between a workload's samples, not around them.
One round timed the same 0.2 ms attention kernel at 0.203 ms and then the
reference at 1.15 ms, while the next round timed that kernel at 0.203 ms and the
reference at 0.78 ms; dividing the first pair gave a floor of 0.177 that no later
run could reproduce, so the gate failed on unchanged code with a 1.4x "regression"
that was entirely the reference's clock. Interleaving puts the numerator and the
denominator in the same window - same clock, same thermal state, same queue - at
the cost of one matmul per sample. Four consecutive runs then put the same
workload at 0.2425-0.2543 instead of 0.177-0.244.

Every workload is also compiled and executed once before timing starts, because
the first plan a process runs pays for the Metal kernel compilation of all its
kernels — that systematically doubled the first workload's numbers even through
its warmups. `--rounds N` repeats the timed half and keeps every sample; a
baseline recorded with `--rounds 3` has a floor from three times as many
samples, and CI compares against it with the same flag.

Baselines are a matrix, not a file: `benchmarks/baselines/<chip>.json`, one per
chip, and `--baseline benchmarks/baselines` resolves to the file for the machine
running. That is what makes the millisecond half scale past one Mac — a second
machine adds its own file instead of overwriting the first. A chip nobody has
recorded on still gets the deterministic half, taken from another chip's file,
with a note naming both devices; it is not a silent pass. So the op-count and
planned-peak gate is enforced on every Mac, and the timing gate is enforced
wherever somebody has recorded — which is the honest statement of coverage, and
the reason `--check-baselines` exists to keep the committed files trustworthy.

The fixed set covers one workload per mechanism that could regress on its own:
a dense projection, a bucket dispatch (96 rows into a 128 bucket, so the tile,
the trim and per-bucket lowering all execute), a gated MLP with the
`silu_mul_paired` epilogue, `RMSNorm`, causal attention, the int4 streaming
GEMV, and one prefill and one decode through each model family's adapter. The
two adapter pairs are what make a family a gated object rather than a claim: a
Llama decode that loses the bias-free projection fold, or grows an extra op
where Qwen2 does not have one, shows up as a count change in the deterministic
half of the gate.

The reference backend has a workload of its own. `reference_linear_1024x1024` is
the same projection as `linear_1024x1024` with `target="mlx-reference"`, so its
`kernels` count is zero by construction and the gate on it is the interpreter
path: the numbers it must not change are `input_ops`, `lowered_ops`, the storage
plan's two figures, and the millisecond floor of walking the graph through MLX.
It is the cheapest way to notice that an op lost its fallback, since the Metal
plan would simply not select the missing kernel while the reference plan has
nothing else to fall back to.

## Qwen2.5-0.5B, 128 prompt tokens and 64 generated

| configuration | prefill | decode, sync each step | decode, async | vs `mlx_lm` |
| --- | --- | --- | --- | --- |
| MLX fallback only | 9 500 tok/s | 146.3 tok/s | **174.8 tok/s** | 0.90x |
| `--codegen-matmul` | 11 800 tok/s | 166.4 tok/s | **231.7 tok/s** | **1.21x** |
| `--codegen-matmul --codegen-attention` | 12 400 tok/s | 166.3 tok/s | 232.3 tok/s | 1.20x |

(Median of 11 repeats; the same 64 generated tokens as the PyTorch reference in
every row, and each was run under the decoder's default `memory="latency"`
policy. Each row was run with its own `mlx_lm` baseline in the same process,
and those baselines read 192.2-195.3 tokens/s. The `--codegen-attention` row is
within noise of the one above it on decode, and its prefill reads a little
higher here rather than lower — the attention kernel's slower tile on prefill
shapes is smaller than this machine's run-to-run spread.)

All 64 greedy tokens match the PyTorch bf16 reference exactly, with or without
native kernels, and `mlx_lm` produces those same 64 tokens, so the comparison
above also compares two implementations that agree on the answer.

Agreeing on tokens is not the same as agreeing on numbers.
`examples/qwen_logit_distance.py` measures each path's last-token logits against
PyTorch, and the gap turns out to depend on how many tokens the prefill sees:

| prompt tokens | fallback mean | fallback max | codegen mean | codegen max |
| --- | --- | --- | --- | --- |
| 7 | 0.1056 | 0.6875 | 0.0290 | 0.1875 |
| 8 | 0.0783 | 0.5469 | 0.0451 | 0.2500 |
| 16 | 0.0312 | 0.2031 | 0.0310 | 0.1953 |
| 128 | 0.0596 | 0.4219 | 0.0566 | 0.4219 |

Read down a column rather than across a row: the fallback's mean distance moves
by 3.4x between a 7-token and a 16-token prompt, so any single figure here
describes the prompt as much as the model. What the table supports is narrower
than "the native path is closer" - code generation is never further from PyTorch
at these lengths, is 3.6x closer at 7 tokens, is level with it at 16 (0.0312 against 0.0310), and
is within 5% by a 128-token prefill. A run that carried on for 32 decode steps
put them within 6% of each other in the other order (fallback 0.061 mean, codegen
0.064), so the ordering is not stable enough to support a claim either way. The
honest summary is that code generation does not cost accuracy, not that it buys
much of one - and either way it is a throughput decision, not an accuracy one.

Earlier revisions of this section quoted a mean 0.043 / max 0.38 against 0.081 /
1.5 from one 7-token measurement, without saying that the prompt was 7 tokens.
Neither pair reproduces at this commit, and the second overstated the gap by 2x
while presenting a prompt-length artifact as a property of the model.

## What the MLX fallback costs in accuracy

A plan's float32 arithmetic is only as precise as the kernel that runs it, and
the two paths do not have the same precision. With `codegen_matmul` and
`codegen_attention` on, a two-layer Llama's last-token logits land `1.043e-07`
from PyTorch — fp32 round-off. With both off, the same graph runs on MLX's own
kernels and the drift is `1.3e-3`; turning on one at a time splits the blame
(`4.2e-4` for matmul alone, `8.5e-4` for attention alone).

The cause is the fallback, not the lowering: MLX's own float32 matmul is `8e-4`
relative to a float64 reference on this GPU, and that figure stays flat from
K=32 to K=2048. Accumulation error would grow with K, so this is a
reduced-precision matmul — TF32-shaped — rather than a summation that drifts.

Two consequences worth knowing. The fallback is a correctness-safe path: greedy
continuations match PyTorch exactly, because `1e-3` relative does not reorder
argmax on logits that differ by more than that. It is not a bit-accurate path,
so a test that pins logits to `1e-6` has to compile the kernels rather than
compare the interpreter against fp32 torch. `examples/tiny_llama_compile.py`
is the executable version of this paragraph.

## Where a decode step goes

Decode is bound by weight bandwidth, not by the host. Building a step costs
0.98 ms of Python and the GPU needs about 4.3 ms, so the host is fully hidden
behind the async queue; before the plan was unrolled the host cost 6.2 ms and
*was* the limit. One step streams 991 MB of weights (24 layers of `qkv` 2.07 MB,
`o_proj` 1.61 MB, one paired gate/up matrix of 17.4 MB, and `down` at 8.73 MB,
plus the 272 MB LM head), which at 4.30 ms per token is 230 GB/s.

Per-bucket GPU cost for one decode step, measured by replaying each bucket enough
times to keep the device queue full, so the wall clock is device time and host
submission is hidden:

| bucket | launches | traffic per launch | GPU time per launch | bandwidth |
| --- | --- | --- | --- | --- |
| LM head GEMV | 1 | 272.3 MB | 662 us | 412 GB/s |
| MLP down GEMV + residual | 24 | 8.73 MB | 32.6 us | 268 GB/s |
| MLP gate+up GEMV + RMSNorm + `silu_mul` | 24 | 17.44 MB | 62.5 us | 279 GB/s |
| fused QKV GEMV + RMSNorm | 24 | 2.07 MB | 10.8 us | 193 GB/s |
| attention, single query | 24 | 0.10 MB | 7.3 us | 14 GB/s |
| o_proj GEMV + residual | 24 | 1.61 MB | 7.8 us | 207 GB/s |
| RoPE q / k | 48 | 1 KB | 7.8 us | - |
| additive attention mask | 1 | 0.2 KB | 26.6 us | - |

That sums to 3.96 ms of device time and 991 MB, against 4.30 ms measured for the
whole step. The LM head plus the MLP matrices are 74% of it, and they are moving
bytes rather than waiting on dispatches; the rest is spread over launches that
each move 2 MB or less. A step carried 170 dependent launches when this table was
made. Removing one of the cheap ones does not move the total: folding the query
and key rotary calls into one 16-head rotation removed 24 launches and 24 graph
ops and decode stayed at 232.9 tokens/s against 232.5, inside the run-to-run
band, so it was not kept. Removing one that carries real traffic does move it:
pairing `gate` and `up` into one launch reads the same 17.4 MB but takes one
launch instead of two, and measured 1.031x-1.038x across interleaved A/B runs. The
difference is what the launch moves, not how many there are — until the launch
count is large enough that the dispatches themselves stop being free, which is
what [cutting 245 operators](#cutting-245-operators-out-of-a-decode-step)
measures.

Two ways of producing this table measure nothing at all, and both look
plausible. Dropping each round's output arrays frees the lazy graph before it
reaches the device, so the wall clock covers host submission only and the
bandwidth column comes out above 1 TB/s. Passing all-zero weights lets the driver
skip the read, which inflates the small buckets. A bucket has to keep every
round's outputs alive and feed real values before its numbers mean anything, and
any figure that exceeds what `mx.sum` over the same bytes reaches is measuring the
harness rather than the kernel. The replay also flatters the tiny buckets for the
opposite reason: the 24 instances of a bucket run back to back there, while in
the real step each layer waits for the previous one, so per-launch figures for
RoPE and attention are upper bounds and the table's sum only approximates the
step.

Two experiments in this area were measured and dropped. Writing the new
key/value straight into slot `positions` and selecting the live slots with the
mask reads better than rolling the window, but MLX lowers a scatter to an
index-compare pass over the whole window where the concatenate is a straight copy
of the same bytes: 0.18 ms slower per step, so the roll stays. What did change
later is that the roll stopped being eleven operators per layer — [it is one
`kv_cache_append` now](#the-kv-cache-roll-is-one-operator), with the index
arithmetic inside the kernel. And the rotary position for a decode token is
`valid_length - 1`, not `valid_length`; `Qwen2StaticCacheRuntime.decode_token`
documented the second and briefly used it, which cost 0.7 in max logit deviation
against the reference and broke the exact-token match. It now matches all 64
tokens.

## Why the small buckets cannot be fixed

Three measurements agree that the MLP buckets are not leaving anything on the
table, and each rules out a different explanation.

The first is the threadgroup size. If a 4864-wide output were too narrow to fill
the device, giving it more, smaller threadgroups would help. Sweeping the same
compiled plan from 64 to 1024 threads per threadgroup moves that bucket by 2%:

| threads per threadgroup | 4864×896 | 151936×896 |
| --- | ---: | ---: |
| 1024 (selected for narrow buckets) | 284 GB/s | 395 GB/s |
| 256 | 286 GB/s | 413 GB/s |
| 128 | 289 GB/s | 410 GB/s |
| 64 | 284 GB/s | 408 GB/s |
| 128, with 64 for the wide bucket | 282 GB/s | 291 GB/s |

The 4864 bucket is flat while the 151936 bucket moves by 30% in the other
direction: it wants 128-thread groups, and the 1024-thread setting that the
narrow buckets use costs it a third of its bandwidth. So occupancy is a real
effect, but it is not what separates the two buckets.

The second is MLX's own reduction. `mx.sum` over the same bytes follows the same
curve, and a lower one:

| bytes read | `mx.sum` | this GEMV |
| --- | ---: | ---: |
| 8 MB | 82 GB/s | 268 GB/s (8.7 MB) |
| 17.4 MB | 180 GB/s | 279 GB/s |
| 35 MB | 225 GB/s | — |
| 140 MB | 295 GB/s | — |
| 280 MB | 296 GB/s | 412 GB/s (272 MB) |

The projection beats MLX's reduction at every size, so this is not an
implementation artifact of this kernel.

The third is the controlled comparison inside the table above: the same
warp-per-column kernel reaches 412 GB/s on a 272 MB matrix and 279 GB/s on a
17.4 MB one. Only the bytes differ.

What all three show is that the rate a launch reaches depends on how many bytes
it reads, not on how its grid is shaped. To run the whole step at 412 GB/s its
991 MB would have to arrive in launches of hundreds of megabytes, but every layer
is cut into five strictly dependent projections — `qkv`, `o_proj`, paired
gate/up, `down` — and the largest is 17.4 MB, because the nonlinearity between
them is a real barrier. Pairing `gate` and `up` is exactly the move that exploits
this, and it is why that one change measured 1.03x while removing 24 launches of
1 KB measured nothing. Faster still means fewer bytes, which means quantized
weights, not a better kernel.

## Weight-only int4, and what it actually bought

`compile_qwen2_static_cache(..., quantize=64)` packs every projection and the LM
head — `qkv`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head` — and
leaves the embeddings and the norms alone. The token embedding is read one row at
a time, so 4 bits per element would buy nothing there and cost accuracy; a norm
is one vector per layer.

The format is stated once, in `aotolith/quantize.py`, because three pieces of
code have to agree on it. A weight stays `[out, in]`, so one row of the packed
matrix is one output unit. The reduction axis is cut into groups of `group_size`
elements that share a scale, `max|w| / 7`, symmetric, round to nearest — the
widest element lands exactly on `+7`, so negating a weight is exact. Eight nibbles
go into each `int32` word, nibble 0 lowest, two's complement, so a word is the
`uchar4` a kernel would load read as one integer. The scales sit beside the words
in the compute dtype: a bf16 scale differs from an f32 one by three orders of
magnitude less than one int4 step, and carrying f32 would force the kernel to
hold a second dtype.

What the compiler does with a packed weight depends on how many rows it
multiplies. A single-row matmul — decode — *is* the streaming kernel:
`fold_quantized_weights` rewrites `matmul(x, dequantize_i4(packed, scales))` into
one `matmul` carrying `weight_group_size`, and the GEMV prologue loads words,
sign-extends each nibble with two instructions (`(int)(word << 28) >> 28`),
multiplies by `x` and by the group scale, and accumulates in float. No unpacked
weight exists at any point. A wider matmul — prefill — keeps the unpack, because
with 128 rows the weight stream is the cheap part of the multiply and int4 would
only add decoding to a compute-bound pass; there the explicit `dequantize_i4`
lowers to one streaming kernel in `codegen/quantize.py` rather than to eight
elementwise MLX passes and a stack that writes several full-sized temporaries per
layer.

The step time tracks the launch count more than the bytes — [the next section](#cutting-245-operators-out-of-a-decode-step)
takes 245 operators out of the step and measures what that bought. What it leaves is
a decode step whose launches are each worth about 14 µs of fixed cost: isolating
one launch of this kernel gives `t ≈ 14 µs + bytes / 373 GB/s`, so a 2.3 MB
gate/up launch takes 20.8 µs at 110 GB/s while the 72 MB LM head takes 288 µs at
252 GB/s. Threadgroup size changes that by a few percent and a leaner nibble
decode changes it by less, which is why the next lever is verifying several
tokens per launch rather than a faster single-row kernel. The four measurements
that line fits are in [the architecture notes](architecture.md) (in Chinese).

Accuracy: int4 rounds each weight onto one of 15 values inside its group, so the
logits move, and greedy decode is a chaotic readout of that movement. Against a
float32 CPU forward on the *same* rounded weights, the runtime sits 0.51 off —
2.7% of the 18.5 logit scale — while torch's own bf16 forward on those weights
sits 0.41, or 2.2%, away. Both are bf16 noise; neither is a decode error. Greedy
then tracks the round-tripped checkpoint for 13 tokens and flips on a near-tie,
matching 16 of 64 overall. So the CI gate does not use token equality on this
model: `tests/test_quantized_qwen2.py` runs a 2-layer model where the check is
exact — every decode token has to match the round-tripped reference — and
`--quantize` reports both the bf16 reference and the round-tripped one so a run
never hides which one it agrees with.

Autotuning is not an escape either, and until now it could not even run on a
model like this one. `--autotune` evaluated the MLX fallback for every operation
in order to have a reference to compare against, but normalization, rotary, and
attention only ever lower to generated kernels, so compiling a transformer raised
`NotImplementedError` at the first standalone `rms_norm`. It now measures such an
operation against itself, records the single benchmark, and notes in the
selection reason that the kernel was the only valid candidate.

## Cutting 245 operators out of a decode step

A decode step of this model traces to 1909 operations — 147 casts, 339
reshapes, 265 slices, 169 multiplies, 49 normalizations — and the GPU work in it
took 2.5 ms of a 5.7 ms step. The rest was the gap between launches, about 4.6 µs
each, across 24 layers of roughly ten strictly dependent launches: a chain that
deep cannot be hidden by a host thread that queues each step in microseconds,
because the GPU sits idle between commands. Cutting that count turned out to be
worth more than making any single kernel faster.

The largest single win was in how PyTorch spells RMSNorm. Written out, it is
`(x.to(f32).pow(2).mean().rsqrt() * x.to(f32) * weight.to(f32)).to(x.dtype)`,
which the importer faithfully records as five operations: a widen cast, a
normalization without a weight, a second widen cast for the weight, a multiply,
and a narrow cast. The compiler had a weighted `rms_norm` for exactly this shape
since it lowers to the same Metal kernel, but nothing matched the five-op spelling
to it, so each of the 49 normalizations in the graph became five launches and
four full-width temporaries.

`fuse_norm_scale` matches the sequence and emits one `rms_norm` carrying the
weight. That is 245 operators gone from the step, the 49 multiplies among them
and the rest casts and layout work that only existed to feed them. The split is
worth stating because the two halves are removed by different passes: 196 go
where the fold runs, and the remaining 49 are the multiplies that
`fuse_rms_norm_epilogue` can only reach once the norm is no longer standing in
front of them. Measured end to end, dropping `fuse_norm_scale` from the pipeline
costs 245 ops on both variants — 1909 ops to 634 (lowered to 683) on bf16, and
2030 to 682 on int4. `examples/qwen_op_counts.py` prints the table, including
each pass's own delta and its net contribution, and it is the reason the numbers
in this section are stable: the per-pass figures hold as long as the pass does,
while the absolute counts move with the transformers version that traced the
graph. An earlier revision of this paragraph quoted 1239 ops to 994 and 1191 to
995; the nets below are unchanged from that measurement, and only the endpoints
moved. The fold needs no tolerance because it is
bit-identical: both the row and the weight arrive in the dtype the product is
narrowed back to, and the fused kernel forms the product in fp32 and rounds once,
which is exactly what the five-op chain did. The pass checks that precondition and
declines when it does not hold — a graph that really wants a standalone fp32 row
keeps one — and `tests/test_passes_norm.py` asserts bit equality against the
unfolded graph rather than a tolerance.

Measured by ablating exactly that pass in one interleaved session — off, on, off,
on — so every pair below shares the machine's thermal and load state:

| | bf16 | int4 g64 |
| --- | ---: | ---: |
| graph ops the fold removes | 245 | 245 |
| decode, synchronized each step | 155.4 → 167.2 tok/s (6.44 → 5.98 ms) | 179.0 → 228.3 tok/s (5.59 → 4.38 ms) |
| decode, asynchronous | 199.2 → 232.2 tok/s (5.02 → 4.31 ms) | 195.3 → 272.8 tok/s (5.12 → 3.67 ms) |
| prefill, 128 tokens | 11 208 → 12 298 tok/s | 9 494 → 10 375 tok/s |
| peak memory | 4.26 GiB, unchanged | 1.99 → 1.89 GiB |

> The op-count rows quote what the fold removes, not the absolute step size. The
> traced base has grown since these timings were taken — the step traces to 1909
> ops on bf16 and 2030 on int4 today, because the view operators the frontend
> records follow the installed transformers version — while each fold's net is
> unchanged at 245. `examples/qwen_op_counts.py` prints both columns.


The before/after pairs are what the ratios rest on, and they are 1.08x and 1.17x
for bf16, 1.28x and 1.40x for int4, and 1.09-1.10x on prefill, for 245 fewer
operators and no kernel change. Async still leads sync after the fold — 4.31 ms
against 5.98 — so the host loop is still hidden behind the queue; the fold did
not move the launch floor into view. The step's cost per operator reads 4.51 µs
before and 4.41 µs after, which is the shape of the finding: the time is the
operator count times a constant, and the constant did not move when the count
did.

The same pass exposed a second problem worth documenting because the fix is
scoped rather than general. `_match_rms_norm_consumers` folds a normalization into
the *prologue* of the projection that reads it, so the GEMV scales its input rows
as it loads them. That is free for a packed projection, whose prologue is
generated Metal. It is not free for a dense one, which falls back to MLX and would
re-derive the normalization once per reader — slower than computing it once. The
matcher absorbed a normalization only when the projection carried a packed
weight, which kept dense graphs honest and gave up the same trade on the dense
projections that *are* generated. `rewrite` now takes `native_matmul`, which
`compile_graph` fills in from `codegen_matmul`: a dense single-row projection
folds when the compiler will emit the GEMV for it, and keeps its standalone norm
when it will not. On Qwen2.5-0.5B decode that is 683 operators to 634 and 227.8
to 233.0 tokens/s async, 171.8 to 177.8 tokens/s sync, over the same 64 tokens;
folding regardless of who lowers the multiply reads 189.6 to 138.8, which is the
27% that made the condition worth keeping.

## The KV-cache roll is one operator

A decode step writes one key and one value per layer into a fixed-capacity
window, and the spellings torch has for that are all longer than the fact. The
adapter slices the oldest slot off the layer's pair, stacks this step's two new
rows, concatenates them onto the window, then slices the pair apart again and
squeezes both halves because attention wants them separate. That is eleven
operators per layer, plus the three views — slice, reshape, transpose — that read
the value back out of the projection row it was just computed in. Across 24 layers
it was 312 of the 946 operators the step carried without it, a third of the graph, for a statement that
fits in one line: slide the window down a slot, write the new key and value into
the freed slot, hand back the window and its two halves.

`kv_cache_append` is that line. It takes the whole layer-stacked cache, this
step's key, and the packed projection row, and returns the new pair plus both
windows. Its contract in `ir/ops/kernels.py` is where the meaning is written
down: the cache is `[layers, 2, batch, heads, slots, dim]` with the pair on axis
1, `layer` says which slice of it this step owns, and `value_start` says where in
the projection row the value half begins — the operator reads it there, so no
shaped copy of it ever has to exist. `fuse_kv_cache_update` in `passes/cache.py`
is the rewrite, and it is deliberately narrow: the concatenate has to build a
pair exactly one slot longer than a window that is a front-slice of the *same*
pair, the two halves have to be unpacked by a slice and a squeeze with exactly
the shapes the operator produces, and nothing the rewrite deletes may be read by
an operator outside the chain. A window that drops two slots, a value that is not
the projection's, a pair with a third consumer — each keeps its general form. It
runs first in the pipeline, because later passes would happily tidy the very
views it has to see.

The backend emits one kernel per layer. Per `(half, batch, head)` the slots are
one contiguous run, so dropping the oldest and appending one is that run read one
row further along: every slot but the last is a vector copy from the source, and
the last slot is picked from the key tensor or from the projection row. Four
elements per thread is eight bytes for a bf16 window, and the only branch is the
one on the final row. It is still a copy of the whole window — under a hundred
kilobytes per layer — so the bytes moved are unchanged and the win is entirely
the launches that disappeared. The MLX fallback in `runtime/mlx_ops` computes the
same three outputs for autotuning and for shapes the kernel declines, and the
unrolled plan runner learned to bind multi-output operators so neither path is
interpreted.

Measured the same way as the section above — the pass off, on, off, on, in one
session — on the 0.5B model, bf16 and int4:

| | bf16 | int4 g64 |
| --- | ---: | ---: |
| graph ops the fold removes | 312 | 312 |
| decode, synchronized each step | 147.9 → 154.6 tok/s (6.76 → 6.47 ms) | 228.2 → 251.5 tok/s (4.38 → 3.98 ms) |
| decode, asynchronous | 178.8 → 188.5 tok/s (5.59 → 5.30 ms) | 277.2 → 308.7 tok/s (3.61 → 3.24 ms) |
| greedy tokens | unchanged, all 64 | unchanged, all 64 |

A third of the graph bought 1.045x and 1.054x on bf16 and 1.10x and 1.11x on
int4, not a third of the throughput, and that is the honest reading: these were
the cheapest operators in the step, each moving a window smaller than the
smallest projection row. The int4 ratios are larger because its step is shorter,
so the same 312 launches were a larger share of it. What this settles is that the
cost really is per launch — an operator that copies 90 kB costs about as much as
one that moves 2 MB, and removing 312 of them is worth roughly what the launch
floor predicts.

The obvious next move was the one this rewrite sets up but does not take: with
the write indexed by a slot counter instead of by position, the window would not
have to shift at all, and the kernel would be a single row store, with attention
reading a ring. That was measured and dropped before writing it. Reducing the
emitted kernel to just the final row — no window copy at all, so the numbers are
wrong but the launch, the grid, and the traffic shape are those of the ring
version — reads 183.2 tok/s async against 188.3 for the real kernel over two runs
each, 186.4-190.2 and 182.9-183.4. There is nothing there to win: the copy is
90 kB against a step that streams 991 MB of weights, so it is absorbed by the
queue, and what the rewrite would actually remove is a barrier rather than
bandwidth. The copy stays, and the ring belongs to an attention kernel that reads
the window as a permutation it does not care about.

## What a bucket boundary costs

A bucket set serves a request the compiler never saw, and the price is the rows
between the request's width and the bucket's. That price is not one number,
because it depends on whether the launch is reading its weight or its
activations. Measured on the M5 Pro with a `1024 x 1024` projection, the same
plan compiled directly at the request's width as the reference, minimum of 15
calls over 7 rounds:

| rows | bucket | rows wasted | direct | through the set | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 96 | 0% | 0.159–0.281 ms | 0.151–0.275 ms | 0.95–1.03x |
| 96 | 128 | 33% | 0.159–0.281 ms | 0.157–0.244 ms | 0.87–1.06x |
| 4096 | 8192 | 100% | 0.572–0.588 ms | 1.241–1.251 ms | 2.16–2.18x |

The first row is the dispatch itself: picking a bucket, checking the width and
running the plan lands inside the run-to-run noise of calling the plan directly,
because a request that fits a bucket exactly needs no padding and no slice. The
second row is the case a bucket set is designed for: 96 rows served by the 128
bucket is 33% more rows, and the launch does not get 33% slower, because it is
reading a 4 MB weight matrix either way and 32 extra rows of a `1024`-wide
activation are cheap next to that. The third row is the case it is not designed
for: past a few thousand rows the projection is activation-bound, so the padded
rows cost their share and a little more.

The rule that follows is the one the Qwen2 buckets already use. Put the
boundaries where the traffic clusters, and keep the widest bucket close to the
largest width the host will admit, because every row between a request and its
bucket is computed and discarded. `plans.stats.snapshot()` reports both halves of
that trade: which bucket served each request, and how many rows were padded away.

## Peak memory, and what releases it

The plan's storage layout has always known which value dies at which operation.
Until this release the runner could not act on it: the unrolled body was one
lazy graph, so every intermediate stayed referenced until the caller evaluated
the outputs, and MLX's evaluator allocated an output for every node in that
graph before releasing the ones it was done with. The peak was therefore the sum
of the intermediates, not the largest live set.

Measured on a chain of eight concatenations of a 64 x 1024 fp32 input, where the
storage plan's live set peaks at 134.2 MB and the intermediates add up to
200.8 MB:

| policy | peak | synchronizations |
| --- | ---: | ---: |
| `latency` | 200.8 MB | 0 |
| `balanced` | 134.5 MB | 4 |
| `minimal` | 134.5 MB | 9 |

The mechanism has to be a synchronization and not just a `del`. Deleting the
Python name frees nothing on its own — the node is still in the graph — and
`mx.async_eval` does not free it either, because the buffer is allocated when the
command is encoded: the same chain measured 200.8 MB through `mx.async_eval` and
134.5 MB through `mx.eval`. So the runner evaluates the values that are still
live at a release point and then deletes the names that died there, which is what
makes the storage plan's releases real rather than advisory.

That is also why the policy is a choice: a synchronization on a decode step whose
whole graph is 0.6 ms is expensive. A `minimal` runner on `qwen2_decode_128`
measured 5.3 ms against 0.6 ms — nine times the work, for a plan whose transient
is 0.4 MB. The `balanced` threshold (`max(8 MiB, transient/8)`) exists to make
the policy free on the synthetic workloads, which all hold less than 8 MiB of
transient: `balanced` emits no synchronization there and measures the same as
`latency` to within noise.

It does not make the policy free on a real model. A 0.5B decode step holds
8.46 MiB of transient — just enough to clear the floor — so `balanced` emits one
barrier per step, and one barrier per step is enough to cost a fifth of the step
rate, because it flushes the pipeline the caller is using to overlap the next
step's host-side build: 240.9 tok/s under `latency` against 196 under `balanced`,
and prefill 11 660 against 8 707 with seven barriers. The threshold's own
estimate of the work, `transient/8`, is 1.06 MB there, which is why the 8 MiB
floor decides the outcome rather than the fraction. That measurement is why the
decoder runtimes default to `latency`; see [Memory](usage.md#memory).

Two costs were paid for that guarantee. The first is one cache-sized comparison
per decode step: `decode_mask` builds the window instead of indexing a
`capacity x capacity` boolean table. The table was 16 MiB at a 4096-token
capacity and would have been 1 GiB at 32k, to save a few microseconds. The
second is the single-bucket entry point sharing its weights with the decode plan;
`compile_qwen2_bucketed` had always shared them across buckets, but
`compile_qwen2_static_cache` traced prefill and decode separately and kept two
copies of the checkpoint. `memory_footprint()` reports both, and
`check_memory_budget()` compares a budget against one decode step, which holds
two caches because a step allocates the cache it returns before the one it was
given is dropped.

`aoto bench` now records all of it per workload: `planned_peak_bytes` and
`peak_bound_bytes` (deterministic, gated exactly on any machine),
`peak_extra_bytes` (measured, gated with the timing tolerance on the baseline's
own device), `first_call_ms` and `compile_ms` (gated with a factor of two, since
the shader compiler moved 1.5x between runs of unchanged code).
