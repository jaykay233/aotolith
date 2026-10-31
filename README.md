# Aotolith

English | [简体中文](README.zh-CN.md)

An experimental Transformer model compiler with native Apple Silicon and Metal support.

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml/badge.svg)](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#status)

Aotolith captures PyTorch models as FX graphs, lowers them through a typed SSA IR, and produces an executable plan backed by generated Metal kernels and MLX fallbacks.

> [!IMPORTANT]
> This project compiles Transformer model computation. Request scheduling, continuous batching, paged KV allocation, sampling, distributed execution, and serving belong to the host runtime.

## Why This Project

Most Transformer compiler work is built around CUDA and mature NVIDIA kernel ecosystems. This project explores a separate path: keep a backend-independent graph IR, then compile model computation natively for Apple Silicon instead of treating MLX as the whole runtime.

The compiler owns graph capture, legalization, layout simplification, operator fusion, common-subexpression elimination, tensor lifetime analysis, kernel selection, and Metal code generation. MLX supplies fallback operations and the numerical execution baseline while native coverage grows.

## Status

The project is usable for static, end-to-end Transformer experiments. It is not yet a stable compiler product.

| Area | Current state |
|---|---|
| PyTorch frontend | Compiles static-shape callables and `nn.Module` instances; model parameters and buffers are bound into the plan |
| Graph compiler | Typed SSA IR with shape/dtype checks, legalization, layout folding, fusion, and CSE |
| Storage planning | Computes live intervals, release points, peak live bytes, reusable buffer slots, and a real arena used during execution |
| Kernel selection | Records a native Metal or MLX implementation and fallback reason for every lowered operation |
| Executable artifact | Versioned `.aoto` archives contain graphs, kernels, selections, storage metadata, and module state; they load and execute in a fresh process |
| Autotuning | Opt-in profiling measures native and MLX candidates; measurements and the final selection are stored in the artifact |

The current regression compiles a two-layer HuggingFace Llama from 220 FX operations to 136 lowered operations. Its logits match a PyTorch reference with maximum absolute error `8.941e-08`, and the compiled plan remains executable after the original module is released.

## Compilation Pipeline

```text
PyTorch FX / nn.Module
        │
        ▼
Typed SSA IR
        │  legalization, layout folding, fusion, CSE
        ▼
Liveness analysis / storage plan
        │
        ▼
Kernel selection / fallback planning
        │
        ├── generated Metal kernels
        └── MLX fallback operations
        ▼
CompiledPlan
        │
        ├── first call unrolls the graph into straight-line Python
        ├── save: manifest + graphs + kernels + selections + state
        │
        ▼
Versioned artifact → reload and rebuild kernels → Apple GPU
```

A step that runs thousands of times per generation should not re-resolve its operands every time. The first call therefore compiles the plan into a Python function whose body is one assignment per operation, with kernel call sites bound as constants; later calls validate their inputs and run that function. Nothing about the kernels changes, only the bookkeeping around them.

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- MLX 0.32
- PyTorch 2.7+ for FX capture

## Installation

```bash
git clone https://github.com/jaykay233/aotolith.git
cd aotolith

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For a smaller install that only compiles Python callables, use `pip install -e '.[fx]'`. The `dev` and `hf` extras add HuggingFace Transformers for model-level compilation and tests.

## Quick Start

The example compiles a small `addmm + SiLU` function. Both the graph and the kernel selection can be inspected before execution.

```python
import mlx.core as mx
import torch

import aotolith as aoto


def fn(x, weight, bias):
    hidden = torch.addmm(bias, x, weight, alpha=2.0)
    return torch.nn.functional.silu(hidden)


torch.manual_seed(0)
example_inputs = (
    torch.randn(4, 16),
    torch.randn(16, 32),
    torch.randn(32),
)

compiled = aoto.compile_callable(fn, example_inputs, codegen_matmul=True)
inputs = tuple(mx.array(value.numpy()) for value in example_inputs)
output = compiled.plan(*inputs)[0]
mx.eval(output)

print("peak live bytes:", compiled.plan.storage_plan.peak_live_bytes)
for selection in compiled.plan.selections:
    print(selection.op, selection.implementation, selection.reason)
```

### Compile a HuggingFace Model

`compile_module` traces the model with its parameters and buffers as explicit state. The returned plan does not retain the original `nn.Module`.

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

The config above leaves `use_cache` at its default of `True`. That is accepted: the cache starts empty, a zero-length concatenation operand contributes nothing, and the frontend folds it away, so the compiled graph is the same one `use_cache=False` produces. A plan is a single forward pass with no hidden state, so the cache the model returns is not part of it; cached decoding needs the static-cache adapter below.

A tensor keyword argument becomes a graph input, so a caller can change it per call instead of compiling it into the graph:

```python
position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"position_ids": position_ids},
)
logits = compiled(input_ids, position_ids=position_ids)[0]
```

Non-tensor keyword arguments stay fixed at trace time. Promotion is recursive, so a keyword that holds tensors inside a list, tuple or mapping produces one graph input per tensor and the caller passes the same nesting back:

```python
cos, sin = position_embeddings  # a pair of tensors, as newer HF layers take them
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"position_embeddings": (cos, sin)},
)
logits = compiled(input_ids, position_embeddings=(cos, sin))[0]
```

Leaves that are not tensors — `use_cache=False`, or a float inside the same mapping — remain tracing-time constants. Passing a different value for one raises instead of being silently ignored.

### Compile a Qwen2.5 Static KV Cache

`aotolith.models.qwen2` provides a single-request static-graph adapter for Qwen2/Qwen2.5. It separates prefill and one-token decode into two graphs connected by a fixed-capacity KV cache, and implements Qwen2 RMSNorm, RoPE, GQA, SwiGLU, and the LM head.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aotolith.models.qwen2 import compile_qwen2_static_cache


model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
).eval()
prompt_ids = tokenizer.encode(
    "Explain KV cache in one sentence", add_special_tokens=False
)[:128]

runtime = compile_qwen2_static_cache(
    model,
    prompt_length=len(prompt_ids),
    max_new_tokens=64,
    quantize=64,  # optional: weight-only int4, group size divides each reduction
)
tokens = runtime.greedy_generate(prompt_ids, max_new_tokens=64)
print(tokens.tolist())
```

The adapter fixes batch size, prompt length, and maximum generation length and implements greedy single-request generation only. The host owns the token loop; scheduling, batching, paged KV allocation, and sampling remain outside the compiler. `runtime.save("qwen")` writes separate prefill and decode `.aoto` files, each currently containing its own copy of the model state.

#### Size buckets

A fixed prompt length means a serving loop recompiles for every new one. `compile_qwen2_bucketed` compiles one plan pair per bucket width instead, and serves any shorter prompt by left-padding it into the smallest bucket that fits:

```python
from aotolith.models.qwen2 import compile_qwen2_bucketed

runtime = compile_qwen2_bucketed(
    model,
    prompt_lengths=[128, 512, 2048],
    max_new_tokens=64,
)
tokens = runtime.greedy_generate(prompt_ids)  # 128, 512, or 2048
```

Left padding is what keeps the decode layout: the decode masks select the trailing window of the cache, so the real tokens have to end there. `is_causal=True` cannot tell a pad from a prompt token, though, so the prefill graph takes an explicit rank-4 mask that carries causality and padding in one tensor. At the full bucket width that mask is exactly the causal mask, so one plan serves both the padded and the exact case.

Only the bucket widths vary a shape. The weights are shared by identity across every bucket's plan, so a three-bucket runtime holds one copy of the model rather than three. `save(prefix)` still writes one archive per bucket plan (`prefix.128.prefill.aoto` and so on), and each archive carries its own copy of the state.

Query, key, and value share one normalized input, so the adapter stores them as a single packed projection per layer and multiplies once. Splitting the result costs two view operations and removes two kernel launches: the 128-row key and value projections were far too small to fill the GPU, and each ran at under a fifth of the bandwidth the fused projection reaches.

A gated MLP has no such packed weight, because `gate` and `up` are multiplied rather than concatenated. `fuse_gate_up_projection` supplies the missing half: it recognises `silu(gate) * up` where both operands trace back to a projection over the same activation, and replaces the pair with one multiply carrying a `silu_mul_paired` epilogue. The kernel streams the gate row and the up row of a decode token's weights in the same loop, and writes `silu(gate) * up` once, so the step reads the same 17.4 MB in one launch instead of two. It only fires for a single-row activation; a taller operand goes to the tiled multiply, which has no paired epilogue, and the projections stay separate. The gate is rounded to BF16 before `silu`, which is the value the separate projection would have written, so the fusion is bit-identical rather than merely close — a version that kept the extra precision in the FP32 accumulator flipped a greedy token on step four of a 24-token run.

The repository benchmark uses the same prompt, precision, and `128 + 64` token settings for PyTorch MPS, MLX, and Aotolith:

```bash
python examples/qwen_static_cache_benchmark.py --output /tmp/qwen-aotolith.json
python examples/qwen_static_cache_benchmark.py \
  --codegen-matmul \
  --mlx-baseline \
  --output /tmp/qwen-aotolith-codegen.json
python examples/qwen_static_cache_benchmark.py --quantize 64 \
  --codegen-attention --output /tmp/qwen-aotolith-int4.json
```

`--mlx-baseline` times `mlx_lm`'s own greedy decode on the same prompt in the same process, so the native number is measured on the machine running the benchmark instead of quoted from elsewhere. On this machine (M5 Pro, MLX 0.32) it reads **195.5-197.2 tokens/s** across runs.

| configuration | prefill | decode, sync each step | decode, async | vs `mlx_lm` |
| --- | --- | --- | --- | --- |
| MLX fallback only | 9 500 tok/s | 146.3 tok/s | **174.8 tok/s** | 0.90x |
| `--codegen-matmul` | 11 800 tok/s | 166.4 tok/s | **231.7 tok/s** | **1.21x** |
| `--codegen-matmul --codegen-attention` | 12 400 tok/s | 166.3 tok/s | 232.3 tok/s | 1.20x |

(Qwen2.5-0.5B-Instruct, 128 prompt tokens, greedy, batch 1, median of 11 repeats, BF16; the same 64 generated tokens as the PyTorch reference in every row. Each row was run with its own `mlx_lm` baseline in the same process, and those baselines read 192.2-195.3 tokens/s. The `--codegen-attention` row is within noise of the one above it on decode, and its prefill reads a little higher here rather than lower — the attention kernel's slower tile on prefill shapes is smaller than this machine's run-to-run spread. Absolute rates move several percent between sessions; the rows above are comparable to each other and not to a session recorded weeks earlier.)


`decode_async` is the figure to compare against `mlx_lm`, because both amortize the host loop across tokens; `decode_sync_each_step` pays a device sync every token, which no serving loop needs. The async path used to trail the synchronous one and now leads it, because the plan only forces a single device sync per step.

All 64 greedy tokens match the PyTorch bf16 reference exactly, with or without native kernels. `mlx_lm` produces those same 64 tokens, so the throughput comparison above also compares two implementations that agree on the answer. Agreeing on tokens is not the same as agreeing on numbers, and the two paths are not equally close to the reference: on the same model's 24-layer forward, the native kernels land a mean 0.043 / max 0.38 from PyTorch's logits, the MLX fallback a mean 0.081 / max 1.5. So enabling code generation moves the arithmetic toward PyTorch rather than away from it, which makes it a throughput decision and not an accuracy one.

Decode is bound by weight bandwidth, not by the host. Building a step costs 0.98 ms of Python and the GPU needs about 4.3 ms, so the host is fully hidden behind the async queue; before the plan was unrolled the host cost 6.2 ms and *was* the limit. One step streams 991 MB of weights (24 layers of `qkv` 2.07 MB, `o_proj` 1.61 MB, one paired gate/up matrix of 17.4 MB, and `down` at 8.73 MB, plus the 272 MB LM head), which at 4.30 ms per token is 230 GB/s.

Where the remaining headroom sits shows up in the per-launch rates below: the 272 MB LM head launch sustains 412 GB/s, while the 17.4 MB MLP launches sustain 279. An earlier version of this section blamed the narrow grid — a 4864-wide output supposedly cannot fill the device the way a 151936-wide one can. That explanation does not survive measurement; see [why the small buckets cannot be fixed](#why-the-small-buckets-cannot-be-fixed) below.

Per-bucket GPU cost for one decode step, measured by replaying each bucket enough times to keep the device queue full, so the wall clock is device time and host submission is hidden:

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

That sums to 3.96 ms of device time and 991 MB, against 4.30 ms measured for the whole step. The LM head plus the MLP matrices are 74% of it, and they are moving bytes rather than waiting on dispatches; the rest is spread over launches that each move 2 MB or less. A step carried 170 dependent launches when this table was made. Removing one of the cheap ones does not move the total: folding the query and key rotary calls into one 16-head rotation removed 24 launches and 24 graph ops and decode stayed at 232.9 tokens/s against 232.5, inside the run-to-run band, so it was not kept. Removing one that carries real traffic does move it: pairing `gate` and `up` into one launch reads the same 17.4 MB but takes one launch instead of two, and measured 1.031x-1.038x across interleaved A/B runs. The difference is what the launch moves, not how many there are — until the launch count is large enough that the dispatches themselves stop being free, which is what [the next section](#cutting-245-operators-out-of-a-decode-step) measures.

Two ways of producing this table measure nothing at all, and both look plausible. Dropping each round's output arrays frees the lazy graph before it reaches the device, so the wall clock covers host submission only and the bandwidth column comes out above 1 TB/s. Passing all-zero weights lets the driver skip the read, which inflates the small buckets. A bucket has to keep every round's outputs alive and feed real values before its numbers mean anything, and any figure that exceeds what `mx.sum` over the same bytes reaches is measuring the harness rather than the kernel. The replay also flatters the tiny buckets for the opposite reason: the 24 instances of a bucket run back to back there, while in the real step each layer waits for the previous one, so per-launch figures for RoPE and attention are upper bounds and the table's sum only approximates the step.

Two experiments in this area were measured and dropped. Writing the new key/value straight into slot `positions` and selecting the live slots with the mask reads better than rolling the window, but MLX lowers a scatter to an index-compare pass over the whole window where the concatenate is a straight copy of the same bytes: 0.18 ms slower per step, so the roll stays. What did change later is that the roll stopped being eleven operators per layer — [it is one `kv_cache_append` now](#the-kv-cache-roll-is-one-operator), with the index arithmetic inside the kernel. And the rotary position for a decode token is `valid_length - 1`, not `valid_length`; `Qwen2StaticCacheRuntime.decode_token` documented the second and briefly used it, which cost 0.7 in max logit deviation against the reference and broke the exact-token match. It now matches all 64 tokens.

### Why the small buckets cannot be fixed

Three measurements agree that the MLP buckets are not leaving anything on the table, and each rules out a different explanation.

The first is the threadgroup size. If a 4864-wide output were too narrow to fill the device, giving it more, smaller threadgroups would help. Sweeping the same compiled plan from 64 to 1024 threads per threadgroup moves that bucket by 2%:

| threads per threadgroup | 4864×896 | 151936×896 |
| --- | ---: | ---: |
| 1024 (selected for narrow buckets) | 284 GB/s | 395 GB/s |
| 256 | 286 GB/s | 413 GB/s |
| 128 | 289 GB/s | 410 GB/s |
| 64 | 284 GB/s | 408 GB/s |
| 128, with 64 for the wide bucket | 282 GB/s | 291 GB/s |

The 4864 bucket is flat while the 151936 bucket moves by 30% in the other direction: it wants 128-thread groups, and the 1024-thread setting that the narrow buckets use costs it a third of its bandwidth. So occupancy is a real effect, but it is not what separates the two buckets.

The second is MLX's own reduction. `mx.sum` over the same bytes follows the same curve, and a lower one:

| bytes read | `mx.sum` | this GEMV |
| --- | ---: | ---: |
| 8 MB | 82 GB/s | 268 GB/s (8.7 MB) |
| 17.4 MB | 180 GB/s | 279 GB/s |
| 35 MB | 225 GB/s | — |
| 140 MB | 295 GB/s | — |
| 280 MB | 296 GB/s | 412 GB/s (272 MB) |

The projection beats MLX's reduction at every size, so this is not an implementation artifact of this kernel.

The third is the controlled comparison inside one table above: the same warp-per-column kernel reaches 412 GB/s on a 272 MB matrix and 279 GB/s on a 17.4 MB one. Only the bytes differ.

What all three show is that the rate a launch reaches depends on how many bytes it reads, not on how its grid is shaped. To run the whole step at 412 GB/s its 991 MB would have to arrive in launches of hundreds of megabytes, but every layer is cut into five strictly dependent projections — `qkv`, `o_proj`, paired gate/up, `down` — and the largest is 17.4 MB, because the nonlinearity between them is a real barrier. Pairing `gate` and `up` is exactly the move that exploits this, and it is why that one change measured 1.03x while removing 24 launches of 1 KB measured nothing. Faster still means fewer bytes, which means quantized weights, not a better kernel.

### Weight-only int4, and what it actually bought

`compile_qwen2_static_cache(..., quantize=64)` packs every projection and the LM head — `qkv`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head` — and leaves the embeddings and the norms alone. The token embedding is read one row at a time, so 4 bits per element would buy nothing there and cost accuracy; a norm is one vector per layer.

The format is stated once, in `aotolith/quantize.py`, because three pieces of code have to agree on it. A weight stays `[out, in]`, so one row of the packed matrix is one output unit. The reduction axis is cut into groups of `group_size` elements that share a scale, `max|w| / 7`, symmetric, round to nearest — the widest element lands exactly on `+7`, so negating a weight is exact. Eight nibbles go into each `int32` word, nibble 0 lowest, two's complement, so a word is the `uchar4` a kernel would load read as one integer. The scales sit beside the words in the compute dtype: a bf16 scale differs from an f32 one by three orders of magnitude less than one int4 step, and carrying f32 would force the kernel to hold a second dtype.

What the compiler does with a packed weight depends on how many rows it multiplies. A single-row matmul — decode — *is* the streaming kernel: `fold_quantized_weights` rewrites `matmul(x, dequantize_i4(packed, scales))` into one `matmul` carrying `weight_group_size`, and the GEMV prologue loads words, sign-extends each nibble with two instructions (`(int)(word << 28) >> 28`), multiplies by `x` and by the group scale, and accumulates in float. No unpacked weight exists at any point. A wider matmul — prefill — keeps the unpack, because with 128 rows the weight stream is the cheap part of the multiply and int4 would only add decoding to a compute-bound pass; there the explicit `dequantize_i4` lowers to one streaming kernel in `codegen/quantize.py` rather than to eight elementwise MLX passes and a stack that writes several full-sized temporaries per layer.

The step time tracks the launch count more than the bytes — [the next section](#cutting-245-operators-out-of-a-decode-step) takes the count from 1239 to 994 and measures what that bought. What it leaves is a decode step whose launches are each worth about 14 µs of fixed cost: isolating one launch of this kernel gives `t ≈ 14 µs + bytes / 373 GB/s`, so a 2.3 MB gate/up launch takes 20.8 µs at 110 GB/s while the 72 MB LM head takes 288 µs at 252 GB/s. Threadgroup size changes that by a few percent and a leaner nibble decode changes it by less, which is why the next lever is verifying several tokens per launch rather than a faster single-row kernel. See [the launch floor](docs/architecture.md#小-launch-地板是固定的-14-µs不是带宽) for the four measurements that line fits.

Accuracy: int4 rounds each weight onto one of 15 values inside its group, so the logits move, and greedy decode is a chaotic readout of that movement. Against a float32 CPU forward on the *same* rounded weights, the runtime sits 0.51 off — 2.7% of the 18.5 logit scale — while torch's own bf16 forward on those weights sits 0.41, or 2.2%, away. Both are bf16 noise; neither is a decode error. Greedy then tracks the round-tripped checkpoint for 13 tokens and flips on a near-tie, matching 16 of 64 overall. So the CI gate does not use token equality on this model: `tests/test_quantized_qwen2.py` runs a 2-layer model where the check is exact — every decode token has to match the round-tripped reference — and `--quantize` reports both the bf16 reference and the round-tripped one so a run never hides which one it agrees with.

Autotuning is not an escape either, and until now it could not even run on a model like this one. `--autotune` evaluated the MLX fallback for every operation in order to have a reference to compare against, but normalization, rotary, and attention only ever lower to generated kernels, so compiling a transformer raised `NotImplementedError` at the first standalone `rms_norm`. It now measures such an operation against itself, records the single benchmark, and notes in the selection reason that the kernel was the only valid candidate.

### Cutting 245 Operators Out Of A Decode Step

A decode step of this model was 1239 operations — 147 casts, 338 reshapes, 168 slices, 49 multiplies, 49 normalizations — and the GPU work in it took 2.5 ms of a 5.7 ms step. The rest was the gap between launches, about 4.6 µs each, across 24 layers of roughly ten strictly dependent launches: a chain that deep cannot be hidden by a host thread that queues each step in microseconds, because the GPU sits idle between commands. Cutting that count turned out to be worth more than making any single kernel faster.

The largest single win was in how PyTorch spells RMSNorm. Written out, it is `(x.to(f32).pow(2).mean().rsqrt() * x.to(f32) * weight.to(f32)).to(x.dtype)`, which the importer faithfully records as five operations: a widen cast, a normalization without a weight, a second widen cast for the weight, a multiply, and a narrow cast. The compiler had a weighted `rms_norm` for exactly this shape since it lowers to the same Metal kernel, but nothing matched the five-op spelling to it, so each of the 49 normalizations in the graph became five launches and four full-width temporaries.

`fuse_norm_scale` matches the sequence and emits one `rms_norm` carrying the weight. That is 245 operators gone from the step, the 49 multiplies among them and the rest casts and layout work that only existed to feed them: 1239 ops to 994, and 1191 to 995 on the bf16 model. The fold needs no tolerance because it is bit-identical: both the row and the weight arrive in the dtype the product is narrowed back to, and the fused kernel forms the product in fp32 and rounds once, which is exactly what the five-op chain did. The pass checks that precondition and declines when it does not hold — a graph that really wants a standalone fp32 row keeps one — and `tests/test_passes_norm.py` asserts bit equality against the unfolded graph rather than a tolerance.

Measured by ablating exactly that pass in one interleaved session — off, on, off, on — so every pair below shares the machine's thermal and load state:

| | bf16 | int4 g64 |
| --- | ---: | ---: |
| graph ops per decode step | 1191 → 995 | 1239 → 994 |
| decode, synchronized each step | 155.4 → 167.2 tok/s (6.44 → 5.98 ms) | 179.0 → 228.3 tok/s (5.59 → 4.38 ms) |
| decode, asynchronous | 199.2 → 232.2 tok/s (5.02 → 4.31 ms) | 195.3 → 272.8 tok/s (5.12 → 3.67 ms) |
| prefill, 128 tokens | 11 208 → 12 298 tok/s | 9 494 → 10 375 tok/s |
| peak memory | 4.26 GiB, unchanged | 1.99 → 1.89 GiB |

The before/after pairs are what the ratios rest on, and they are 1.08x and 1.17x for bf16, 1.28x and 1.40x for int4, and 1.09-1.10x on prefill, for 245 fewer operators and no kernel change. Async still leads sync after the fold — 4.31 ms against 5.98 — so the host loop is still hidden behind the queue; the fold did not move the launch floor into view. The step's cost per operator reads 4.51 µs before and 4.41 µs after, which is the shape of the finding: the time is the operator count times a constant, and the constant did not move when the count did.

The same pass exposed a second problem worth documenting because the fix is scoped rather than general. `_match_rms_norm_consumers` folds a normalization into the *prologue* of the projection that reads it, so the GEMV scales its input rows as it loads them. That is free for a packed projection, whose prologue is generated Metal. It is not free for a dense one, which falls back to MLX and would re-derive the normalization once per reader — slower than computing it once. The matcher now absorbs a normalization only when the projection carries a packed weight, and leaves dense graphs alone.


### The KV-Cache Roll Is One Operator

A decode step writes one key and one value per layer into a fixed-capacity window, and the spellings torch has for that are all longer than the fact. The adapter slices the oldest slot off the layer's pair, stacks this step's two new rows, concatenates them onto the window, then slices the pair apart again and squeezes both halves because attention wants them separate. That is eleven operators per layer, plus the three views — slice, reshape, transpose — that read the value back out of the projection row it was just computed in. Across 24 layers it was 312 of the step's 995 operators, a third of the graph, for a statement that fits in one line: slide the window down a slot, write the new key and value into the freed slot, hand back the window and its two halves.

`kv_cache_append` is that line. It takes the whole layer-stacked cache, this step's key, and the packed projection row, and returns the new pair plus both windows. Its contract in `ir/ops/kernels.py` is where the meaning is written down: the cache is `[layers, 2, batch, heads, slots, dim]` with the pair on axis 1, `layer` says which slice of it this step owns, and `value_start` says where in the projection row the value half begins — the operator reads it there, so no shaped copy of it ever has to exist. `fuse_kv_cache_update` in `passes/cache.py` is the rewrite, and it is deliberately narrow: the concatenate has to build a pair exactly one slot longer than a window that is a front-slice of the *same* pair, the two halves have to be unpacked by a slice and a squeeze with exactly the shapes the operator produces, and nothing the rewrite deletes may be read by an operator outside the chain. A window that drops two slots, a value that is not the projection's, a pair with a third consumer — each keeps its general form. It runs first in the pipeline, because later passes would happily tidy the very views it has to see.

The backend emits one kernel per layer. Per `(half, batch, head)` the slots are one contiguous run, so dropping the oldest and appending one is that run read one row further along: every slot but the last is a vector copy from the source, and the last slot is picked from the key tensor or from the projection row. Four elements per thread is eight bytes for a bf16 window, and the only branch is the one on the final row. It is still a copy of the whole window — under a hundred kilobytes per layer — so the bytes moved are unchanged and the win is entirely the launches that disappeared. The MLX fallback in `runtime/mlx_ops` computes the same three outputs for autotuning and for shapes the kernel declines, and the unrolled plan runner learned to bind multi-output operators so neither path is interpreted.

Measured the same way as the section above — the pass off, on, off, on, in one session — on the 0.5B model, bf16 and int4:

| | bf16 | int4 g64 |
| --- | ---: | ---: |
| graph ops per decode step | 995 → 683 | 994 → 682 |
| decode, synchronized each step | 147.9 → 154.6 tok/s (6.76 → 6.47 ms) | 228.2 → 251.5 tok/s (4.38 → 3.98 ms) |
| decode, asynchronous | 178.8 → 188.5 tok/s (5.59 → 5.30 ms) | 277.2 → 308.7 tok/s (3.61 → 3.24 ms) |
| greedy tokens | unchanged, all 64 | unchanged, all 64 |

A third of the graph bought 1.045x and 1.054x on bf16 and 1.10x and 1.11x on int4, not a third of the throughput, and that is the honest reading: these were the cheapest operators in the step, each moving a window smaller than the smallest projection row. The int4 ratios are larger because its step is shorter, so the same 312 launches were a larger share of it. What this settles is that the cost really is per launch — an operator that copies 90 kB costs about as much as one that moves 2 MB, and removing 312 of them is worth roughly what the launch floor predicts. The obvious next move is the one this rewrite sets up but does not take: with the write indexed by a slot counter instead of by position, the window would not have to shift at all, and the kernel would be a single row store. That changes what attention reads, so it belongs with the attention kernel rather than here.

### Command Line And Artifacts

The installed `aoto` command compiles a callable, module, or HuggingFace model. It can also inspect or execute the result without importing the original model.

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
```

`inputs.json` contains positional tensor specifications. Scalars use an empty shape and a one-element data list:

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

Use repeated `--input '<json>'` arguments instead of `--input-file` when convenient. `--callable MODULE:ATTR` traces a function, `--module MODULE:ATTR` accepts an `nn.Module`, class, or factory, and `--hf-model` loads a HuggingFace causal language model. `python -m aotolith` provides the same commands when the console script is not on `PATH`.

A tensor keyword argument becomes a graph input, written here as `NAME=<json spec>`: `aoto compile ... --forward-kwarg position_ids='{"shape":[1,4],"dtype":"int64","data":[[0,1,2,3]]}'` and `aoto run ... --kwarg position_ids=...`. `aoto inspect` lists them under `entrypoint.kwargs`.

`aoto bench` is the regression gate for the compiler itself. It measures a fixed set of shapes — a projection, a gated MLP, an RMSNorm, causal attention, an int4 GEMV, and the Qwen2 prefill and decode adapters — and can compare the result against a recorded baseline:

```bash
aoto bench --baseline benchmarks/baselines/apple-m5-pro.json --output perf-report.json
aoto bench --update-baseline benchmarks/baselines/apple-m5-pro.json
```

The gate has two halves. `input_ops` and `lowered_ops` are deterministic, so they are compared exactly on any machine and a pass that adds a copy or drops a fusion fails with no timing noise. Kernel counts and milliseconds depend on the GPU, so they are compared only when the baseline names the same device and both reports have at least three repeats; otherwise the report still carries a `ratio` of each workload against one raw MLX matmul measured in the same process. CI runs the gate against the committed baseline, where the paravirtual runner compares the op counts and skips the millisecond half.

Two measurement choices are what make a rerun comparable. Every workload is compiled and executed once before any timing starts, because the first plan a process runs pays for the Metal kernel compilation of all its kernels — that systematically doubled the first workload's numbers even through its warmups. And the threshold compares `min_ms` rather than the reported median: contention only ever adds time, so three runs of an unchanged 0.35 ms projection gave medians of 1.14, 0.38, and 0.36 ms while the minimums stayed at 0.38, 0.35, and 0.34. A real slowdown raises the floor.

Artifacts use format version 5 and ABI version 1.3. Each graph, kernel table, selection table, storage plan, and state tensor has a SHA-256 digest. Loading validates the ABI, target, graph, entrypoint layout, state shape/dtype, serialized kernels, rebuilt selections, and storage plan. Kernel objects are rebuilt from the validated graph and persisted selection, so the artifact does not need the original Python model and loading does not repeat autotuning.

Strict GPU matching is optional. `--require-device-match` or `require_device_match=True` rejects an artifact compiled for another Apple GPU model; the default allows execution on another Apple Silicon device after rebuilding kernels.

## Native Metal Code Generation

Pointwise operations, softmax, reductions, LayerNorm, and RMSNorm generate Metal kernels by default. Matmul and attention use MLX unless native code generation is enabled:

```python
compiled = aoto.compile_callable(
    fn,
    example_inputs,
    codegen_matmul=True,
    codegen_attention=True,
)
```

The generated tensor-core kernels beat MLX on matmul but still trail MLX's SDPA on attention, so the two options are opt-in for different reasons:

| Benchmark | Aotolith | MLX | Ratio |
|---|---|---|---|
| matmul 512x2048x512 f32 | 367.4 us | 118.1 us | 0.32x |
| matmul 512x2048x512 f16 | 60.9 us | 80.9 us | 1.33x |
| matmul 512x2048x512 bf16 | 62.4 us | 82.2 us | 1.32x |
| matmul 1024x4864x896 f16 | 373.0 us | 519.3 us | 1.39x |
| matmul 4096x4096x4096 f16 | 5.53 ms | 5.66 ms | 1.02x |
| matmul 128x896x896 bf16 (prefill) | 18.4 us | 16.5 us | 0.89x |
| matmul 128x896x4864 bf16 (prefill) | 53.5 us | 54.3 us | 1.02x |
| matmul 128x4864x896 bf16 (prefill) | 56.9 us | 83.6 us | 1.47x |
| attention 1x8x512x64 f16 causal | 34.2 us | 28.5 us | 0.83x |
| attention 1x8x2048x128 f16 causal | 530.8 us | 466.3 us | 0.88x |
| attention 1x64x4096x128 f16 causal | 14.8 ms | 13.3 ms | 0.90x |

The matmul kernel drives MPP tensor-core `matmul2d` over tiled operands. Tiles are chosen from the output shape and the right operand layout: a transposed right operand is K-contiguous and rewards wide column blocks, while a row-major right operand strides through K and instead rewards a taller reduction step. Across 43 measured shape and layout combinations the selector stays within 5% of the fastest tile measured for that shape, and against MLX it averages 1.16x (transposed right operand) and 1.10x (row-major) with no shape below 1.01x. The largest shapes converge on parity because both implementations approach the same tensor-core throughput. The three 128-row rows above are a 128-token prefill at Qwen2.5-0.5B's dimensions, and they are the honest exception: the narrowest one trails MLX by 11%, which over a whole prefill (one `o_proj` per layer, so 24 of them at about 3 us each) costs roughly 1% — the measured prefill table below shows no difference from enabling the native matmul at all — while the projection that dominates prefill, `128x4864x896`, is 1.47x ahead.

The f32 row is the exception, and it is deliberate. The hardware has no float32 tensor cores: a `T = float` `matmul2d` runs at bfloat16 precision, which put that shape 1.6e-1 away from PyTorch — exactly where MLX's own f32 matmul lands. f32 therefore stays on the scalar tiled kernel, which matches PyTorch bit for bit and costs the 3.1x above. Half and bfloat16 are unaffected.

The attention kernel is a register-resident FlashAttention pass: one simdgroup owns a 16-row query fragment and the full value width, so the online-softmax accumulators never leave registers, `Q` is staged once in threadgroup memory, and `K`/`V` stream straight from device memory. It reaches 18.6 TFLOPS on long sequences, and MLX's SDPA is 10-17% ahead on prefill shapes. `examples/metal_attention_benchmark.py --ablate` prices the remaining gap: removing the `QK` matmul takes 42-45% off the pass and removing `PV` takes 33-43%, so the two contractions are 88% of the time and the device operand loads are already hidden behind them — pointing `K`/`V` at one resident head instead of their own changes the total by under 3%. The gap is therefore MMA utilisation, not streaming, and the kernels tried against it measured worse: staging `K`/`V` in threadgroup memory costs 1.9x-2.6x depending on the threadgroup shape, and the loss shrinks as more warps share one tile, which is the signature of occupancy — the tile is what limits resident threadgroups. Holding the query fragments in registers instead costs 20% to spills, and widening the `matmul2d` reduction from 16 to 32 halves costs 2.7x. What did help was the operand address arithmetic: `K`/`V` rows are clamped to the last key on every element, and the clamp is provably the identity whenever the key length and query tile are multiples of the 32-key tile, so it is only emitted when a tile can actually run past the end. Dropping it where it cannot fire measured 1.02x-1.13x, and the ratios above are from after it. Decode is a different problem: a one-row query has no use for tiled online softmax, so the backend emits a separate single-query kernel that keeps a whole head's scores in threadgroup memory, reduces `max` and `sum(exp)` in two levels, and accumulates values without rescaling. That kernel is why `--codegen-attention` no longer costs decode throughput: it is one launch per layer over a 192-slot cache and lands within noise of MLX's SDPA. `CompiledPlan.selections` records which path each operation used and why a fallback occurred. Both kernels only cover FP16/BF16; anything else selects MLX.

Tensor cores need Apple GPU family 9 or later, so both tensor-core paths are gated on the compiler's own `__HAVE_TENSOR__` macro. Everything before that — M1, M2, and anything virtualised — takes the simdgroup path instead, which is the same code the fallback tests check against PyTorch, rather than a kernel that fails to load. The macro alone is not enough, because a paravirtualised GPU (hosted CI, cloud Macs) defines it and still rejects the metallib with `unsupported deferred-static-alloca-size function body`; that reached users as an opaque `RuntimeError` on their first matmul. `compile` therefore loads and evaluates one emitted tensor-core kernel before choosing a path, and re-lowers the whole graph without tensor cores if the device refuses, recording the reason in `plan.selections`. That costs one launch (about 0.25 ms once the kernel is cached) and only when a tensor-core kernel was emitted.


## Supported Surface

| Area | Coverage |
|---|---|
| Models | Decoder-only Transformer forward pass; tiny Llama/Qwen-style models; Qwen2/Qwen2.5 single-request static KV-cache prefill/decode, including prompt-length buckets with left padding and weight-only int4 streaming; Qwen2.5-0.5B-Instruct validated |
| Matrix operations | `mm`, `bmm`, `addmm`, `baddbmm`, `addbmm` (batch products summed into one matrix), `mv`/`addmv`, including supported batch broadcasting |
| Linear algebra | `solve`, `solve_triangular`, `inv`, `det`, `slogdet`, `cholesky`, `cholesky_solve`, `cholesky_inverse`, `eigh`, `svd`, `qr`, `pinv`, `lu_factor` and `matrix_exp`, each lowered per selected output; `cdist` for every norm order |
| Indexing | Embedding, `index_select`, advanced indexing, `gather`, `scatter`, `scatter_add`, `index_add`, `masked_scatter`, and the `scatter_reduce` reductions `sum`/`prod`/`amax`/`amin`/`mean` |
| Transformer operations | Scaled dot-product attention, decomposed causal attention, softmax, log_softmax, logsumexp, sum/mean/prod/amax/amin/nansum/nanmean/var/std/var_mean/std_mean/count_nonzero reductions, `all`/`any`, `vector_norm`, `dist`, `F.normalize` and `F.cosine_similarity`, cumsum/cumprod, LayerNorm (including mean/rstd), RMSNorm, SiLU, GELU, SwiGLU, eval-mode RReLU, and other common pointwise operations including `ldexp` |
| Spatial operations | `conv1d`/`conv2d`/`conv3d` with stride, padding, dilation and groups; `max_pool`/`avg_pool`/`adaptive_max_pool`/`adaptive_avg_pool` in 1D, 2D and 3D, including `ceil_mode`, `count_include_pad`, `divisor_override` and the pooled indices; BatchNorm, GroupNorm and InstanceNorm decomposed into moments plus a per-channel affine |
| Resizing | `interpolate`/`upsample` in nearest, linear, bilinear, trilinear and bicubic modes, with `align_corners` and fractional scale factors; antialiased resizing is refused by name |
| Ordering and selection | `argmax`, `argmin`, `max`/`min` with `dim` (values/indices), `median`, `sort` (values/indices), `kthvalue` (values/indices), `topk` (values/indices, largest or smallest), each lowered per selected output |
| Layout and data movement | View/reshape (including a rank-0 target), transpose/permute, slice, `split`/`chunk`/`unbind` (including the elided-dimension spelling), `squeeze` (including the all-axes form), expand, `repeat`/`tile`, `diagonal` and `unfold` as strided views, cat/stack, constant/reflect/replicate padding, flip/roll, clone, casts, `eye`, `empty` (materialized as zeros, since Torch leaves its contents unspecified), and static constants |
| Dtypes | FP16, BF16, FP32, bool, I32, and I64 where supported by the operation |
| KV cache | Static-shape prefill/decode graph contracts and layout; allocation and paging remain external |

Operator coverage is intentionally narrower than PyTorch. Unsupported targets fail during capture or IR construction rather than silently executing outside the plan. See [docs/architecture.md](docs/architecture.md) for the detailed lowering table (currently in Chinese).

## Known Limitations

- Shapes and dtypes are fixed at compile time. Dynamic shapes, symbolic dimensions, and control flow are not supported.
- Tensor-valued `example_kwargs` are promoted to graph inputs and passed by name. Promotion is recursive: tensors nested inside lists, tuples and mappings each become a graph input, so `position_embeddings=(cos, sin)` is one keyword over two inputs and `past_key_values=[(k, v), ...]` works the same way; callers pass the traced nesting back. Non-tensor leaves inside such a value stay tracing-time constants, and passing a different value for one is an error rather than a silent substitution.
- `F.embedding` with `max_norm` is refused by name. It renormalizes the rows the index selects *in place*, and a plan never mutates an input or a bound weight, so the gather would read the table as it was compiled - plausible numbers that ignore the norm. Renormalize the weight once outside the traced function and compile with `max_norm=None`. This is caught because the "nothing reads this node" pruning only applies to pure operators: a mutable target nothing reads still reaches legalization, where the write is either accepted as a rebinding or refused.
- A `None` leaf in a module's output is dropped rather than refused. `nn.MultiheadAttention(need_weights=False)` returns `(output, None)`, and the plan returns the output alone; the flattened leaves are the tensors the module produced.
- Tied parameters are bound once: a tensor reachable under two names shares one MLX buffer and one artifact payload. The graph still has an input per name, because that is what tracing produced. Views that share storage without sharing contents (a tensor and its `.t()`) are kept separate.
- Non-empty constant tensors are bound instead of rejected. `compile_module` promotes every tensor that tracing captured as a graph constant - a `torch.tensor([...])` written inside `forward`, or a tensor kept on the module without `register_buffer` - into the plan's private state, in graph order, visible in `state_names` as `<constant N>` for literals or under the attribute's own name for module tensors. The module is left untouched: a plain attribute is registered only for the duration of capture and restored afterwards, including when capture fails. Artifacts keep those constants. A bare `capture_fx` call still rejects a constant, because it has no state binding to attach one to, and its message now says so.
- The artifact ABI accepts only the `apple-metal` target. Cross-vendor artifacts and host-runtime ABI compatibility are not implemented.
- Advanced indexing runs on MLX as one `index` op over `source` plus every non-`None` index tensor. Adjacent index axes keep PyTorch's rule that the broadcast index dimensions land where those axes were; index tensors separated by a fixed axis get the broadcast dimensions at the front, which the frontend expresses by a `permute` that moves the indexed axes to the front before the same gather, so `x[i, :, j]` and `x[i, j]` share the runtime path. Index tensors broadcast against each other, and the bounds check covers every axis.
- `scatter`, `scatter_add`, `scatter_reduce` and `index_add` execute on MLX rather than in a generated kernel. The reductions cover `sum`/`prod`/`amax`/`amin`/`mean`, each with `include_self=False`; `mean` accumulates a matching write count next to the values and divides by it, so duplicates average instead of summing, and `amax`/`amin` without `include_self` need a floating-point input since their identity is ±inf. Duplicate indices accumulate in the device's order, which is the guarantee CUDA `scatter_add` gives and PyTorch's CPU path does not. The in-place spellings (`scatter_`, `scatter_add_`, `scatter_reduce_`, `index_add_`) lower to the same operations when the mutated tensor has no other observer in the graph; the compiled call returns the updated tensor and never writes back into the tensor the caller still holds.
- `F.pad` in its default `constant` mode lowers to a `pad` layout op and runs on MLX. The padding pairs index the trailing dimensions first, as PyTorch defines them, and the fill value keeps the source dtype. An all-zero pad is erased as an identity, and negative entries crop the padded extent the way ATen does: a crop-only pad becomes a `slice`, a mixed one becomes `slice` then `pad` with the negative widths clamped to zero, and a crop that would empty an axis is rejected because the static IR has no zero-sized dimensions. Any other mode fails during capture.
- `F.pad(..., mode="reflect")` and `mode="replicate"` lower to a view chain with no kernel: each axis takes an inner window and `flip`s it (reflect) or repeats its edge with `expand` (replicate), then `cat` splices it around the source. Negative entries crop the *padded* extent the way ATen does, so the mirror window stays anchored on the uncropped border; padding and cropping on the same axis both work, and the frontend raises when a reflection window is as wide as the axis itself.
- Integer pointwise ops no longer require matching operand dtypes where PyTorch itself would promote. The frontend inserts a `cast` node instead of failing the graph: I32 mixed with I64 becomes I64, integer operands of `div` become the float output dtype, and an integer operand paired with a float becomes that float. `where` promotes its two branches the same way, so `where(mask, int32, int64)` runs as an I64 select. Mixing `bool` with a numeric dtype is still rejected. Integer `pow` is exact: the kernel runs a square-and-multiply loop whose multiplies wrap at the operand width, so `2**70` wraps to `0` exactly as PyTorch's does, and `0**0` is `1`. A negative scalar exponent is rejected at compile time; a negative exponent that only appears in a tensor is not checked per element the way PyTorch checks it at runtime, and the kernel returns `0` for it.
- `solve`, `solve_triangular`, `inv`, `det`, `slogdet`, `cholesky`, `eigh`, `svd`, `qr`, `pinv`, `lu_factor` and `matrix_exp` run on MLX's CPU stream: this build of `mx.linalg` has no GPU implementation and raises "not yet supported on the GPU". The operands are still ordinary MLX arrays, so MLX stages them across and the cost is that trip, not a change in semantics; `matrix_exp`'s series runs there too, because a float32 matrix product on the GPU is far coarser than the CPU's. The factorizations themselves are float32 only, so a half operand is promoted for the call and cast back to the dtype the plan declared, which is what Torch does with its LAPACK path. A decomposition returns a tuple that the graph reads through `getitem`, and each member is its own operation, so `torch.linalg.svd(x)[1]` lowers only `svd_values`, and the `_linalg_check_errors` node Torch inserts afterwards is dropped rather than computed. `solve` reads its right-hand side the way Torch does - a 1-D operand, or one whose shape is exactly the matrix's batch prefix plus its equation axis, is one vector and returns one dimension less, anything else is a stack of columns - and broadcasts only the batch axes, because broadcasting the operands whole would line the vector up against the matrix's own columns. MLX reports LU pivots zero-based where ATen reports them one-based, and returns full singular-vector matrices where a reduced factorization wants the leading `min(m, n)` rows or columns, so both are translated at the call. The decompositions that are only defined up to a sign (`eigh`, `svd`, `qr`, `lu_factor`) reconstruct the operand rather than matching element by element; the tests assert exactly that.
- `dot`, `mv` and `trace` are compositions of existing ops, not new kernels: `dot` is `mul` plus a full-axis `sum`, `mv` reshapes the vector to `(n, 1)` and runs the existing `matmul`, and `trace` reuses the strided `diagonal` view (flatten, then step by `columns + 1`) followed by a `sum`, so a rectangular input sums the shorter diagonal exactly as ATen does. `round` also keeps its `decimals` argument: the kernel scales by `10**decimals`, rounds ties-to-even, and scales back, which is how ATen implements it without a decimal floating-point type.
- `eye`, `ldexp`, `var_mean`, `std_mean`, `dist` and `addbmm` are compositions too. `eye` is the equality of two `arange`s cast to the requested dtype, which is how Torch's own reference implementation writes it. `ldexp(x, e)` is `x * 2**e` with the integer exponent cast to the result dtype, exact for every exponent a float can hold, so only the product rounds. `var_mean`/`std_mean` become the existing `var`/`std` plus a `mean` over the same axes and correction, and a graph that reads one member does not emit the other. `dist(a, b, p)` is `sub` followed by a full-tensor `vector_norm(ord=p)` and always answers a rank-0 scalar. `addbmm` is a batched `matmul`, a `sum` over the batch axis, and the two scale factors folded into pointwise multiplies; a factor of zero drops its term rather than multiplying by it, because ATen ignores that operand and multiplying would propagate an inf or NaN out of it, and both factors being zero answers zeros. `cholesky_solve` and `cholesky_inverse` stay in the same solving layer: `A⁻¹B` with `A = LLᵀ` (or `UᵀU` when `upper=True`) is two `solve_triangular` calls, and the inverse is the same pair against an identity built from two `arange`s like `eye`. `cdist` broadcasts the two batch prefixes, grows a point axis on each operand, and reduces the feature axis with `vector_norm`, so orders 0, 1, 2, 3 and `inf` share one lowering and the `P × R × M` difference tensor is materialized exactly as ATen's non-matmul path does. `masked_scatter` is `cumsum` over the mask, minus one, clamped to the source length, gathered from the flattened source and selected with `where`; the source may be longer than the mask needs, and a short source is not caught at compile time because ATen only checks it while running. `RReLU` in eval mode is a fixed slope, ATen's midpoint `(lower + upper) / 2`, so `rrelu_with_noise` folds into `leaky_relu` and the noise allocation it carries has no reader; training mode samples a slope per element and is refused by name. `slice` fills in ATen's defaults for `dim`/`start`/`end`/`step`, because tracing only emits the arguments the caller actually passed - `cumulative_trapezoid` sends `(self, dim, start)`.
- `logit`, `erfc`, `frac`, `sinc`, `hypot`, `copysign`, `xlogy` and `nan_to_num` fuse into pointwise kernels. `logit` is `log(x / (1 - x))` and clamps to `[eps, 1 - eps]` when the optional `eps` is given. `erfc` reuses the `erf` approximation's polynomial as a tail term, because MSL has no `erfc` in the `metal` namespace and `1 - erf(x)` would cancel away the tail's digits; `hypot` and `log1p` are likewise emitted unqualified since MSL declares them globally. `frac` is `x - trunc(x)`, so it keeps the input's sign instead of using the floored `fract`. `xlogy` short-circuits only when the left operand is zero and the right one is not NaN, matching ATen, so `xlogy(0, -1)` is `0` while `xlogy(0, nan)` stays NaN. `sinc` uses a small-argument series below `1e-4` because the GPU's `sin` can flush such an argument to zero, which would turn `sinc(1e-8)` into `0` instead of ATen's `1`. `nan_to_num` keeps its `nan`/`posinf`/`neginf` replacements and fills the defaults from the dtype's finite limit (65504 for F16, the `max` for BF16/F32); an integer tensor has no non-finite value to replace and lowering returns it unchanged. These operators take floating-point tensors only, and an integer input gets the same `cast` to the output dtype that PyTorch's own opmath promotion performs.
- `floor_divide`, `remainder` and `fmod` accept floating-point operands as well as integers and fuse into the surrounding pointwise kernel. `floor_divide` is `floor(a / b)`; `remainder` follows ATen's `fmod` plus a same-divisor adjustment, so its result always shares the divisor's sign, and `fmod` keeps the dividend's sign. `lerp` covers both the tensor and the scalar `weight` spelling and reproduces ATen's two-branch form (`|weight| < 0.5` uses `start + weight * (end - start)`, otherwise `end - (end - start) * (1 - weight)`) so the branch that avoids cancellation is the one that runs. `addcmul`/`addcdiv` keep their `value` argument, and `addcdiv` evaluates `value * t1 / t2` left to right like ATen, because grouping the division differently rounds differently. `lerp`, `addcmul` and `addcdiv` accept floating-point tensors only; the integer spellings are rejected during lowering.
- `prod`, `amax`, `amin` and `log_softmax` execute on MLX rather than in a generated kernel. The three reductions accept any non-empty axis set plus `keepdim`; `amax`/`amin` match PyTorch bit for bit, `prod` can differ by a few ulps on floats because it multiplies in a different order, and integer products wrap at the operand width. `log_softmax` is the input minus its own log-sum-exp, takes floating-point inputs only, and rejects a repeated axis during lowering.
- Convolutions and pooling run on MLX primitives rather than in generated kernels. A `conv` is a channels-first to channels-last transpose, `mx.conv2d` (or its 1D/3D spelling), and the transpose back, with the bias as a broadcast add. MLX groups 1D and 2D convolutions only, so a grouped `conv3d` runs one plain convolution per group and concatenates the pieces along the channel axis, which is the group-major order Torch returns. A pooling window is a padded strided read: the input is padded once with the reduction's identity (`-inf` for a maximum, zero for a sum) plus any ceil-mode overhang, and the reduction runs over the window axes. Average pooling divides by the same read over a mask of ones, which is what makes `count_include_pad=False` and `divisor_override` cheap, and the pooled indices come from that one read too. Adaptive pooling has no fixed stride, so its windows are unrolled per output position. Dilated windows are materialized with `mx.contiguous` before the reduction: MLX reduces a strided view with non-unit inner offsets incorrectly. `batch_norm`, `group_norm` and `instance_norm` are decompositions, not kernels, and their training-mode running-statistics update is rejected because a plan's state inputs are immutable.
- `native_layer_norm` exposes all three outputs, and they come out of one kernel: an IR operation may write several values, so the statistics are the same reduction's other outputs rather than a second pass. The kernel already computes the row mean and variance twice over; the thread that owns a row stores them alongside the normalized value, so reading `mean` or `rstd` costs two stores, not another reduction. A graph that reads only `rstd` still runs the normalization kernel, because that kernel is what reduces the moments, and leaves the normalized value unused. The statistics match PyTorch's internal Welford statistics to within a few ulps (measured `1e-5` on the f32 tests).
- `logsumexp` is a composite, not a new op: `amax` (keepdim) shifts the input for stability, `masked_fill` replaces an infinite maximum with zero before the subtraction, and `sum`/`log`/`add` finish the reduction. That follows ATen's own implementation, so a `+inf` row stays `+inf` and an all `-inf` row stays `-inf` instead of turning into `nan`; `sum`/`exp`/`log` and the shift all fuse into the surrounding pointwise region.
- `cumprod` runs on MLX like `cumsum`, with the same single `dim` attribute and dtype preservation. `diagonal` becomes a strided view: the two axes are permuted to the back, the pair is flattened, and the diagonal is every `columns + 1`-th element of that flat axis, so `torch.diagonal` and `torch.diag` on a matrix copy nothing and launch no kernel. An offset that leaves no elements is rejected because the static IR has no zero-sized dimensions. `masked_fill` with a 0-dimensional tensor value lowers to `where(mask, value, input)`, which is the same select with the operand order flipped.
- `elu` keeps ATen's `alpha`/`scale`/`input_scale`, so `selu` (scale 1.0507, alpha 1.6733) and a hand-written `elu(x, alpha, scale, input_scale)` both compile. Only the negative branch is scaled by `input_scale`, which matches ATen and is asserted directly in the tests.
- `count_nonzero`, `all` and `any` lower as "bool then count" because the IR has no boolean reduction: `count_nonzero` is `ne(0) -> sum` with an I64 result, and `all`/`any` compare that same count against the number of reduced elements or against zero and return BOOL. NaN counts as true, because PyTorch's boolean cast treats it as nonzero, and an operand that is already boolean skips the `ne`, so `count_nonzero(x > 0)` adds only the `sum`. `x.squeeze()` with no axis becomes a single `reshape` that drops every singleton axis; an empty target shape is accepted only when the input has exactly one element, which is the same rank-0 rule `argmax()` already follows for a full reduction.
- `repeat`/`tile` lower to `reshape -> expand -> reshape` with no kernel and no copy: every axis splits into `(repeat, size)` and broadcasts, so a repeat list shorter or longer than the rank is left-padded with ones exactly as PyTorch does, an all-ones repeat disappears, and `repeat=0` is rejected at capture because the IR does not model empty dimensions. `vector_norm` runs on MLX: because `mx.linalg.norm` takes one axis, the reduced axes are permuted to the back and collapsed before the call and the result is reshaped back for `keepdim`, which covers any axis set with one implementation. `F.normalize` is `vector_norm -> clamp_min -> expand -> div`, and half inputs are promoted to float32 for the norm. `F.cosine_similarity` is two `vector_norm` calls plus one fused `dot / max(norm product, eps)` region; its denominator clamp arrives as `clamp_min_`, which legalizes like the other in-place spellings.
- `var` and `std` execute on MLX. All three ATen spellings (`correction`, `dim`, `default`) normalise to `dims`/`keepdim`/`correction`, with `unbiased` folded into `correction`. `correction` may be fractional: MLX's `ddof` is integral, so the backend computes the biased variance and scales it by `n / (n - correction)` over the elements actually reduced, which is how ATen defines it. `correction` at or above the reduced element count yields `inf` just as PyTorch does, and a non-finite correction is rejected at capture. Half inputs are promoted to float32 for the reduction and cast back to the input dtype, which is the accumulation width PyTorch uses.
- `nansum` and `nanmean` are composites too. `nansum` is ATen's own definition, `masked_fill(isnan(x), 0) -> sum`, so an all-NaN slice sums to zero and an integer input skips the mask; a requested `dtype` casts the operand first, which is the width ATen accumulates in. `nanmean` arrives from PyTorch already decomposed into `nansum` divided by a `max(count, 1)` over `x != x`, so it compiles into two reductions plus a fused division and an all-NaN slice stays NaN. That decomposition is also why `sum.dim_IntList` accepts `dim=None` as a full reduction: ATen's decompositions spell it that way instead of omitting the argument.
- `argmax`, `argmin`, `sort` and `topk` execute on MLX rather than in generated kernels, and each selection of a tuple-returning operation is lowered on its own, so a graph that reads only `sort(...).values` never computes the indices. `torch.max`/`torch.min` with a `dim` reuse that machinery: the values selection lowers to the same `amax`/`amin` the dim-less form uses and the indices selection to `argmax`/`argmin`, so `values, indices = x.max(dim=-1)` costs two reductions while reading only `.values` costs one. Index outputs are I64, and `topk` handles both directions: `mx.topk`'s ascending window is reversed to match `torch.topk`'s sorted output, and a full-length `k` falls back to a sort because MLX's selection path does not order a complete window. Ties resolve in MLX's order, so `sort(stable=True)` keeps matching values but not necessarily PyTorch's index order, and `topk(sorted=False)` still returns sorted values, which is a subset of what PyTorch promises. NaN placement matches PyTorch in both directions.
- The RMSNorm fold recognises `F.rms_norm` and the hand-written spellings around it: `pow(x, 2)` or `x * x` for the square, `add_` or out-of-place `add` for eps, `rsqrt` or `sqrt` plus `div` for the scaling, and the weight on either side of its multiply. Anything else falls through to plain elementwise operations instead of failing, so an upstream decomposition change costs a fold, not compilability.
- Native kernel coverage remains narrower than MLX. `codegen_matmul` stays opt-in only because it is still experimental, and it beats MLX across the measured FP16/BF16 matmul shapes. f32 is slower than MLX on purpose: it stays on the exact scalar kernel because the tensor cores have no f32 mode. `codegen_attention` trails MLX's SDPA by 10-17% on prefill shapes but matches it on the single-query decode path, where the dedicated kernel replaces tiled online softmax with a whole-head threadgroup reduction.
- `arange` generates a Metal constant kernel for int32/int64 and for float16/bfloat16/float32. Integer outputs require integer bounds and keep the `long` path, so values above 2**53 stay exact. Float outputs accept any finite bounds: since Metal has no double, the compile step splits `start` and `step` into a float head and the remaining float tail, and the kernel rebuilds `start + index * step` from the fma product residual plus a two-sum, which is the same single rounding PyTorch's double expression performs. Integer steps and steps such as 0.1 or 0.25 match PyTorch bit for bit; other fractional steps stay within one ulp of the sequence, while PyTorch's own CPU kernel rounds a different intermediate and lands one ulp off the exact value.
- Native attention requires rank-3/4 inputs with matching dtypes and supports MHA/GQA/MQA under the documented mask and head-dimension constraints. Other cases fall back to MLX SDPA.
- The Qwen2 static-cache adapter calls each layer's own `mlp.act_fn`, so `silu` (the Qwen2 default), both `gelu` spellings and the rest of the pointwise activation set compile against the model's real computation; the matmul epilogue fold still recognizes `silu` only, so a `gelu` MLP pays one extra pointwise kernel. At construction time the activation is run through the same capture the compiler uses, so an activation it cannot lower (a parameterised `PReLU`, say) is rejected before the model is traced, with a message naming the layer. Sliding-window layers are still rejected: the adapter builds its own masks, and the installed transformers version's own sliding path raises on this cache, so there is no reference to check against.
- The repository does not implement scheduler, continuous batching, preemption, paged KV cache, prefix cache, sampling policies, PD disaggregation, EP/MoE dispatch, EPLB, or serving APIs.
- Training and autograd are out of scope.

- `heaviside` fuses into the pointwise kernel and follows ATen's comparison order: `x == 0` picks the boundary value, then `x > 0` gives 1, everything else 0. That means `-0.0` takes the boundary and a NaN input becomes 0 rather than the boundary. Both floating-point and integer operands are supported, the integer kernel uses integer literals, and the output keeps the operand dtype; boolean operands are rejected.
- Boolean `mul` and `add` are conjunction and disjunction, in one kernel as `&&`/`||`. ATen writes mask combinations that way - `isclose` is two comparisons, a scale and an offset combined with boolean `*` and `&` - so `torch.isclose` now compiles into a single pointwise kernel that keeps ATen's NaN, infinity and tolerance rules (`equal_nan=True` only flips the NaN comparison). The `.Scalar` spellings of `add`, `sub`, `div` and `mul`, and `remainder` with a scalar dividend (`2.0 % x`, which is the `Scalar_Tensor` schema and keeps the divisor's sign), lower exactly like their tensor spellings.
- In-place pointwise spellings are legalized the same way the in-place scatter family is. The write is a rebinding, so a reader of the destination is fine as long as it sits earlier in the graph: Torch's tracer rebinds the mutated name, which leaves every reader of the old value before the write, and a use after `x.relu_()` reads this node's result rather than the tensor it wrote into. Only a reader placed after the write would see a stale value, so those graphs are refused. The spellings covered are `add_`, `sub_`, `mul_`, `div_`, `remainder_`, `bitwise_and_`/`bitwise_or_`/`bitwise_xor_`/`bitwise_not_`, `logical_not_`, the `clamp_` family, the unary kernels ATen keeps in place (`relu_`, `sigmoid_`, `tanh_`, `exp_`, `expm1_`, `log_`, `log1p_`, `neg_`, `abs_`, `sqrt_`, `rsqrt_`, `reciprocal_`, `floor_`, `ceil_`, `round_`, `trunc_`, `erf_`), and `copy_`, which is also how `fill_`, `zero_`, `masked_fill_` and the activations without an in-place kernel (`silu_`, `gelu_`, `mish_`, `hardsigmoid_`, ...) arrive - their decompositions compute the functional form and write it back. `zero_` builds its constant with `full_like`, a reader of the destination that must not count as an observer of the write. The compiled call returns the updated tensor, and the tensor the caller still holds is never written back.

## Testing

```bash
python -m pytest -q
ruff check src tests examples
python examples/tiny_llama_compile.py
```

CI runs exactly that on every push and pull request: `ruff` on Linux, and on an Apple Silicon macOS runner an editable install of the `dev` extra, the suite, the Llama example, and `python -m build`. The hosted runners have a paravirtual GPU that cannot load the tensor-core kernels, so `compile` falls back to simdgroup kernels there and those tests skip themselves — `pytest -rs` prints why, and a green run covers the simdgroup paths only.


Additional examples cover embeddings, MLP and matmul kernels, softmax, reductions, normalization, attention, and static KV-cache generation:

```bash
python examples/tiny_transformer.py
python examples/tiny_transformer_generate.py
python examples/metal_matmul_benchmark.py
python examples/metal_attention_benchmark.py
```

First execution includes MLX kernel JIT compilation and GPU dispatch. Timing from a first run is not a steady-state benchmark.

PyTorch raises `IndexError` for out-of-range embedding and gather indices, while MLX silently wraps or fills them, so the backend reproduces that check. The values live on the device, so validating them costs a sync: checking every operation up front cost six device synchronizations per decode step, and queuing the checks and collapsing them into one reduction brought that to one. A caller that already knows the indices are in range can drop the last one — `plan(..., verify_index_bounds=False)`, or `runtime.decode_token(..., verify_token=False)` — which is what the greedy loop does, because every token it feeds back came from an `argmax` over the vocabulary. The strict default is unchanged and still raises `IndexError`.

## Roadmap

1. Add symbolic shapes and limited control flow without weakening static-shape validation.
2. Expand autotuning from per-op candidate timing to shape-aware backend policies and reproducible benchmark reports.
3. Improve native matmul and attention performance, then broaden model, dtype, operator, and GPU coverage while keeping PyTorch parity as the acceptance gate.

## Repository Layout

```text
src/aotolith/
  capture/          FX tracing only (make_fx, fake mode, decomposition keep set)
  legalize/         FX -> core ops: ATen target tables, per-family lowerings
  ir/               typed SSA IR, plus the per-family operator contract table
  target/           what this GPU can compile: capabilities, tile ladders
  codegen/          MSL emission, one module per operator family
  runtime/          index-bound checks, the MLX fallback, plan execution
  passes/           graph rewrites: matchers, layout, fusion, CSE, and the order they run in
  planning.py       operation legality and the storage plan
  op_sets.py        how each operator executes (pointwise / layout / constant)
  selection.py      native kernel vs MLX fallback
  backends/         the Apple Metal backend, composing codegen/runtime/target
tests/                                IR, lowering, numerical parity, generation, and planner tests
examples/                             Runnable model, kernel, and static KV-cache examples
docs/architecture.md                  Detailed implementation notes and known boundaries
```

`tests/test_architecture.py` gives every module a layer and fails if an import points at the same or a higher one, so a package can be read on its own.

## Contributing

Issues and pull requests are welcome. Changes should include a focused motivation, a reproducible test, and numerical parity evidence when lowering behavior changes.

## License

MIT. See [LICENSE](LICENSE).
