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
prompt_ids = tokenizer.encode("Explain KV cache in one sentence", add_special_tokens=False)[:128]

runtime = compile_qwen2_static_cache(
    model,
    prompt_length=len(prompt_ids),
    max_new_tokens=64,
)
tokens = runtime.greedy_generate(prompt_ids, max_new_tokens=64)
print(tokens.tolist())
```

The adapter fixes batch size, prompt length, and maximum generation length and implements greedy single-request generation only. The host owns the token loop; scheduling, batching, paged KV allocation, and sampling remain outside the compiler. `runtime.save("qwen")` writes separate prefill and decode `.aoto` files, each currently containing its own copy of the model state.

Query, key, and value share one normalized input, so the adapter stores them as a single packed projection per layer and multiplies once. Splitting the result costs two view operations and removes two kernel launches: the 128-row key and value projections were far too small to fill the GPU, and each ran at under a fifth of the bandwidth the fused projection reaches.

A gated MLP has no such packed weight, because `gate` and `up` are multiplied rather than concatenated. `fuse_gate_up_projection` supplies the missing half: it recognises `silu(gate) * up` where both operands trace back to a projection over the same activation, and replaces the pair with one multiply carrying a `silu_mul_paired` epilogue. The kernel streams the gate row and the up row of a decode token's weights in the same loop, and writes `silu(gate) * up` once, so the step reads the same 17.4 MB in one launch instead of two. It only fires for a single-row activation; a taller operand goes to the tiled multiply, which has no paired epilogue, and the projections stay separate. The gate is rounded to BF16 before `silu`, which is the value the separate projection would have written, so the fusion is bit-identical rather than merely close — a version that kept the extra precision in the FP32 accumulator flipped a greedy token on step four of a 24-token run.

The repository benchmark uses the same prompt, precision, and `128 + 64` token settings for PyTorch MPS, MLX, and Aotolith:

```bash
python examples/qwen_static_cache_benchmark.py --output /tmp/qwen-aotolith.json
python examples/qwen_static_cache_benchmark.py \
  --codegen-matmul \
  --mlx-baseline \
  --output /tmp/qwen-aotolith-codegen.json
```

`--mlx-baseline` times `mlx_lm`'s own greedy decode on the same prompt in the same process, so the native number is measured on the machine running the benchmark instead of quoted from elsewhere. On this machine (M5 Pro, MLX 0.32) it reads **195.5-197.2 tokens/s** across runs.

| configuration | prefill | decode, sync each step | decode, async | vs `mlx_lm` |
| --- | --- | --- | --- | --- |
| MLX fallback only | 10 000 tok/s | 119.0 tok/s | **134.6 tok/s** | 0.70x |
| `--codegen-matmul` | 13 000 tok/s | 178.4 tok/s | **240.1 tok/s** | **1.23x** |
| `--codegen-matmul --codegen-attention` | 12 400 tok/s | 176.8 tok/s | 233.0 tok/s | 1.21x |

(Qwen2.5-0.5B-Instruct, 128 prompt tokens, greedy, batch 1, median of 11 repeats, BF16; the same 64 generated tokens as the PyTorch reference in every row. Across runs of this build the `--codegen-matmul` decode lands in 240.0-240.2 tokens/s — that configuration is stable enough to read to a tenth — while the `--codegen-attention` row varies in 229.8-238.2 and its prefill in 11.6k-12.8k. Prefill staying around 13.0k without native attention and dropping to roughly 12.4k with it is the attention kernel's slower tile on prefill shapes, not a decode cost. The `mlx_lm` baseline reads 190.5-195.0 tokens/s across the same runs.)

`decode_async` is the figure to compare against `mlx_lm`, because both amortize the host loop across tokens; `decode_sync_each_step` pays a device sync every token, which no serving loop needs. The async path used to trail the synchronous one and now leads it, because the plan only forces a single device sync per step.

All 64 greedy tokens match the PyTorch bf16 reference exactly, with or without native kernels. `mlx_lm` produces those same 64 tokens, so the throughput comparison above also compares two implementations that agree on the answer. Agreeing on tokens is not the same as agreeing on numbers, and the two paths are not equally close to the reference: on the same model's 24-layer forward, the native kernels land a mean 0.043 / max 0.38 from PyTorch's logits, the MLX fallback a mean 0.081 / max 1.5. So enabling code generation moves the arithmetic toward PyTorch rather than away from it, which makes it a throughput decision and not an accuracy one.

Decode is bound by weight bandwidth, not by the host. Building a step costs 0.98 ms of Python and the GPU needs about 4.1 ms, so the host is fully hidden behind the async queue; before the plan was unrolled the host cost 6.2 ms and *was* the limit. One step streams 991 MB of weights (24 layers of `qkv` 2.07 MB, `o_proj` 1.61 MB, one paired gate/up matrix of 17.4 MB, and `down` at 8.73 MB, plus the 272 MB LM head), which at the measured 4.17 ms per token is 238 GB/s.

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

That sums to 3.96 ms of device time and 991 MB, against 4.17 ms measured for the whole step. The LM head plus the MLP matrices are 74% of it, and they are moving bytes rather than waiting on dispatches; the rest is spread over 121 launches that each move 2 MB or less. A step carries 170 dependent launches. Removing one of the cheap ones does not move the total: folding the query and key rotary calls into one 16-head rotation removed 24 launches and 24 graph ops and decode stayed at 232.9 tokens/s against 232.5, inside the run-to-run band, so it was not kept. Removing one that carries real traffic does move it: pairing `gate` and `up` into one launch reads the same 17.4 MB but takes one launch instead of two, and measured 1.031x-1.038x across interleaved A/B runs. The difference is what the launch moves, not how many there are.

Two ways of producing this table measure nothing at all, and both look plausible. Dropping each round's output arrays frees the lazy graph before it reaches the device, so the wall clock covers host submission only and the bandwidth column comes out above 1 TB/s. Passing all-zero weights lets the driver skip the read, which inflates the small buckets. A bucket has to keep every round's outputs alive and feed real values before its numbers mean anything, and any figure that exceeds what `mx.sum` over the same bytes reaches is measuring the harness rather than the kernel. The replay also flatters the tiny buckets for the opposite reason: the 24 instances of a bucket run back to back there, while in the real step each layer waits for the previous one, so per-launch figures for RoPE and attention are upper bounds and the table's sum only approximates the step.

Two experiments in this area were measured and dropped. Writing the new key/value straight into slot `positions` and selecting the live slots with the mask reads better than rolling the window, but MLX lowers a scatter to an index-compare pass over the whole window where the concatenate is a straight copy of the same bytes: 0.18 ms slower per step, so the roll stays. And the rotary position for a decode token is `valid_length - 1`, not `valid_length`; `Qwen2StaticCacheRuntime.decode_token` documented the second and briefly used it, which cost 0.7 in max logit deviation against the reference and broke the exact-token match. It now matches all 64 tokens.

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

What all three show is that the rate a launch reaches depends on how many bytes it reads, not on how its grid is shaped. To run the whole step at 412 GB/s its 991 MB would have to arrive in launches of hundreds of megabytes, but every layer is cut into five strictly dependent projections — `qkv`, `o_proj`, paired gate/up, `down` — and the largest is 17.4 MB, because the nonlinearity between them is a real barrier. Pairing `gate` and `up` is exactly the move that exploits this, and it is why that one change measured 1.03x while removing 24 launches of 1 KB measured nothing. 4.17 ms per token is therefore the BF16 floor on this device: going meaningfully faster needs fewer bytes, which means quantized weights, not a better kernel.

Autotuning is not an escape either, and until now it could not even run on a model like this one. `--autotune` evaluated the MLX fallback for every operation in order to have a reference to compare against, but normalization, rotary, and attention only ever lower to generated kernels, so compiling a transformer raised `NotImplementedError` at the first standalone `rms_norm`. It now measures such an operation against itself, records the single benchmark, and notes in the selection reason that the kernel was the only valid candidate.

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

Artifacts use format version 4 and ABI version 1.3. Each graph, kernel table, selection table, storage plan, and state tensor has a SHA-256 digest. Loading validates the ABI, target, graph, entrypoint layout, state shape/dtype, serialized kernels, rebuilt selections, and storage plan. Kernel objects are rebuilt from the validated graph and persisted selection, so the artifact does not need the original Python model and loading does not repeat autotuning.

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
| Models | Decoder-only Transformer forward pass; tiny Llama/Qwen-style models; Qwen2/Qwen2.5 single-request static KV-cache prefill/decode; Qwen2.5-0.5B-Instruct validated |
| Matrix operations | `mm`, `bmm`, `addmm`, `baddbmm`, including supported batch broadcasting |
| Indexing | Embedding, `index_select`, advanced indexing, `gather`, `scatter`, `scatter_add`, `index_add`, and the `scatter_reduce` reductions `sum`/`prod`/`amax`/`amin`/`mean` |
| Transformer operations | Scaled dot-product attention, decomposed causal attention, softmax, log_softmax, sum/mean/prod/amax/amin reductions, cumsum, LayerNorm, RMSNorm, SiLU, GELU, SwiGLU, and other common pointwise operations |
| Layout and data movement | View/reshape, transpose/permute, slice, expand, cat/stack, constant padding, flip/roll, clone, casts, and static constants |
| Dtypes | FP16, BF16, FP32, bool, I32, and I64 where supported by the operation |
| KV cache | Static-shape prefill/decode graph contracts and layout; allocation and paging remain external |

Operator coverage is intentionally narrower than PyTorch. Unsupported targets fail during capture or IR construction rather than silently executing outside the plan. See [docs/architecture.md](docs/architecture.md) for the detailed lowering table (currently in Chinese).

## Known Limitations

- Shapes and dtypes are fixed at compile time. Dynamic shapes, symbolic dimensions, and control flow are not supported.
- Tensor-valued `example_kwargs` are promoted to graph inputs and passed by name. Promotion is recursive: tensors nested inside lists, tuples and mappings each become a graph input, so `position_embeddings=(cos, sin)` is one keyword over two inputs and `past_key_values=[(k, v), ...]` works the same way; callers pass the traced nesting back. Non-tensor leaves inside such a value stay tracing-time constants, and passing a different value for one is an error rather than a silent substitution.
- Tied parameters are bound once: a tensor reachable under two names shares one MLX buffer and one artifact payload. The graph still has an input per name, because that is what tracing produced. Views that share storage without sharing contents (a tensor and its `.t()`) are kept separate.
- The artifact ABI accepts only the `apple-metal` target. Cross-vendor artifacts and host-runtime ABI compatibility are not implemented.
- `scatter`, `scatter_add`, `scatter_reduce` and `index_add` execute on MLX rather than in a generated kernel. The reductions cover `sum`/`prod`/`amax`/`amin`/`mean`, each with `include_self=False`; `mean` accumulates a matching write count next to the values and divides by it, so duplicates average instead of summing, and `amax`/`amin` without `include_self` need a floating-point input since their identity is ±inf. Duplicate indices accumulate in the device's order, which is the guarantee CUDA `scatter_add` gives and PyTorch's CPU path does not. The in-place spellings (`scatter_`, `scatter_add_`, `scatter_reduce_`, `index_add_`) lower to the same operations when the mutated tensor has no other observer in the graph; the compiled call returns the updated tensor and never writes back into the tensor the caller still holds.
- `F.pad` in its default `constant` mode lowers to a `pad` layout op and runs on MLX. The padding pairs index the trailing dimensions first, as PyTorch defines them, and the fill value keeps the source dtype. An all-zero pad is erased as an identity, negative entries (which crop in PyTorch) are rejected at capture with a pointer to `slice`/`narrow`, and any other mode fails during capture.
- Integer pointwise ops no longer require matching operand dtypes where PyTorch itself would promote. The frontend inserts a `cast` node instead of failing the graph: I32 mixed with I64 becomes I64, integer operands of `div` become the float output dtype, and an integer operand paired with a float becomes that float. `where` promotes its two branches the same way, so `where(mask, int32, int64)` runs as an I64 select. Mixing `bool` with a numeric dtype is still rejected. Integer `pow` is exact: the kernel runs a square-and-multiply loop whose multiplies wrap at the operand width, so `2**70` wraps to `0` exactly as PyTorch's does, and `0**0` is `1`. A negative scalar exponent is rejected at compile time; a negative exponent that only appears in a tensor is not checked per element the way PyTorch checks it at runtime, and the kernel returns `0` for it.
- `prod`, `amax`, `amin` and `log_softmax` execute on MLX rather than in a generated kernel. The three reductions accept any non-empty axis set plus `keepdim`; `amax`/`amin` match PyTorch bit for bit, `prod` can differ by a few ulps on floats because it multiplies in a different order, and integer products wrap at the operand width. `log_softmax` is the input minus its own log-sum-exp, takes floating-point inputs only, and rejects a repeated axis during lowering.
- Native kernel coverage remains narrower than MLX. `codegen_matmul` stays opt-in only because it is still experimental, and it beats MLX across the measured FP16/BF16 matmul shapes. f32 is slower than MLX on purpose: it stays on the exact scalar kernel because the tensor cores have no f32 mode. `codegen_attention` trails MLX's SDPA by 10-17% on prefill shapes but matches it on the single-query decode path, where the dedicated kernel replaces tiled online softmax with a whole-head threadgroup reduction.
- `arange` generates a Metal constant kernel for int32/int64 and for float16/bfloat16/float32. Integer outputs require integer bounds and keep the `long` path, so values above 2**53 stay exact. Float outputs accept any finite bounds: since Metal has no double, the compile step splits `start` and `step` into a float head and the remaining float tail, and the kernel rebuilds `start + index * step` from the fma product residual plus a two-sum, which is the same single rounding PyTorch's double expression performs. Integer steps and steps such as 0.1 or 0.25 match PyTorch bit for bit; other fractional steps stay within one ulp of the sequence, while PyTorch's own CPU kernel rounds a different intermediate and lands one ulp off the exact value.
- Native attention requires rank-3/4 inputs with matching dtypes and supports MHA/GQA/MQA under the documented mask and head-dimension constraints. Other cases fall back to MLX SDPA.
- The Qwen2 static-cache adapter implements SiLU MLP activation and a plain causal mask only. A config that selects another activation or sliding-window layers is rejected at compile time rather than compiled against the wrong semantics; the check reads the constructed layers rather than the config string, because the library normalises some of those fields away. The adapter verifies the activation numerically, which also catches a custom `act_fn` that the config does not describe.
- The repository does not implement scheduler, continuous batching, preemption, paged KV cache, prefix cache, sampling policies, PD disaggregation, EP/MoE dispatch, EPLB, or serving APIs.
- Training and autograd are out of scope.

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
src/aotolith/   Compiler frontend, IR, passes, planner, selection, and Metal backend
tests/                                IR, lowering, numerical parity, generation, and planner tests
examples/                             Runnable model, kernel, and static KV-cache examples
docs/architecture.md                  Detailed implementation notes and known boundaries
```

## Contributing

Issues and pull requests are welcome. Changes should include a focused motivation, a reproducible test, and numerical parity evidence when lowering behavior changes.

## License

MIT. See [LICENSE](LICENSE).
