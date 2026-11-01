# Aotolith

English | [简体中文](README.zh-CN.md)

**A compiler for Transformer models on Apple Silicon.** Capture a PyTorch model
as an FX graph, lower it through a typed SSA IR, and run it on Metal kernels the
compiler generates itself, with MLX as the fallback and the numerical reference.

[![PyPI](https://img.shields.io/pypi/v/aotolith?logo=pypi&logoColor=white)](https://pypi.org/project/aotolith/)
[![CI](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml/badge.svg)](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#limitations)

```bash
pip install aotolith
```

Most Transformer compilers assume CUDA. This one keeps a backend-independent IR
and takes the whole path to Metal: graph capture, legalization, layout folding,
fusion, CSE, tensor lifetimes, kernel selection, code generation, and a versioned
artifact that reloads in a fresh process.

> [!IMPORTANT]
> This project compiles model computation. Request scheduling, continuous
> batching, paged KV allocation, sampling, and serving belong to the host runtime.

```python
import torch
import aotolith as aoto

def fn(x, weight, bias):
    return torch.nn.functional.silu(torch.addmm(bias, x, weight, alpha=2.0))

inputs = (torch.randn(4, 16), torch.randn(16, 32), torch.randn(32))
compiled = aoto.compile_callable(fn, inputs, codegen_matmul=True)

print(compiled.lowered_graph)       # fused IR
print(compiled.plan.selections)     # native kernel or MLX, and why
```

## Install

```bash
pip install aotolith
```

From source, which is also how the test suite is run:

```bash
git clone https://github.com/jaykay233/aotolith.git
cd aotolith
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Requires macOS on Apple Silicon, Python 3.11+, MLX 0.32, and PyTorch 2.7+ for FX
capture. `.[fx]` installs only what compiling Python callables needs; `.[hf]` adds
Transformers.

## Quickstart

Compile a HuggingFace model. Parameters and buffers become plan state, and the
plan does not retain the module:

```python
from transformers import LlamaConfig, LlamaForCausalLM

import torch
import aotolith as aoto

model = LlamaForCausalLM(LlamaConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=16,
)).eval()

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
compiled = aoto.compile_module(
    model, (input_ids,), codegen_matmul=True, codegen_attention=True
)
logits = compiled(input_ids)[0]
```

Compile a Qwen2.5 static KV cache and generate greedily:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aotolith.models.qwen2 import compile_qwen2_static_cache

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
prompt_ids = tokenizer.encode("Explain KV cache in one sentence")[:128]

runtime = compile_qwen2_static_cache(
    model, prompt_length=len(prompt_ids), max_new_tokens=64, quantize=64
)
tokens = runtime.greedy_generate(prompt_ids, max_new_tokens=64)
```

The adapter fixes batch size, prompt length, and maximum generation length, and
implements greedy single-request generation only. `compile_qwen2_bucketed`
compiles one plan pair per prompt-length bucket and left-pads a shorter prompt
into the smallest one that fits.

The same static-cache runtime covers a second family. `aotolith.models.llama`
exports `compile_llama_static_cache` and `compile_llama_bucketed`, which differ
from the Qwen2 entry points only in what they insist on at compile time —

```python
from aotolith.models.llama import compile_llama_static_cache

runtime = compile_llama_static_cache(
    model, prompt_length=128, max_new_tokens=64, quantize=64
)
tokens = runtime.greedy_generate(prompt_ids)
```

— because the two families disagree about three things and agree about
everything else: Llama's q/k/v/o projections carry no bias, so the packed
projection kernel has no epilogue fold to apply; `num_key_value_heads` can equal
`num_attention_heads` as well as divide it; and the RoPE base is a config value
(`rope_theta`) rather than a constant. `check_llama_model` and
`check_qwen2_model` refuse the other family instead of silently compiling a
graph whose projections are shaped for a different checkpoint.

Both backends are reachable from one entry point. `target="mlx-reference"`
compiles the same graph with no kernels at all — every node runs on the MLX
interpreter — which is how a plan can be built and verified on a host that has
no Metal, and how the same lowered graph gets a second, independent execution to
compare shapes, dtypes, cache contents and token decisions against.

The same idea is available for any module: `compile_bucketed(model,
example_inputs, axis=0, sizes=(1, 4, 8))` compiles one plan per width and
dispatches a request to the smallest bucket that holds it, sharing one copy of
the weights across the set and accounting for the worst bucket's memory before a
request runs.

## Results

Qwen2.5-0.5B-Instruct, 128 prompt tokens, 64 greedy tokens, BF16, batch 1, on an
M5 Pro. `mlx_lm` measured 195.5-197.2 tok/s in the same process:

| configuration | prefill | decode, async | vs `mlx_lm` |
| --- | --- | --- | --- |
| MLX fallback only | 9 500 tok/s | 174.8 tok/s | 0.90x |
| `--codegen-matmul` | 11 800 tok/s | **231.7 tok/s** | **1.21x** |
| `--codegen-matmul --codegen-attention` | 12 400 tok/s | 232.3 tok/s | 1.20x |

All 64 tokens match the PyTorch BF16 reference exactly, with or without native
kernels, and `mlx_lm` produces those same tokens.

Agreeing on tokens is not agreeing on numbers, and the gap depends on how many
tokens the prefill sees — `examples/qwen_logit_distance.py` measures it:

| prompt tokens | fallback mean | fallback max | codegen mean | codegen max |
| --- | --- | --- | --- | --- |
| 7 | 0.1056 | 0.6875 | 0.0290 | 0.1875 |
| 8 | 0.0783 | 0.5469 | 0.0451 | 0.2500 |
| 16 | 0.0312 | 0.2031 | 0.0310 | 0.1953 |
| 128 | 0.0596 | 0.4219 | 0.0566 | 0.4219 |

Read down a column: the fallback's mean distance moves by 3.4x between a 7- and a
16-token prompt, so a single figure describes the prompt as much as the model.
What the table does say is that code generation is never further from PyTorch at
these lengths — 3.6x closer at 7 tokens, level at 16 (0.0312 against 0.0310) — and that
by a 128-token prefill the two paths are within 5%. Enabling code generation is a
throughput decision, not an accuracy one.

Reproduce the tokens with `examples/qwen_static_cache_benchmark.py
--codegen-matmul --mlx-baseline`, and the distances with
`examples/qwen_logit_distance.py`.

## How a step is compiled

```text
PyTorch FX / nn.Module
  → typed SSA IR        legalize, fold layouts, fuse, eliminate common subexpressions
  → storage plan        live intervals, release points, one arena
  → kernel selection    generated Metal, or the MLX fallback, with a reason per op
  → CompiledPlan        unrolled on first call, savable as a versioned .aoto artifact
```

The first call compiles the plan into a function whose body is one assignment per
operation, with kernel call sites bound as constants. Nothing about the kernels
changes, only the bookkeeping around them: on this model it took the host cost of
a decode step from 6.2 ms to 0.98 ms.

As a smaller end-to-end check, `examples/tiny_llama_compile.py` compiles a
two-layer HuggingFace Llama: 131 captured operations become 79 in prefill (151
to 62 in decode), its logits land `1.043e-07` from the PyTorch reference with
both matmul and attention codegen on — float32 round-off, not accumulated drift
— and the plan still executes after the original module is released.

The last two words matter. With codegen off, the same graph runs on MLX's own
kernels, and the drift becomes `1.3e-3`: MLX's float32 matmul on this GPU is
about `8e-4` relative to a float64 reference, flat in K, which is the shape of a
reduced-precision matmul rather than an accumulation error. So the fallback is a
correctness-safe path and not a bit-accurate one, and the difference is the
runtime's, not the compiler's.

## Native Metal code generation

Pointwise operations, softmax, reductions, LayerNorm, and RMSNorm generate Metal
kernels by default. Matmul and attention are opt-in:

```python
compiled = aoto.compile_callable(fn, inputs, codegen_matmul=True, codegen_attention=True)
```

- `codegen_matmul` drives MPP tensor-core `matmul2d` and beats MLX across the
  measured FP16/BF16 shapes (1.32-1.47x on the large ones), and it lets a dense
  decode projection absorb its RMSNorm. f32 stays on the exact scalar kernel
  because the hardware has no f32 tensor cores.
- `codegen_attention` is a register-resident FlashAttention pass that trails
  MLX's SDPA by 10-17% on prefill and matches it on the single-query decode path,
  where a dedicated whole-head kernel replaces tiled online softmax.
- Tensor cores need Apple GPU family 9+. `compile` loads one emitted tensor-core
  kernel before choosing a path, and re-lowers without them if the device refuses
  — which is what happens on paravirtual GPUs.

## Supported surface

| Area | Coverage |
| --- | --- |
| Models | Decoder-only Transformer forward; Qwen2/Qwen2.5 and Llama static KV-cache prefill/decode with buckets and weight-only int4; Qwen2.5-0.5B-Instruct validated |
| Matrix | `mm`, `bmm`, `addmm`, `baddbmm`, `addbmm`, `mv`/`addmv`, with batch broadcasting |
| Linear algebra | `solve`, `inv`, `det`, `cholesky` and friends, `eigh`, `svd`, `qr`, `pinv`, `lu_factor`, `matrix_exp`, `cdist` |
| Indexing | Embedding, `index_select`, advanced indexing, `gather`, `scatter*`, `index_add`, `masked_scatter` |
| Transformer | SDPA and decomposed causal attention, softmax family, reductions including `var`/`std`/`nansum`, `vector_norm`, cumsum/cumprod, LayerNorm, RMSNorm, SiLU, GELU, SwiGLU |
| Spatial | `conv1d/2d/3d` with stride/padding/dilation/groups, the pooling family, `interpolate`, BatchNorm/GroupNorm/InstanceNorm |
| Selection | `argmax`, `argmin`, `max`/`min` with `dim`, `median`, `sort`, `kthvalue`, `topk` |
| Layout | Views, transpose/permute, slice, split/chunk/unbind, squeeze, expand, `repeat`/`tile`, cat/stack, padding, flip/roll, casts, `arange` |
| Targets | `apple-metal` (generated Metal kernels with MLX fallback) and `mlx-reference` (no kernels, MLX interpreter only) |
| Dtypes | FP16, BF16, FP32, bool, I32, I64 where the operation supports them |

Coverage is intentionally narrower than PyTorch. Unsupported targets fail during
capture or IR construction rather than silently executing outside the plan. How
each family executes — generated Metal, MLX, or a composition — is in
[docs/operators.md](docs/operators.md), with the pass-by-pass detail in
[docs/architecture.md](docs/architecture.md) (Chinese).

## Limitations

- Shapes and dtypes are fixed at compile time. No symbolic dimensions or control
  flow. A request whose width varies is covered by compiling a set of widths
  (`compile_bucketed`, `aoto compile --bucket-sizes`), which serves any width up
  to the largest bucket; the cost is the rows between a request and its bucket.
- An artifact names its target and loads only where that target is available: `apple-metal` artifacts need an Apple GPU (they rebuild their kernels from MSL), `mlx-reference` artifacts need MLX and nothing else. There is no CUDA or CPU target yet.
- Training and autograd are out of scope.
- No scheduler, continuous batching, preemption, paged KV cache, prefix cache,
  sampling policies, PD disaggregation, EP/MoE dispatch, or serving APIs.
  [docs/host-runtime.md](docs/host-runtime.md) records what an embedding host can
  rely on instead of each of them.
- Several operator families run on MLX rather than in generated kernels, and
  `mx.linalg` factorizations run on MLX's CPU stream. The reasons and the exact
  boundaries are recorded per family in [docs/operators.md](docs/operators.md).

## Documentation

- [docs/usage.md](docs/usage.md) — tensor keyword arguments, shape buckets, the
  `aoto` command, and artifact versioning.
- [docs/errors.md](docs/errors.md) — every error code, what it means, what a
  caller should do, and what the package logs ([Chinese](docs/errors.zh-CN.md)).
- [docs/compatibility.md](docs/compatibility.md) — what is promised between
  releases, what counts as breaking, and how a deprecation works
  ([Chinese](docs/compatibility.zh-CN.md)).
- [CHANGELOG.md](CHANGELOG.md) — what changed in each release.
- [docs/operators.md](docs/operators.md) — how each operator family executes and
  where a spelling deliberately differs from PyTorch.
- [docs/performance.md](docs/performance.md) — where a decode step's time goes,
  what each rewrite bought, and which ideas were measured and dropped.
- [docs/host-runtime.md](docs/host-runtime.md) — the boundary between the
  compiler and a serving engine, and what an embedding host may rely on.
- [docs/architecture.md](docs/architecture.md) — IR, passes, and code generation,
  in more depth than the README (Chinese).

## Development

```bash
python -m pytest -q
ruff check src tests examples
python examples/tiny_llama_compile.py
```

CI runs exactly that on every push: `ruff` on Linux, and on an Apple Silicon
runner an editable install, the suite, the Llama example, and `python -m build`.
Hosted runners expose a paravirtual GPU that cannot load tensor-core kernels, so
`compile` falls back to simdgroup kernels there and those tests skip themselves.

`aoto bench` is the compiler's own regression gate: it measures fixed shapes and
compares deterministic op counts exactly, and timings, peak memory and cold start
against a recorded GPU baseline. More examples live in `examples/`.

## Layout

```text
src/aotolith/
  capture/     FX tracing (make_fx, fake mode, decomposition keep set)
  legalize/    FX -> core ops: ATen target tables, per-family lowerings
  ir/          typed SSA IR and the operator contract table
  target/      what this GPU can compile: capabilities, tile ladders
  codegen/     MSL emission, one module per operator family
  runtime/     index checks, MLX fallback, plan execution
  passes/      rewrites: matchers, layout, fusion, CSE, and their order
  planning.py  operation legality and the storage plan
  selection.py native kernel vs MLX fallback
  backends/    the Apple Metal backend and the MLX reference backend
  models/      decoder static-cache adapters: family registry, Qwen2, Llama
tests/         IR, lowering, numerical parity, generation, planner
examples/      runnable models, kernels, and static KV-cache benchmarks
```

## Roadmap

- **Symbolic shapes and limited control flow**, without weakening the strict
  verifier the static path relies on. Request widths already come from compiled
  buckets; what is left is extent that depends on values, and the padding a
  bucket boundary costs.
- **Shape-aware backend policy**, replacing per-op autotune with a selection that
  accounts for the whole graph, and a reproducible benchmark report.
- **Multi-row decode**: one plan that consumes and verifies several tokens per
  launch, which is what speculative decoding needs. `m` stays a compile-time
  constant, so it fits the existing contract instead of adding a dynamic
  dimension.
- **An end-to-end command**: `aoto generate` takes a prompt and prints or streams
  tokens, with a stop condition and one sampler, so a pip install can run a model
  without writing a loop. Scheduling and serving stay with the engine that embeds
  the compiler; see [docs/host-runtime.md](docs/host-runtime.md).
- **Wider native coverage**: split-K for matmul, tile search for attention, and
  generated kernels for the families still on MLX.
- **More model families**: the decoder adapter takes a family, so the next one
  is a config reader plus its quirks — Mistral's sliding window, Qwen3's head
  layout — rather than a second adapter. Long-context variants need RoPE
  scaling, which the position table currently assumes is plain.
- **A third backend** on a different device, which is what would prove the
  `mlx-reference` seam is general rather than MLX-shaped. The ABI already
  carries the target; what is missing is a provider that does not speak MLX.

## Contributing

Issues and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) covers
what a change needs, how to add an operator, and how a release is published. A
change to lowering behaviour needs a stated motivation, a reproducible test, and
numerical parity against PyTorch.

## License

MIT
