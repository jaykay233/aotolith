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

## Results

Qwen2.5-0.5B-Instruct, 128 prompt tokens, 64 greedy tokens, BF16, batch 1, on an
M5 Pro. `mlx_lm` measured 195.5-197.2 tok/s in the same process:

| configuration | prefill | decode, async | vs `mlx_lm` |
| --- | --- | --- | --- |
| MLX fallback only | 9 500 tok/s | 174.8 tok/s | 0.90x |
| `--codegen-matmul` | 11 800 tok/s | **231.7 tok/s** | **1.21x** |
| `--codegen-matmul --codegen-attention` | 12 400 tok/s | 232.3 tok/s | 1.20x |

All 64 tokens match the PyTorch BF16 reference exactly, with or without native
kernels, and `mlx_lm` produces those same tokens. The native path is also closer
to PyTorch's logits than the fallback (mean 0.043 vs 0.081), so enabling code
generation is a throughput decision, not an accuracy one.

Reproduce with `examples/qwen_static_cache_benchmark.py --codegen-matmul
--mlx-baseline`.

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

As a smaller end-to-end check, a two-layer HuggingFace Llama compiles from 220 FX
operations to 136 lowered ones, its logits land `8.941e-08` from the PyTorch
reference, and the plan still executes after the original module is released.

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
| Models | Decoder-only Transformer forward; Qwen2/Qwen2.5 static KV-cache prefill/decode with buckets and weight-only int4; Qwen2.5-0.5B-Instruct validated |
| Matrix | `mm`, `bmm`, `addmm`, `baddbmm`, `addbmm`, `mv`/`addmv`, with batch broadcasting |
| Linear algebra | `solve`, `inv`, `det`, `cholesky` and friends, `eigh`, `svd`, `qr`, `pinv`, `lu_factor`, `matrix_exp`, `cdist` |
| Indexing | Embedding, `index_select`, advanced indexing, `gather`, `scatter*`, `index_add`, `masked_scatter` |
| Transformer | SDPA and decomposed causal attention, softmax family, reductions including `var`/`std`/`nansum`, `vector_norm`, cumsum/cumprod, LayerNorm, RMSNorm, SiLU, GELU, SwiGLU |
| Spatial | `conv1d/2d/3d` with stride/padding/dilation/groups, the pooling family, `interpolate`, BatchNorm/GroupNorm/InstanceNorm |
| Selection | `argmax`, `argmin`, `max`/`min` with `dim`, `median`, `sort`, `kthvalue`, `topk` |
| Layout | Views, transpose/permute, slice, split/chunk/unbind, squeeze, expand, `repeat`/`tile`, cat/stack, padding, flip/roll, casts, `arange` |
| Dtypes | FP16, BF16, FP32, bool, I32, I64 where the operation supports them |

Coverage is intentionally narrower than PyTorch. Unsupported targets fail during
capture or IR construction rather than silently executing outside the plan. How
each family executes — generated Metal, MLX, or a composition — is in
[docs/operators.md](docs/operators.md), with the pass-by-pass detail in
[docs/architecture.md](docs/architecture.md) (Chinese).

## Limitations

- Shapes and dtypes are fixed at compile time. No dynamic shapes, symbolic
  dimensions, or control flow.
- The artifact ABI accepts only the `apple-metal` target.
- Training and autograd are out of scope.
- No scheduler, continuous batching, preemption, paged KV cache, prefix cache,
  sampling policies, PD disaggregation, EP/MoE dispatch, or serving APIs.
- Several operator families run on MLX rather than in generated kernels, and
  `mx.linalg` factorizations run on MLX's CPU stream. The reasons and the exact
  boundaries are recorded per family in [docs/operators.md](docs/operators.md).

## Documentation

- [docs/usage.md](docs/usage.md) — tensor keyword arguments, prompt-length
  buckets, the `aoto` command, and artifact versioning.
- [docs/operators.md](docs/operators.md) — how each operator family executes and
  where a spelling deliberately differs from PyTorch.
- [docs/performance.md](docs/performance.md) — where a decode step's time goes,
  what each rewrite bought, and which ideas were measured and dropped.
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
compares deterministic op counts exactly and timings against a recorded GPU
baseline. More examples live in `examples/`.

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
  backends/    the Apple Metal backend
tests/         IR, lowering, numerical parity, generation, planner
examples/      runnable models, kernels, and static KV-cache benchmarks
```

## Roadmap

- **Symbolic shapes and limited control flow**, without weakening the strict
  verifier the static path relies on.
- **Shape-aware backend policy**, replacing per-op autotune with a selection that
  accounts for the whole graph, and a reproducible benchmark report.
- **Wider native coverage**: split-K for matmul, tile search for attention, and
  generated kernels for the families still on MLX.
- **More models**: adapters beyond Qwen2, starting with the Llama family, and
  long-context variants that need RoPE scaling.
- **Other backends**, once the core ABI has settled.

## Contributing

Issues and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) covers
what a change needs, how to add an operator, and how a release is published. A
change to lowering behaviour needs a stated motivation, a reproducible test, and
numerical parity against PyTorch.

## License

MIT
