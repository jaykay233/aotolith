# Transformer Subgraph Compiler

English | [简体中文](README.zh-CN.md)

An experimental Transformer model compiler with native Apple Silicon and Metal support.

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#status)

Transformer Subgraph Compiler captures PyTorch models as FX graphs, lowers them through a typed SSA IR, and produces an executable plan backed by generated Metal kernels and MLX fallbacks.

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
| Executable artifact | Versioned `.tsca` archives contain graphs, kernels, selections, storage metadata, and module state; they load and execute in a fresh process |
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
        ├── save: manifest + graphs + kernels + selections + state
        │
        ▼
Versioned artifact → reload and rebuild kernels → Apple GPU
```

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- MLX 0.32
- PyTorch 2.7+ for FX capture

## Installation

```bash
git clone https://github.com/jaykay233/transformer-subgraph-compiler.git
cd transformer-subgraph-compiler

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

from transformer_subgraph_compiler import compile_callable


def fn(x, weight, bias):
    hidden = torch.addmm(bias, x, weight, alpha=2.0)
    return torch.nn.functional.silu(hidden)


torch.manual_seed(0)
example_inputs = (
    torch.randn(4, 16),
    torch.randn(16, 32),
    torch.randn(32),
)

compiled = compile_callable(fn, example_inputs, codegen_matmul=True)
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

from transformer_subgraph_compiler import compile_module


model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        use_cache=False,
    )
).eval()

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
compiled = compile_module(
    model,
    (input_ids,),
    example_kwargs={"use_cache": False},
    codegen_matmul=True,
    codegen_attention=True,
)

logits = compiled(input_ids)[0]
mx.eval(logits)
```

### Command Line And Artifacts

The installed `tsc` command compiles a callable, module, or HuggingFace model. It can also inspect or execute the result without importing the original model.

```bash
tsc compile \
  --module 'my_module:build_model' \
  --input-file inputs.json \
  --output model.tsca \
  --codegen-matmul \
  --autotune

tsc inspect model.tsca
tsc verify model.tsca --require-device-match
tsc run model.tsca --input-file inputs.json --output outputs.json
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

Use repeated `--input '<json>'` arguments instead of `--input-file` when convenient. `--callable MODULE:ATTR` traces a function, `--module MODULE:ATTR` accepts an `nn.Module`, class, or factory, and `--hf-model` loads a HuggingFace causal language model. `python -m transformer_subgraph_compiler` provides the same commands when the console script is not on `PATH`.

Artifacts use format version 3 and ABI version 1.2. Each graph, kernel table, selection table, storage plan, and state tensor has a SHA-256 digest. Loading validates the ABI, target, graph, entrypoint layout, state shape/dtype, serialized kernels, rebuilt selections, and storage plan. Kernel objects are rebuilt from the validated graph and persisted selection, so the artifact does not need the original Python model and loading does not repeat autotuning.

Strict GPU matching is optional. `--require-device-match` or `require_device_match=True` rejects an artifact compiled for another Apple GPU model; the default allows execution on another Apple Silicon device after rebuilding kernels.

## Native Metal Code Generation

Pointwise operations, softmax, reductions, LayerNorm, and RMSNorm generate Metal kernels by default. Matmul and attention use MLX unless native code generation is enabled:

```python
compiled = compile_callable(
    fn,
    example_inputs,
    codegen_matmul=True,
    codegen_attention=True,
)
```

Native code generation is not automatically faster. On the shapes benchmarked so far, the custom matmul and attention kernels are slower than MLX's tuned implementations, which is why both options remain opt-in. `CompiledPlan.selections` records which path each operation used and why a fallback occurred.

## Supported Surface

| Area | Coverage |
|---|---|
| Models | Decoder-only Transformer forward pass; tiny Llama/Qwen-style models; fixed-capacity static KV cache prefill/decode |
| Matrix operations | `mm`, `bmm`, `addmm`, `baddbmm`, including supported batch broadcasting |
| Indexing | Embedding, `index_select`, advanced indexing, `gather`, and `scatter` |
| Transformer operations | Scaled dot-product attention, decomposed causal attention, softmax, sum/mean, cumsum, LayerNorm, RMSNorm, SiLU, GELU, SwiGLU, and other common pointwise operations |
| Layout and data movement | View/reshape, transpose/permute, slice, expand, cat/stack, clone, casts, and static constants |
| Dtypes | FP16, BF16, FP32, bool, I32, and I64 where supported by the operation |
| KV cache | Static-shape prefill/decode graph contracts and layout; allocation and paging remain external |

Operator coverage is intentionally narrower than PyTorch. Unsupported targets fail during capture or IR construction rather than silently executing outside the plan. See [docs/architecture.md](docs/architecture.md) for the detailed lowering table (currently in Chinese).

## Known Limitations

- Shapes and dtypes are fixed at compile time. Dynamic shapes, symbolic dimensions, and control flow are not supported.
- Tensor-valued keyword arguments are not graph inputs yet.
- Shared parameter storage aliases are currently serialized as separate state entries. Correctness is preserved, but the artifact does not yet deduplicate aliased storage.
- The artifact ABI accepts only the `apple-metal` target. Cross-vendor artifacts and host-runtime ABI compatibility are not implemented.
- Native kernel coverage and performance remain narrower than MLX. `codegen_matmul` and `codegen_attention` are opt-in because the current custom kernels are not consistently faster.
- Native attention requires rank-3/4 inputs with matching dtypes and supports MHA/GQA/MQA under the documented mask and head-dimension constraints. Other cases fall back to MLX SDPA.
- The repository does not implement scheduler, continuous batching, preemption, paged KV cache, prefix cache, sampling policies, PD disaggregation, EP/MoE dispatch, EPLB, or serving APIs.
- Training and autograd are out of scope.

## Testing

```bash
python -m pytest -q
python examples/tiny_llama_compile.py
```

Additional examples cover embeddings, MLP and matmul kernels, softmax, reductions, normalization, attention, and static KV-cache generation:

```bash
python examples/tiny_transformer.py
python examples/tiny_transformer_generate.py
python examples/metal_matmul_benchmark.py
python examples/metal_attention_benchmark.py
```

First execution includes MLX kernel JIT compilation and GPU dispatch. Timing from a first run is not a steady-state benchmark.

## Roadmap

1. Preserve shared storage aliases in the state ABI instead of serializing duplicate parameter views.
2. Add symbolic shapes and limited control flow without weakening static-shape validation.
3. Expand autotuning from per-op candidate timing to shape-aware backend policies and reproducible benchmark reports.
4. Improve native matmul and attention performance, then broaden model, dtype, operator, and GPU coverage while keeping PyTorch parity as the acceptance gate.

## Repository Layout

```text
src/transformer_subgraph_compiler/   Compiler frontend, IR, passes, planner, selection, and Metal backend
tests/                                IR, lowering, numerical parity, generation, and planner tests
examples/                             Runnable model, kernel, and static KV-cache examples
docs/architecture.md                  Detailed implementation notes and known boundaries
```

## Contributing

Issues and pull requests are welcome. Changes should include a focused motivation, a reproducible test, and numerical parity evidence when lowering behavior changes.

## License

MIT. See [LICENSE](LICENSE).
