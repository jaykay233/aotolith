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
```

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

`aoto bench` is the regression gate for the compiler itself. It measures a fixed
set of shapes — a projection, a gated MLP, an RMSNorm, causal attention, an int4
GEMV, and the Qwen2 prefill and decode adapters — and can compare the result
against a recorded baseline:

```bash
aoto bench --baseline benchmarks/baselines/apple-m5-pro.json --output perf-report.json
aoto bench --update-baseline benchmarks/baselines/apple-m5-pro.json
```

Which numbers it compares exactly and which it only compares on the same GPU is
in [the performance notes](performance.md#how-the-numbers-were-taken).

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

## Artifacts

Artifacts use format version 5 and ABI version 1.3. Each graph, kernel table,
selection table, storage plan, and state tensor has a SHA-256 digest. Loading
validates the ABI, target, graph, entrypoint layout, state shape/dtype,
serialized kernels, rebuilt selections, and storage plan. Kernel objects are
rebuilt from the validated graph and persisted selection, so the artifact does
not need the original Python model and loading does not repeat autotuning.

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
