# Aotolith

[English](README.md) | 简体中文

一个原生支持 Apple Silicon 和 Metal 的实验性 Transformer 模型编译器。

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](#环境要求)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#当前状态)

Aotolith 把 PyTorch 模型捕获成 FX 图，经过带类型和 shape 的 SSA IR，生成由 Metal kernel 和 MLX fallback 共同执行的计算计划。

> [!IMPORTANT]
> 这个项目编译 Transformer 模型计算。请求调度、continuous batching、分页 KV 分配、采样、分布式执行和 serving 由外部运行时负责。

## 为什么做这个项目

主流 Transformer 编译栈大多围绕 CUDA 和成熟的 NVIDIA kernel 生态构建。这个项目走另一条路：保留与后端无关的图 IR，再把模型计算原生编译到 Apple Silicon，而不是直接把 MLX 当成完整运行时。

编译器负责图捕获、合法化、布局简化、算子融合、公共子表达式消除、tensor 生命周期分析、kernel selection 和 Metal code generation。MLX 提供 fallback 算子，并在原生覆盖范围扩大前作为数值执行基线。

## 当前状态

项目已经可以运行静态 shape 的端到端 Transformer 实验，但还不是稳定的编译器产品。

| 模块 | 当前状态 |
|---|---|
| PyTorch 前端 | 可编译静态 shape 的 callable 和 `nn.Module`；模型参数与 buffer 会绑定到执行计划 |
| 图编译器 | 使用 typed SSA IR，支持 shape/dtype 校验、合法化、布局折叠、融合和 CSE |
| 存储规划 | 已生成 live interval、释放点、峰值活跃字节和可复用 buffer slot，执行时使用真实 arena |
| Kernel 选择 | 为每个 lowered op 记录原生 Metal 或 MLX 实现，以及 fallback 原因 |
| 编译产物 | 版本化 `.aoto` archive 包含图、kernel、selection、storage metadata 和模型 state，可在新进程加载执行 |
| Autotune | 可选的 profile 流程会测量原生 kernel 和 MLX 候选，测量结果与最终选择写入 artifact |

当前回归用例把两层 HuggingFace Llama 从 220 个 FX 算子降到 136 个 lowered op，logits 相对 PyTorch 参考实现的最大绝对误差为 `8.941e-08`。释放原始 module 后，编译计划仍可执行。

## 编译流程

```text
PyTorch FX / nn.Module
        │
        ▼
Typed SSA IR
        │  legalization、layout folding、fusion、CSE
        ▼
Liveness analysis / storage plan
        │
        ▼
Kernel selection / fallback planning
        │
        ├── 生成的 Metal kernel
        └── MLX fallback 算子
        ▼
CompiledPlan
        │
        ├── 首次调用把图展开成直线 Python 代码
        ├── save：manifest + 图 + kernel + selection + state
        │
        ▼
版本化 artifact → 重新加载并构建 kernel → Apple GPU
```

一次生成里要重复成千上万次的 step 不该每次都重新解析自己的操作数。因此首次调用会把 plan 编译成一个 Python 函数，函数体里每个算子对应一行赋值，kernel 调用点直接绑定成常量；之后的调用只校验输入，然后执行这个函数。kernel 本身没有任何变化，变的只是它们周围的簿记开销。

## 环境要求

- Apple Silicon Mac，macOS
- Python 3.11+
- MLX 0.32
- PyTorch 2.7+，用于 FX 捕获

## 安装

```bash
git clone https://github.com/jaykay233/aotolith.git
cd aotolith

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

只编译 Python callable 时可以安装 `.[fx]`。`dev` 和 `hf` extras 会额外安装 HuggingFace Transformers，用于模型级编译和测试。

## 快速开始

下面的例子编译一个小型 `addmm + SiLU` 函数。执行前可以检查图、存储计划和 kernel selection。

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

### 编译 HuggingFace 模型

`compile_module` 会把模型参数和 buffer 作为显式 state 参与 tracing。返回的计划不持有原始 `nn.Module`。

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
        use_cache=False,
    )
).eval()

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"use_cache": False},
    codegen_matmul=True,
    codegen_attention=True,
)

logits = compiled(input_ids)[0]
mx.eval(logits)
```

### 编译 Qwen2.5 静态 KV Cache

`aotolith.models.qwen2` 提供 Qwen2/Qwen2.5 的单请求静态图适配层。它把预填和逐 token 解码拆成两张图，用固定容量的 KV Cache 连接，并包含 Qwen2 的 RMSNorm、RoPE、GQA、SwiGLU 和 LM head。

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
prompt_ids = tokenizer.encode("用一句话解释 KV Cache", add_special_tokens=False)[:128]

runtime = compile_qwen2_static_cache(
    model,
    prompt_length=len(prompt_ids),
    max_new_tokens=64,
)
tokens = runtime.greedy_generate(prompt_ids, max_new_tokens=64)
print(tokens.tolist())
```

这个适配层固定 batch、prompt 长度和最大生成长度，只实现 greedy 单请求生成；host 侧负责 token 循环，scheduler、batching、分页 KV 和采样仍不包含在编译器中。`runtime.save("qwen")` 会保存 prefill 和 decode 两个 `.aoto`，每张图当前各自包含一份模型 state。

query、key、value 共享同一份归一化输入，因此适配层把三者按层存成一个打包后的投影并只做一次矩阵乘。拆分结果只需两个 view 操作，却省掉两次 kernel launch：128 行的 key/value 投影对 GPU 来说太小，带宽利用率不到融合后投影的五分之一。

仓库内 benchmark 使用相同的 prompt、精度和 `128 + 64` token 参数比较 PyTorch MPS、MLX 和 Aotolith：

```bash
python examples/qwen_static_cache_benchmark.py --output /tmp/qwen-aotolith.json
python examples/qwen_static_cache_benchmark.py \
  --codegen-matmul \
  --mlx-baseline \
  --output /tmp/qwen-aotolith-codegen.json
```

`--mlx-baseline` 会在同一个进程里用同一段 prompt 给 `mlx_lm` 自己的 greedy decode 计时，原生基线因此是在跑 benchmark 的这台机器上测出来的，而不是从别处引用。在这台机器上（M5 Pro，MLX 0.32）多次运行读到 **195.5-197.2 token/s**。

| 配置 | prefill | decode（每步同步） | decode（异步） | 相对 `mlx_lm` |
| --- | --- | --- | --- | --- |
| 仅 MLX fallback | 10 146 token/s | 103.3 token/s | 116.2 token/s | 0.59x |
| `--codegen-matmul` | 13 274 token/s | 178.6 token/s | **232.5 token/s** | **1.18x** |
| `--codegen-matmul --codegen-attention` | 12 746 token/s | 176.8 token/s | 231.8 token/s | 1.18x |

（Qwen2.5-0.5B-Instruct、128 prompt token、greedy、batch 1、BF16、11 次重复取中位数；每一行的 64 个生成 token 都与 PyTorch 参考实现一致。同一份构建跑十一次下来，decode 落在 231.4-232.9 token/s、prefill 在 13.0k-13.5k token/s、`mlx_lm` 基线在 195.3-197.2 token/s；加 `--codegen-attention` 后 prefill 降到 12.7k 是 attention kernel 长序列 tile 更慢所致，不是 decode 的代价。）

和 `mlx_lm` 对比应该看 `decode_async`，因为两者都把 host 循环摊到了多个 token 上；`decode_sync_each_step` 每个 token 都要同步一次设备，任何服务循环都不会这么做。异步路径此前落后于同步路径，现在反超，因为每个 step 只强制一次设备同步。

decode 现在受权重带宽限制，而不是受 host 限制。构建一个 step 只需 0.98 ms Python，GPU 需要 4.0 ms，所以 host 完全被异步队列盖住；在 plan 展开之前 host 要 6.2 ms，那时它才是瓶颈。一个 step 要流式读取 991 MB 权重（24 层的 `qkv` 2.07 MB、`o_proj` 1.61 MB、三个 8.72 MB 的 MLP 矩阵，再加 272 MB 的 LM head），按 4.0 ms 算是 248 GB/s。剩下的余量在哪，看下面的单次速率就知道：272 MB 的 LM head 单次能跑到 414 GB/s，而 8.7 MB 的 MLP 单次只有 265-281 GB/s。输出宽度只有 4864 列的一次 launch 填不满这台设备，而 151936 列可以，这就是小分桶落后的具体原因。要明显比现在更快，需要的是少读字节，而不是减少 kernel。

单个 decode step 的分桶 GPU 开销。测量时把同一组 kernel 连着回放足够多遍，让设备队列一直是满的，于是墙钟时间就是设备时间，host 提交被盖住：

| 分桶 | launch 次数 | 每次搬运 | 单次 GPU 时间 | 带宽 |
| --- | --- | --- | --- | --- |
| LM head GEMV | 1 | 272.3 MB | 658 us | 414 GB/s |
| MLP up GEMV + `silu_mul` | 24 | 8.73 MB | 33.0 us | 265 GB/s |
| 带残差的 MLP down GEMV | 24 | 8.73 MB | 32.7 us | 267 GB/s |
| MLP gate GEMV + RMSNorm | 24 | 8.72 MB | 31.0 us | 281 GB/s |
| 融合 QKV GEMV + RMSNorm | 24 | 2.07 MB | 10.9 us | 190 GB/s |
| attention（单 query） | 24 | 0.10 MB | 7.7 us | 13 GB/s |
| 带残差的 o_proj GEMV | 24 | 1.61 MB | 7.7 us | 210 GB/s |
| RoPE q / k | 48 | 1 KB | 7.5 us | - |
| 加性 attention mask | 1 | 0.2 KB | 23.6 us | - |

加起来是 4.0 ms 设备时间、991 MB，也就是整个 step。LM head 加三个 MLP 矩阵占了其中 75%，而它们是在搬字节，不是在等 launch；其余四分之一分散在 121 次每次最多搬 2 MB 的 launch 上。一个 step 仍然带着 194 次有依赖的 launch，但几乎不搬数据的那些（RoPE、mask）在真实链条里便宜到删掉也不影响总量。把 query 和 key 的旋转合成一次 16 头旋转正是这个实验：它省掉 24 次 launch 和 24 个图节点，decode 却是 232.9 token/s 对之前的 232.5，落在运行间波动内，所以没有保留。

有两种做法会把这个表测成“什么都没测”，而且看起来都很正常。一是每轮丢掉输出的数组，惰性图在到达设备之前就被释放，墙钟时间只覆盖 host 提交，带宽那一列会算出 1 TB/s 以上。二是传全零权重，驱动可以跳过读取，小分桶因此被抬高。一个分桶必须让每一轮的输出都活着，并且喂真实数值，它的数字才有意义；任何超过 `mx.sum` 读同样字节能达到的读数，测的是 harness 而不是 kernel。

这个方向上有两个实验被测过并放弃。把新的 key/value 直接写到 slot `positions`、用 mask 挑出有效 slot，读起来比滚动窗口更顺，但 MLX 会把 scatter 降级成对整窗口做一遍下标比较，而 concatenate 只是同样字节数的一次拷贝：每个 step 慢 0.18 ms，所以保留滚动。另外，decode token 的旋转位置是 `valid_length - 1` 而不是 `valid_length`；`Qwen2StaticCacheRuntime.decode_token` 曾把后者写进文档并短暂使用，代价是相对参考实现 0.7 的 logit 最大偏差，token 也无法完全对齐。现在 64 个 token 全部一致。

### 命令行与编译产物

安装后可以用 `aoto` 编译 callable、module 或 HuggingFace 模型。产物也可以在不导入原始模型的情况下检查和执行。

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

`inputs.json` 是按位置排列的 tensor 规格。标量使用空 shape 和单元素 data：

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

也可以重复传入 `--input '<json>'`，不写输入文件。`--callable MODULE:ATTR` 捕获函数，`--module MODULE:ATTR` 接受 `nn.Module`、类或 factory，`--hf-model` 加载 HuggingFace causal language model。console script 不在 `PATH` 时，用 `python -m aotolith` 执行相同命令。

artifact 使用 format version 4 和 ABI version 1.2。两张图、kernel table、selection table、storage plan 和每个 state tensor 都有 SHA-256 digest。加载时会校验 ABI、target、图、入口布局、state shape/dtype、序列化 kernel、重新构建的 selection 和 storage plan。kernel 根据校验后的图和已保存的 selection 重建，因此 artifact 不需要原始 Python 模型，加载时也不会重新 autotune。

是否要求 GPU 完全相同由调用方决定。`--require-device-match` 或 `require_device_match=True` 会拒绝为其他 Apple GPU 型号编译的 artifact；默认允许在其他 Apple Silicon 设备上重新构建 kernel 后执行。

## 原生 Metal Code Generation

pointwise、softmax、reduction、LayerNorm 和 RMSNorm 默认生成 Metal kernel。matmul 和 attention 默认使用 MLX，开启下面的选项后才生成原生 kernel：

```python
compiled = aoto.compile_callable(
    fn,
    example_inputs,
    codegen_matmul=True,
    codegen_attention=True,
)
```

生成的原生 kernel 在 matmul 上快于 MLX，但在 attention 上仍落后于 MLX 的 SDPA，两个选项默认关闭的原因因此并不相同：

| 基准 | Aotolith | MLX | 倍率 |
|---|---|---|---|
| matmul 512x2048x512 f32 | 89.4 us | 114.2 us | 1.28x |
| matmul 512x2048x512 f16 | 64.8 us | 85.4 us | 1.32x |
| matmul 512x2048x512 bf16 | 64.5 us | 76.6 us | 1.19x |
| matmul 1024x4864x896 f16 | 372.9 us | 531.2 us | 1.42x |
| matmul 4096x4096x4096 f16 | 5.51 ms | 5.45 ms | 0.99x |
| attention 1x8x512x64 f16 causal | 35.2 us | 28.5 us | 0.81x |
| attention 1x8x2048x128 f16 causal | 585.6 us | 461.6 us | 0.79x |
| attention 1x64x4096x128 f16 causal | 16.3 ms | 13.2 ms | 0.81x |

matmul kernel 用 MPP tensor-core 的 `matmul2d` 跑分块操作数。分块配置由输出 shape 和右操作数的布局共同决定：右操作数转置时 K 方向连续，适合更宽的列分块；右操作数行主序时沿 K 方向跨步，更适合更高的归约步长。在 43 组 shape 与布局组合上，选择器对每个 shape 都保持在实测最优分块的 5% 以内；相对 MLX 平均为 1.16x（右操作数转置）和 1.10x（行主序），没有 shape 低于 1.01x。最大的几个 shape 收敛到持平，因为两边都逼近同一套 tensor core 吞吐。

attention kernel 是寄存器驻留的 FlashAttention：一个 simdgroup 负责 16 行 query 片段和整个 value 宽度，online-softmax 的累加器始终留在寄存器里，`Q` 只在开头暂存一次到 threadgroup memory，`K`/`V` 直接从 device memory 流式读取。长序列上能跑到 16.8 TFLOPS，但在 prefill 形态上仍比 MLX 的 SDPA 慢约 20%。decode 是另一个问题：单行 query 用不上分块 online softmax，所以后端会另外发射一个单 query kernel，把整个 head 的分数留在 threadgroup memory 里，用两级归约求 `max` 和 `sum(exp)`，再做 value 累加而不需要重新缩放。因此 `--codegen-attention` 不再损失 decode 吞吐：它每层只有一次 launch、覆盖 192 个 cache slot，与 MLX 的 SDPA 落在同一噪声范围内，而不是慢 20%。对分块 prefill kernel 而言，把 `K`/`V` 暂存进 threadgroup memory 实测慢 3 倍：这份 tile 只有两个 simdgroup 共享，而每个 tile 的 barrier 代价超过它省下的重复 device 读取。`CompiledPlan.selections` 会记录每个算子实际使用的路径以及 fallback 原因。两个 kernel 都只覆盖 FP16/BF16，其他精度走 MLX。

## 支持范围

| 类别 | 覆盖范围 |
|---|---|
| 模型 | decoder-only Transformer forward；tiny Llama/Qwen 风格模型；Qwen2/Qwen2.5 单请求静态 KV Cache prefill/decode；已验证 Qwen2.5-0.5B-Instruct |
| 矩阵计算 | `mm`、`bmm`、`addmm`、`baddbmm`，包括支持范围内的 batch broadcasting |
| 索引 | embedding、`index_select`、advanced indexing、`gather` 和 `scatter` |
| Transformer 计算 | scaled dot-product attention、分解后的 causal attention、softmax、sum/mean、cumsum、LayerNorm、RMSNorm、SiLU、GELU、SwiGLU 和常见 pointwise 算子 |
| 布局与数据移动 | view/reshape、transpose/permute、slice、expand、cat/stack、clone、cast 和静态常量 |
| 数据类型 | 支持范围内的 FP16、BF16、FP32、bool、I32 和 I64 |
| KV Cache | 静态 shape 的 prefill/decode 图契约和布局；分配与分页仍由外部运行时负责 |

算子覆盖范围小于 PyTorch。不支持的 target 会在捕获或构造 IR 时失败，不会静默绕过编译计划。逐算子的 lowering 表和边界见 [docs/architecture.md](docs/architecture.md)。

## 已知限制

- shape 和 dtype 在编译期固定，不支持动态 shape、符号维度和控制流。
- tensor-valued keyword arguments 目前不会提升为图输入。
- 共享参数 storage alias 目前会按多个 state entry 序列化。数值结果不受影响，但 artifact 还没有去重这些 aliased storage。
- artifact ABI 只接受 `apple-metal` target，尚未实现跨厂商 target 和 host runtime ABI 兼容。
- 原生 kernel 的覆盖范围仍窄于 MLX。`codegen_matmul` 仍默认关闭只是因为它还在实验阶段，在实测的 matmul shape 上都快于 MLX；`codegen_attention` 在 prefill 形态上比 MLX 的 SDPA 慢约 20%，但在单 query 的 decode 路径上与它持平，因为专用 kernel 用整 head 的 threadgroup 归约取代了分块 online softmax。
- 原生 attention 要求 rank-3/4 输入、Q/K/V dtype 一致，并在文档规定的 mask 和 head dimension 约束内支持 MHA/GQA/MQA；其他情况回退到 MLX SDPA。
- 仓库不实现 scheduler、continuous batching、抢占、paged KV cache、prefix cache、采样策略、PD 分离、EP/MoE dispatch、EPLB 或 serving API。
- 不支持训练和 autograd。

## 测试

```bash
python -m pytest -q
python examples/tiny_llama_compile.py
```

其他示例覆盖 embedding、MLP、matmul kernel、softmax、reduction、normalization、attention 和静态 KV Cache 生成：

```bash
python examples/tiny_transformer.py
python examples/tiny_transformer_generate.py
python examples/metal_matmul_benchmark.py
python examples/metal_attention_benchmark.py
```

首次执行包含 MLX kernel JIT 和 GPU dispatch，首轮耗时不能作为 steady-state benchmark。

PyTorch 对越界的 embedding／gather 下标抛出 `IndexError`，而 MLX 会静默 wrap 或填值，因此后端需要复现该检查。下标值在设备上，所以校验它必然要同步：逐个算子检查会让每个 decode step 产生六次设备同步，改成把越界检查排队累积、最后合并成一次 reduction 之后只剩一次。如果调用方已经知道下标一定在范围内，可以去掉这最后一次同步——`plan(..., verify_index_bounds=False)`，或 `runtime.decode_token(..., verify_token=False)`；greedy 主循环正是这么做的，因为它回填的 token 全部来自词表上的 `argmax`。默认的严格行为没有变化，越界仍然抛 `IndexError`。

## Roadmap

1. 在 state ABI 中保留共享 storage alias，而不是重复序列化相同参数的视图。
2. 增加符号 shape 和有限控制流，同时保留静态 shape 的严格校验。
3. 把逐 op autotune 扩展成 shape-aware backend policy 和可复现 benchmark report。
4. 提升原生 matmul、attention 的性能，再扩展模型、dtype、算子和 GPU 覆盖；PyTorch 数值对齐仍是验收条件。

## 仓库结构

```text
src/aotolith/   编译器前端、IR、passes、planner、selection 和 Metal backend
tests/                               IR、lowering、数值对齐、生成和 planner 测试
examples/                            可运行的模型、kernel 和静态 KV Cache 示例
docs/architecture.md                 详细实现说明与已知边界
```

## 参与贡献

欢迎提交 issue 和 pull request。修改 lowering 行为时，请附上明确的动机、可复现测试和数值对齐结果。

## License

MIT，详见 [LICENSE](LICENSE)。
