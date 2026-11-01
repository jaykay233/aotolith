# Aotolith

[English](README.md) | 简体中文

**面向 Apple Silicon 的 Transformer 编译器。** 把 PyTorch 模型捕获成 FX 图，经过 typed SSA IR 下沉，最后跑在编译器自己生成的 Metal kernel 上；MLX 既是 fallback，也是数值基准。

[![PyPI](https://img.shields.io/pypi/v/aotolith?logo=pypi&logoColor=white)](https://pypi.org/project/aotolith/)
[![CI](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml/badge.svg)](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](#环境要求)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#已知限制)

```bash
pip install aotolith
```

大多数 Transformer 编译器以 CUDA 为前提。这个项目保留一套后端无关的 IR，并把整条路走到 Metal：图捕获、合法化、布局折叠、融合、CSE、张量生命周期、kernel 选择、代码生成，以及一个能在全新进程里重新加载的版本化产物。

> [!IMPORTANT]
> 这个项目只编译模型计算本身。请求调度、continuous batching、分页 KV、采样和服务接口属于 host runtime。

```python
import torch
import aotolith as aoto

def fn(x, weight, bias):
    return torch.nn.functional.silu(torch.addmm(bias, x, weight, alpha=2.0))

inputs = (torch.randn(4, 16), torch.randn(16, 32), torch.randn(32))
compiled = aoto.compile_callable(fn, inputs, codegen_matmul=True)

print(compiled.lowered_graph)       # 融合后的 IR
print(compiled.plan.selections)     # 原生 kernel 还是 MLX，以及原因
```

## 环境要求

- Apple Silicon 上的 macOS
- Python 3.11+
- MLX 0.32
- PyTorch 2.7+（FX 捕获）

## 安装

```bash
pip install aotolith
```

从源码安装（跑测试也要用这种方式）：

```bash
git clone https://github.com/jaykay233/aotolith.git
cd aotolith
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
```

只想编译 Python callable 可以装 `.[fx]`；`dev` 和 `hf` extra 会装上 HuggingFace Transformers，用于模型级编译和测试。

## 快速开始

编译一个 HuggingFace 模型。参数和 buffer 成为 plan 的 state，plan 不再持有原 module：

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

编译 Qwen2.5 静态 KV Cache 并做 greedy 生成：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aotolith.models.qwen2 import compile_qwen2_static_cache

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
prompt_ids = tokenizer.encode("用一句话解释 KV Cache")[:128]

runtime = compile_qwen2_static_cache(
    model, prompt_length=len(prompt_ids), max_new_tokens=64, quantize=64
)
tokens = runtime.greedy_generate(prompt_ids, max_new_tokens=64)
```

适配层固定 batch、prompt 长度和最大生成长度，只实现 greedy 单请求生成。`compile_qwen2_bucketed` 改为每个 prompt 长度桶编译一对计划，更短的 prompt 左填充进能装下它的最小桶。

同一套静态 Cache 运行时也覆盖第二个模型族。`aotolith.models.llama` 导出 `compile_llama_static_cache` 与 `compile_llama_bucketed`，与 Qwen2 的入口只在编译期坚持什么上不同：

```python
from aotolith.models.llama import compile_llama_static_cache

runtime = compile_llama_static_cache(
    model, prompt_length=128, max_new_tokens=64, quantize=64
)
tokens = runtime.greedy_generate(prompt_ids)
```

两个族在三件事上不一致、其余全部一致：Llama 的 q/k/v/o 投影没有 bias，所以打包投影 kernel 少一段 epilogue 折叠；`num_key_value_heads` 既可以整除 `num_attention_heads`，也可以等于它；RoPE 底数是配置项 `rope_theta` 而不是常量。`check_llama_model` 与 `check_qwen2_model` 会拒绝另一个族，而不是悄悄编出一张按别的 checkpoint 排布的投影图。

两个后端共用同一个入口。`target="mlx-reference"` 把同一张图编成一份完全没有 kernel 的计划——每个节点都由 MLX 解释器执行——于是没有 Metal 的 host 也能编出并校验计划，同一张 lowering 后的图也获得一次独立执行，用来对照 shape、dtype、cache 内容和 token 决策。

## 实测结果

Qwen2.5-0.5B-Instruct，128 个 prompt token，生成 64 个 greedy token，BF16，batch 1，M5 Pro。`mlx_lm` 在同一进程、同样形状的循环里计时：

| 配置 | prefill | decode（每步同步） | decode（异步） | 相对 `mlx_lm`（异步） |
| --- | --- | --- | --- | --- |
| 仅 MLX fallback | 9 650 token/s | 154.7 token/s | 185.9 token/s | 0.77x |
| `--codegen-matmul` | 10 872 token/s | 182.4 token/s | **239.1 token/s** | **1.00x** |
| `--codegen-matmul --codegen-attention` | 12 379 token/s | 184.9 token/s | 239.0 token/s | 0.99x |
| `mlx_lm` 本身 | — | 198.0 token/s | 242.1 token/s | — |

codegen 的意义在**第二次**跨步，而不是第一次：全走 fallback 时比 `mlx_lm` 落后 23%，打开 matmul kernel 把这个差距抹平。它并不带来领先。本表早先写的是 1.21x，那个数字来自两边计时方式不同——本仓库的 decode 路径每步用 `mx.async_eval` 提交，而 `mlx_lm` 基线每步都等，每 token 一次设备往返值大约 15%。按同口径跨三个 prompt 长度测，decode 落在 `mlx_lm` 的 0.89-0.96x，prefill 在 32 token 上是 0.60x、128 token 起是 1.08-1.10x。`examples/compare_mlx_lm.py` 会打出这张网格。

token 用 `examples/qwen_static_cache_benchmark.py --codegen-matmul --mlx-baseline` 复现，距离用 `examples/qwen_logit_distance.py`。

64 个 token 与 PyTorch BF16 参考实现逐位一致，开不开原生 kernel 都是如此，`mlx_lm` 产出的也是同一批 token。

token 一致不等于数字一致，而且这个差距取决于 prefill 看到多少 token —— `examples/qwen_logit_distance.py` 就是测它：

| prompt token 数 | fallback 平均 | fallback 最大 | codegen 平均 | codegen 最大 |
| --- | --- | --- | --- | --- |
| 7 | 0.1056 | 0.6875 | 0.0290 | 0.1875 |
| 8 | 0.0783 | 0.5469 | 0.0451 | 0.2500 |
| 16 | 0.0312 | 0.2031 | 0.0310 | 0.1953 |
| 128 | 0.0596 | 0.4219 | 0.0566 | 0.4219 |

顺着列往下读：fallback 的平均距离在 7 token 和 16 token 之间差 3.4 倍，所以单独一个数字描述的是 prompt，不是模型。这张表能说明的是：在这些长度上 codegen 从不比 fallback 更远离 PyTorch —— 7 token 时近 3.6 倍，16 token 时持平（0.0312 对 0.0310）—— 到 128 token 的 prefill 时两者相差在 5% 以内。打开 codegen 是吞吐决策，不是精度决策。

### 量化权重

`quantize=<group size>` 打包每个投影，`quantize_bits=` 选位宽（4 或 8 bit）。这个对比只有在一件事成立时才有意义：基线也要量化到**同样的位宽**，并且用**同样的循环形状**计时——这正是 `--mlx-baseline` 现在做的事。用 int4 的 plan 去比 bf16 的 `mlx_lm`，量到的是格式差异，不是编译器差异。

同一个 prompt、同样 64 步：

| 权重 | 每步字节数 | 本仓库 | `mlx_lm`（同位宽） | 比值 |
| --- | --- | --- | --- | --- |
| BF16 | 991 MB | 200.6 token/s | 208.7 token/s | 0.96x |
| int4，group 64 | 250 MB | **373.1 token/s** | 328.7 token/s | **1.13x** |
| int8，group 64 | 495 MB | 328.6 token/s | 301.3 token/s | 1.09x |

4 bit 的差距更大，原因在带宽：bf16 一步要读 991 MB，两边都被压满，没有可赢的空间。4 bit 一步只读 250 MB、约 2.7 ms，算下来不到 100 GB/s——远低于这台机器能流的速率——所以一步又变成由每步开销决定，而这正是编译好的 plan 胜过"每层注意力 + 每个投影各一个节点"的地方。8 bit 夹在中间：495 MB 还剩一些余量，所以收益小一些。

同一对配置测三次、交替谁先跑：4 bit 是 1.135x、1.125x、1.138x；8 bit 是 1.090x、1.091x、1.099x。BF16 在这里读到 0.96x 和 1.00x，跨 prompt 长度是 0.89-0.96x，所以它是持平而不是领先——领先来自把位宽降下来。

> [!NOTE]
> 如果改用 `mlx_lm.generate()` 而不是它原生的 decode 循环来计时，量化后的比值会变成 1.35x 和 1.19x，因为 `generate()` 每 token 多做了一次文本处理。这正是本 README 早先在 bf16 上犯过、并已纠正的同一个错误，也是这里的 helper 要把基线也量化、两种循环形状都测并分开报的原因。可信的是表里那个数字。

代价是精度，和所有量化一样。对原始权重平均相对误差是 4 bit 的 11.5% 和 8 bit 的 0.63%。8 bit 的 plan 在基准 prompt 的全部 64 步上都复现了 BF16 checkpoint 的 greedy 选择；4 bit 的 plan 跟了 13 步，然后在一个近似平局上翻掉。所以 4 bit 是吞吐档，而当你不能接受措辞变化时用 8 bit，代价大约是 4 bit 速率的 12%。

一步在每种位宽下 host 与 kernel 的时间拆分在 `docs/performance.md`，中间那一列可以用 `examples/qwen_static_cache_benchmark.py --quantize 64 --quantize-bits 8 --mlx-baseline` 复现。

## 一步是怎么编出来的

```text
PyTorch FX / nn.Module
  → typed SSA IR        合法化、布局折叠、融合、公共子表达式消除
  → 存储计划            活跃区间、释放点、一块 arena
  → kernel 选择         生成的 Metal kernel 或 MLX fallback，每个 op 都记下原因
  → CompiledPlan        首次调用展开，可存成版本化 .aoto 产物
```

第一次调用会把 plan 编成一个函数，函数体是每个 op 一行赋值，kernel 调用点作为常量绑定。kernel 本身一点没变，变的是围绕它们的记账：在这个模型上它把 decode 一步的 host 成本从 6.2 ms 降到 0.98 ms。

作为更小的端到端检查，`examples/tiny_llama_compile.py` 编译一个两层的 HuggingFace Llama：捕获的 131 个 op 在 prefill 里降到 79（decode 从 151 降到 62），在 matmul 与 attention 都开着 codegen 时 logits 与 PyTorch 参考相差 `1.043e-07`——float32 舍入量级，不是累积漂移——并且在原 module 被释放之后 plan 仍然可执行。

最后这一点值得说清楚。关掉 codegen，同一张图跑在 MLX 自己的 kernel 上，漂移变成 `1.3e-3`：MLX 在这块 GPU 上的 float32 matmul 相对 float64 参考是 `8e-4` 量级，而且不随 K 增长，这是「降精度 matmul」的形状，不是累积误差。所以回退路径在正确性上是安全的，但不是逐位精确的，而这个差异属于 runtime，不属于编译器。

## 原生 Metal Code Generation

pointwise、softmax、reduction、LayerNorm 和 RMSNorm 默认生成 Metal kernel，matmul 和 attention 需要显式开启：

```python
compiled = aoto.compile_callable(fn, inputs, codegen_matmul=True, codegen_attention=True)
```

- `codegen_matmul` 走 MPP tensor-core 的 `matmul2d`，在实测的 FP16/BF16 形状上快于 MLX（大形状 1.32-1.47x），并让稠密 decode 投影吸收掉自己的 RMSNorm。f32 留在标量 kernel 上，因为硬件没有 f32 tensor core。
- `codegen_attention` 是寄存器驻留的 FlashAttention：prefill 形状比 MLX 的 SDPA 慢 10-17%，但在单 query 的 decode 路径上持平——那里由一个专门的全 head kernel 取代分块 online softmax。
- tensor core 需要 Apple GPU family 9 及以上。`compile` 会先真正加载并求值一个生成的 tensor-core kernel 再选路，设备拒绝时整张图改走 simdgroup 路径——半虚拟化的 GPU 就是这种情况。

## 支持范围

| 类别 | 覆盖范围 |
| --- | --- |
| 模型 | decoder-only Transformer forward；Qwen2/Qwen2.5 与 Llama 静态 KV Cache prefill/decode，含尺寸桶与权重 int4；已验证 Qwen2.5-0.5B-Instruct |
| 矩阵 | `mm`、`bmm`、`addmm`、`baddbmm`、`addbmm`、`mv`/`addmv`，含 batch broadcasting |
| 线性代数 | `solve`、`inv`、`det`、`cholesky` 一族、`eigh`、`svd`、`qr`、`pinv`、`lu_factor`、`matrix_exp`、`cdist` |
| 索引 | embedding、`index_select`、advanced indexing、`gather`、`scatter*`、`index_add`、`masked_scatter` |
| Transformer | SDPA 与分解后的 causal attention、softmax 一族、含 `var`/`std`/`nansum` 的归约、`vector_norm`、cumsum/cumprod、LayerNorm、RMSNorm、SiLU、GELU、SwiGLU |
| 空间 | `conv1d/2d/3d`（stride/padding/dilation/groups）、pooling 一族、`interpolate`、BatchNorm/GroupNorm/InstanceNorm |
| 排序与选择 | `argmax`、`argmin`、带 `dim` 的 `max`/`min`、`median`、`sort`、`kthvalue`、`topk` |
| 布局 | view、transpose/permute、slice、split/chunk/unbind、squeeze、expand、`repeat`/`tile`、cat/stack、padding、flip/roll、cast、`arange` |
| target | `apple-metal`（生成的 Metal kernel，其余回退 MLX）与 `mlx-reference`（不生成 kernel，只走 MLX 解释器） |
| 数据类型 | 支持范围内的 FP16、BF16、FP32、bool、I32、I64 |

算子覆盖范围小于 PyTorch。不支持的 target 会在捕获或构造 IR 时失败，不会静默绕过编译计划。每个算子族实际怎么执行，见 [docs/operators.md](docs/operators.md)；逐 pass 的实现细节见 [docs/architecture.md](docs/architecture.md)。

## 已知限制

- shape 和 dtype 在编译期固定，不支持动态 shape、符号维度和控制流。
- 产物在 manifest 里写明 target，只在对应后端可用的地方加载：`apple-metal` 需要 Apple GPU（kernel 由 MSL 重建），`mlx-reference` 只需要 MLX。目前没有 CUDA 或 CPU target。
- 训练和 autograd 不在范围内。
- 不包含 scheduler、continuous batching、抢占、分页 KV Cache、prefix cache、采样策略、PD 分离、EP/MoE dispatch 和服务 API。这些是嵌入方的职责，[docs/host-runtime.md](docs/host-runtime.md) 记录了 host 可以依赖什么。
- 部分算子族走 MLX 而不是生成 kernel，`mx.linalg` 的分解跑在 MLX 的 CPU stream 上。原因与确切边界按算子族记录在 [docs/operators.md](docs/operators.md)。

## 文档

- [docs/usage.md](docs/usage.md)：tensor 关键字参数、尺寸桶、`aoto` 命令、产物版本。
- [docs/errors.zh-CN.md](docs/errors.zh-CN.md)：全部错误码、含义、调用方该做什么，以及包会写什么日志（[English](docs/errors.md)）。
- [docs/compatibility.zh-CN.md](docs/compatibility.zh-CN.md)：版本之间承诺什么、什么算破坏性变更、弃用怎么走（[English](docs/compatibility.md)）。
- [CHANGELOG.md](CHANGELOG.md)：每个版本改了什么。
- [docs/operators.md](docs/operators.md)：每个算子族的执行方式，以及哪些拼写与 PyTorch 不同、为什么。
- [docs/performance.zh-CN.md](docs/performance.zh-CN.md)：decode step 的时间花在哪、每次重写买到了什么、哪些想法测过之后被放弃。
- [docs/host-runtime.md](docs/host-runtime.md)：编译器与服务引擎的分界线，以及嵌入方可以依赖什么。
- [docs/architecture.md](docs/architecture.md)：IR、pass 与代码生成，比 README 更细。

## 测试

```bash
python -m pytest -q
ruff check src tests examples
python examples/tiny_llama_compile.py
```

CI 在每次 push 上跑的就是这些：`ruff` 跑在 Linux 上；Apple Silicon 的 macOS runner 上做 `dev` extra 的可编辑安装、跑测试和 Llama 示例、再跑 `python -m build`。托管 runner 的 GPU 是半虚拟化的，装不上 tensor-core kernel，所以那里的 `compile` 会回退到 simdgroup 路径，相关测试自行跳过——绿灯只覆盖 simdgroup 路径。

`aoto bench` 是编译器自身的回归门禁：测量固定 shape，确定性 op 数精确比对，计时、峰值内存与冷启动与记录的 GPU baseline 比对。更多示例在 `examples/`。

## 仓库结构

```text
src/aotolith/
  capture/     FX 追踪（make_fx、fake mode、分解表保留集）
  legalize/    FX → 核心 op：ATen target 表、逐族 lowering
  ir/          typed SSA IR 与算子契约表
  target/      这条 GPU 能编译什么：能力查询与 tile 选择
  codegen/     MSL 发射，一族一个模块
  runtime/     越界检查、MLX fallback、计划执行
  passes/      图重写：matcher、layout、fusion、CSE，以及执行顺序
  planning.py  算子合法性与存储计划
  selection.py 原生 kernel 与 MLX fallback 之间的选择
  backends/    Apple Metal backend 与 MLX reference backend
  models/      decoder 静态 Cache 适配层：族注册表、Qwen2、Llama
tests/         IR、lowering、数值对齐、生成、planner
examples/      可运行的模型、kernel 和静态 KV Cache 示例
```

## Roadmap

- **符号 shape 与有限控制流**，同时不削弱静态路径依赖的严格 verifier。
- **shape-aware backend policy**：把逐 op autotune 换成看整张图的选择，以及可复现的 benchmark report。
- **多行 decode**：一张计划一次吃进并验证多个 token，这是投机解码需要的东西。`m` 仍然是编译期常量，所以它落在既有契约里，而不是引入一个新的动态维度。
- **端到端命令**：`aoto generate` 接受 prompt，直接打印或流式输出 token，带停止条件和一种采样，让 `pip install` 之后不写循环也能跑模型。调度和 serving 仍归嵌入编译器的引擎；见 [docs/host-runtime.md](docs/host-runtime.md)。
- **更宽的原生覆盖**：matmul 的 split-K、attention 的 tile 搜索，以及给仍在 MLX 上的算子族做生成 kernel。
- **更多模型族**：decoder 适配层已经按族组织，下一个族是一份配置读取加它自己的怪癖——Mistral 的 sliding window、Qwen3 的 head 排布——而不是再写一遍适配层。长上下文变体需要 RoPE scaling，而当前的 position 表假设的是朴素形式。
- **第三个后端**，跑在别的设备上，那才能证明 `mlx-reference` 这条缝是通用的、而不是 MLX 形状的。ABI 已经带上 target，缺的是一个不说 MLX 的 provider。

## 参与贡献

欢迎提交 issue 和 pull request；[CONTRIBUTING.md](CONTRIBUTING.md)（英文）说明了一次改动需要什么、怎么加一个算子、以及发布流程。修改 lowering 行为时，请附上明确的动机、可复现测试和数值对齐结果。

## License

MIT
