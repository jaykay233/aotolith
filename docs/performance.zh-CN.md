# 性能笔记

一个 decode step 的耗时花在哪里、最近几次重写各买到了什么、以及哪些想法是测过之后才放弃的。README 只留摘要，这一页留测量。

所有数字来自同一台机器：Apple M5 Pro，MLX 0.32，Qwen2.5-0.5B-Instruct，batch 1，除注明外都是 BF16。

## 数字是怎么测的

Qwen2 的每一次测量都用同一段 prompt、同一精度、同样的 `128 + 64` token 设置；`--mlx-baseline` 会在同一个进程里用同一段 prompt 给 `mlx_lm` 自己的 greedy decode 计时，所以基线是在跑 benchmark 的这台机器上测出来的，而不是从别处引用。

两个报告口径比看起来更重要。`decode_async` 把 host 循环摊到多个 token 上，这是服务循环的做法；`decode_sync_each_step` 每个 token 都要同步一次设备，任何服务循环都不会这么做，它存在的理由是隔离设备与 host。每次判断一个重写时，都在同一个会话里对那个 pass 做消融 —— 关、开、关、开 —— 让每一对数字共享同一份机器热状态与负载。绝对值在多次运行之间会差几个百分点，所以这些行适合互相比较，不适合和几周前记录的某次运行比。

`aoto bench` 是回归闸门，它对两类数字的比较方式不同。`input_ops`、`lowered_ops`、`planned_peak_bytes`、`peak_bound_bytes` 是确定的，所以在任何机器上都精确比较，一个多了拷贝或少了一次融合的 pass 会直接失败，没有计时噪声。kernel 数和毫秒数依赖 GPU，只在基线记录了同一设备、且两份报告都至少三次重复时才比较。阈值比较的是 `min_ratio`，既不是 `min_ms` 也不是中位数：竞争只会加时间，所以最好的一次采样才是该看的统计量，而把它除以同一时刻测的参考 matmul，就是把机器从数字里除掉的那一步。这台机器上两段没改过的代码把同一个 0.34 ms 投影的地板测成 0.344 和 0.431 ms，而参考 matmul 从 0.89 变到 1.56 ms；换成比值，两次都是 0.39x。毫秒变慢但比值没变的运行只会打印成 note，不会让闸门失败。默认容差 25%，高于这台机器上两次 best-of-33 地板之间 15% 的摆动——而且只有计时那一半有容差。

参考 matmul 是在一个 workload 的采样之间取的，不是取在它们前后。有一轮把同一个 0.2 ms 的 attention kernel 测成 0.203 ms、紧接着把参考测成 1.15 ms，下一轮同一个 kernel 还是 0.203 ms，参考却是 0.78 ms；把第一对相除得到的 0.177 之后再也复现不了，于是闸门在没改过的代码上报了一次 1.4 倍的「回归」，而那完全是参考自己的时钟。交错采样把分子和分母放进同一个窗口——同一个时钟、同一个热状态、同一个队列——代价是每个采样多一次 matmul。四段连续运行随后把同一个 workload 测成 0.2425-0.2543，而不是 0.177-0.244。

另外每个 workload 在计时开始前都先编译并执行一次，因为一个进程里第一个 plan 要付所有 kernel 的 Metal 编译代价——这一项会把第一个 workload 的数字系统性地翻倍，连它的 warmup 都盖不住。`--rounds N` 会把计时部分重复 N 轮并保留全部采样；用 `--rounds 3` 记录的基线，地板来自三倍的采样，CI 用同一个参数与它比较。

基线是矩阵而不是一个文件：`benchmarks/baselines/<芯片>.json`，一块芯片一份，`--baseline benchmarks/baselines` 会解析到当前这台机器的那份。这正是让毫秒那一半能跨过一台 Mac 的原因——第二台机器是加一份自己的文件，而不是覆盖第一份。没人记录过的芯片照样拿到确定的那一半，取自另一块芯片的文件，并用 note 点名两边的设备；它不是静默通过。所以 op 数与计划峰值的门禁在每台 Mac 上都执行，计时门禁在所有有人记录过的地方执行——这是对覆盖范围的诚实说明，也是 `--check-baselines` 存在的理由：让提交进仓库的文件保持可信。

固定负载里的每一项都对应一个能单独坏掉的机制：一个稠密投影、一次桶派发（96 行请求交给 128 的桶，于是平铺、裁剪和逐桶 lowering 都真的执行）、带 `silu_mul_paired` epilogue 的 gated MLP、`RMSNorm`、causal attention、int4 流式 GEMV，以及两个模型族各自的 prefill 与 decode。后两对是「族被门禁覆盖」而不是「族被声明」的地方：Llama decode 如果丢了免 bias 投影的折叠，或者在 Qwen2 没有多 op 的层上多出一个 op，就会在闸门的确定性那一半里以计数变化出现。

参考后端有自己的一项。`reference_linear_1024x1024` 与 `linear_1024x1024` 是同一次投影，只是 `target="mlx-reference"`，所以它的 `kernels` 按构造就是零，它卡的是解释器路径：不许变的是 `input_ops`、`lowered_ops`、存储计划的两个数字，以及走一遍 MLX 的毫秒地板。这也是发现「某个 op 丢了 fallback」最便宜的办法——Metal 计划只是不再选那个 kernel，而参考计划没有别的东西可以回退。

## Qwen2.5-0.5B，128 个 prompt token，生成 64 个

| 配置 | prefill | decode（每步同步） | decode（异步） | 相对 `mlx_lm`（异步） |
| --- | --- | --- | --- | --- |
| 仅 MLX fallback | 9 650 token/s | 154.7 token/s | 185.9 token/s | 0.77x |
| `--codegen-matmul` | 10 872 token/s | 182.4 token/s | **239.1 token/s** | **1.00x** |
| `--codegen-matmul --codegen-attention` | 12 379 token/s | 184.9 token/s | 239.0 token/s | 0.99x |
| `mlx_lm` 本身 | — | 198.0 token/s | 242.1 token/s | — |

（同一个会话里 5 次重复取中位数；每一行的 64 个生成 token 都与 PyTorch 参考实现一致，且都在 decoder 默认的 `memory="latency"` 策略下运行。这台机器的运行间波动大到一个小时前后同一份代码量到 204 和 239 token/s，所以有意义的是比值，不是绝对值——每一行都对着旁边那行 `mlx_lm` 读。

`相对 mlx_lm` 这一列是异步对异步。这是一次更正：本表早先把这个仓库的异步 decode 拿去比一个每步都用 `mx.eval` 加 `.item()` 等待的基线，而那种等待本身就值大约 15%，由此得出 1.21x。按服务循环的写法计时，`mlx_lm` 是 242.1 token/s，所以 codegen 抹平的是 fallback 路径那 23% 的差距，到达的是持平而不是领先。同步那一列从另一面说的是同一件事：0.93x——到那里 kernel 的选择已经不再决定结果，每步的 host 开销开始决定。

`--codegen-attention` 那行在 decode 上与上一行落在噪声内，prefill 这里甚至略高——attention kernel 在 prefill 形状上更慢的 tile 比这台机器的运行间波动还小。）

prompt 长度对 prefill 的影响远大于对 decode 的影响。在同一个进程里交错计时、两边都测、每格取 5 次的中位数：

| prompt token 数 | prefill：本仓库 / `mlx_lm` | decode 同步 | decode 异步 |
| --- | --- | --- | --- |
| 32 | 3 392 / 5 661 token/s（0.60x） | 181.2 / 198.2（0.91x） | 226.8 / 241.2（0.94x） |
| 128 | 12 029 / 11 122 token/s（1.08x） | 176.2 / 196.3（0.90x） | 230.7 / 241.4（0.96x） |
| 178 | 14 702 / 13 341 token/s（1.10x） | 174.3 / 196.2（0.89x） | 228.1 / 241.4（0.94x） |

32 token 的 prefill 短到让每次 launch 的地板价主导：算术只有约 0.02 ms，却摊在几百次 4-14 µs 的 launch 上，所以成本就是那个次数，而 `mlx_lm` 的图更短。到 128 token，算术开始值得这些 launch，本仓库领先 8-10%。decode 在三个长度上两边都是平的，这正是带宽受限的一步该有的样子，同时它一直落后 4-10%。`examples/compare_mlx_lm.py` 会打出整张网格。

token 一致不等于数字一致。`examples/qwen_logit_distance.py` 把两条路径的最后一个 token logits 分别与 PyTorch 对比，而这个差距取决于 prefill 看到多少 token：

| prompt token 数 | fallback 平均 | fallback 最大 | codegen 平均 | codegen 最大 |
| --- | --- | --- | --- | --- |
| 7 | 0.1056 | 0.6875 | 0.0290 | 0.1875 |
| 8 | 0.0783 | 0.5469 | 0.0451 | 0.2500 |
| 16 | 0.0312 | 0.2031 | 0.0310 | 0.1953 |
| 128 | 0.0596 | 0.4219 | 0.0566 | 0.4219 |

顺着列往下读，而不是横着读：fallback 的平均距离在 7 token 和 16 token 之间差 3.4 倍，所以这里任何一个单点数字描述的都是 prompt，不是模型。这张表支持的结论比「原生路径更近」要窄：在这些长度上 codegen 从不比 fallback 更远离 PyTorch，7 token 时近 3.6 倍，16 token 时持平（0.0312 对 0.0310），到 128 token 的 prefill 时相差在 5% 以内。继续跑 32 个 decode step 的那一次，两者的平均距离是 0.061 对 0.064，顺序反了过来、相差 6% 以内，所以这个排序并不稳定，不足以支撑任何一方的结论。诚实的总结是：打开 codegen 不损失精度——而不是它买到了很多精度——无论哪种说法，它都是吞吐决策，不是精度决策。

本节早先的版本引用的是「平均 0.043、最大 0.38」对「平均 0.081、最大 1.5」，来自一次 7 token 的测量，却没有说明 prompt 就是 7 个 token。两组数字在当前提交上都复现不出来，而且第二组把差距夸大了 2 倍，同时把一个 prompt 长度的产物当成了模型的属性。

和 `mlx_lm` 对比应该看 `decode_async`，因为两者都把 host 循环摊到了多个 token 上。异步路径此前落后于同步路径，现在反超，因为每个 step 只强制一次设备同步。

## MLX 回退路径的精度代价

一个 plan 的 float32 算术只和真正执行它的 kernel 一样精确，而两条路径的精度并不相同。开着 `codegen_matmul` 和 `codegen_attention` 时，两层 Llama 的最后一个 token logits 与 PyTorch 相差 `1.043e-07`——float32 舍入量级。两个都关掉，同一张图跑在 MLX 自己的 kernel 上，漂移是 `1.3e-3`；一次只开一个就能把责任分开（只开 matmul 是 `4.2e-4`，只开 attention 是 `8.5e-4`）。

原因在回退路径，不在 lowering：MLX 自己的 float32 matmul 在这块 GPU 上相对 float64 参考是 `8e-4`，而且从 K=32 到 K=2048 一直平着。累积误差会随 K 增长，所以这是降精度的 matmul——TF32 的形状——而不是越加越偏的求和。

两个值得知道的后果。回退路径在正确性上是安全的：greedy 续写与 PyTorch 逐 token 一致，因为 logits 之间的差大于 `1e-3` 时，`1e-3` 的相对误差不会改变 argmax。它不是逐位精确的路径，所以想把 logits 卡到 `1e-6` 的测试必须编译 kernel，而不是拿解释器去对 fp32 torch。`examples/tiny_llama_compile.py` 就是这段话的可执行版本。

## 一个 decode step 的时间在哪

decode 现在受权重带宽限制，而不是受 host 限制。构建一个 step 只需 0.98 ms Python，GPU 需要约 4.3 ms，所以 host 完全被异步队列盖住；在 plan 展开之前 host 要 6.2 ms，那时它才是瓶颈。一个 step 要流式读取 991 MB 权重（24 层的 `qkv` 2.07 MB、`o_proj` 1.61 MB、一个 17.4 MB 的成对 gate/up 矩阵、8.73 MB 的 `down`，再加 272 MB 的 LM head），按每 token 4.30 ms 算是 230 GB/s。

单个 decode step 的分桶 GPU 开销。测量时把同一组 kernel 连着回放足够多遍，让设备队列一直是满的，于是墙钟时间就是设备时间，host 提交被盖住：

| 分桶 | launch 次数 | 每次搬运 | 单次 GPU 时间 | 带宽 |
| --- | --- | --- | --- | --- |
| LM head GEMV | 1 | 272.3 MB | 662 us | 412 GB/s |
| 带残差的 MLP down GEMV | 24 | 8.73 MB | 32.6 us | 268 GB/s |
| MLP gate+up GEMV + RMSNorm + `silu_mul` | 24 | 17.44 MB | 62.5 us | 279 GB/s |
| 融合 QKV GEMV + RMSNorm | 24 | 2.07 MB | 10.8 us | 193 GB/s |
| attention（单 query） | 24 | 0.10 MB | 7.3 us | 14 GB/s |
| 带残差的 o_proj GEMV | 24 | 1.61 MB | 7.8 us | 207 GB/s |
| RoPE q / k | 48 | 1 KB | 7.8 us | - |
| 加性 attention mask | 1 | 0.2 KB | 26.6 us | - |

加起来是 3.96 ms 设备时间、991 MB，而整个 step 实测 4.30 ms。LM head 加 MLP 矩阵占了其中 74%，而它们是在搬字节，不是在等 launch；其余分散在每次最多搬 2 MB 的 launch 上。做这张表时一个 step 带着 170 次有依赖的 launch。删掉便宜的 launch 不影响总量：把 query 和 key 的旋转合成一次 16 头旋转省掉了 24 次 launch 和 24 个图节点，decode 却是 232.9 token/s 对之前的 232.5，落在运行间波动内，所以没有保留。删掉搬运真实数据的 launch 就会影响：把 `gate` 和 `up` 合成一次 launch 读的仍是同样的 17.4 MB，但 launch 从两次变成一次，交替 A/B 测到 1.031x-1.038x。区别在于这次 launch 搬多少，而不在于有多少次——直到 launch 多到发射本身不再免费，那就是[从 decode step 里砍掉 245 个 op](#从-decode-step-里砍掉-245-个-op)测的东西。

有三种做法会把这个表测成「什么都没测」，而且看起来都很正常。一是每轮丢掉输出的数组，惰性图在到达设备之前就被释放，墙钟时间只覆盖 host 提交，带宽那一列会算出 1 TB/s 以上。二是传全零权重，驱动可以跳过读取，小分桶因此被抬高。一个分桶必须让每一轮的输出都活着，并且喂真实数值，它的数字才有意义；任何超过 `mx.sum` 读同样字节能达到的读数，测的是 harness 而不是 kernel。三是回放本身会往反方向偏：表里同一个分桶的 24 个实例是连着跑的，而真实 step 里每一层都要等上一层，所以 RoPE 和 attention 的单次数字是上界，整表的和只是近似整个 step。

这个方向上有两个实验被测过并放弃。把新的 key/value 直接写到 slot `positions`、用 mask 挑出有效 slot，读起来比滚动窗口更顺，但 MLX 会把 scatter 降级成对整窗口做一遍下标比较，而 concatenate 只是同样字节数的一次拷贝：每个 step 慢 0.18 ms，所以保留滚动。后来变的是滚动的写法——它不再是每层十一个 op，而是[每层一个 `kv_cache_append`](#把-kv-cache-滚动链收成一个算子)，下标算术挪进了 kernel 里。另外，decode token 的旋转位置是 `valid_length - 1` 而不是 `valid_length`；`Qwen2StaticCacheRuntime.decode_token` 曾把后者写进文档并短暂使用，代价是相对参考实现 0.7 的 logit 最大偏差，token 也无法完全对齐。现在 64 个 token 全部一致。

## 为什么小分桶修不动

三个测量指向同一个结论：MLP 分桶没有留下可捡的东西，而且每一个测量都排除了另一种解释。

第一个是 threadgroup 大小。如果 4864 列的输出真的「太窄填不满设备」，给它更多更小的 threadgroup 就该有改善。把同一份已编译的 plan 从 64 个线程扫到 1024 个，这个分桶只动了 2%：

| 每 threadgroup 线程数 | 4864×896 | 151936×896 |
| --- | ---: | ---: |
| 1024（窄分桶当前选择） | 284 GB/s | 395 GB/s |
| 256 | 286 GB/s | 413 GB/s |
| 128 | 289 GB/s | 410 GB/s |
| 64 | 284 GB/s | 408 GB/s |
| 128，宽分桶仍用 64 | 282 GB/s | 291 GB/s |

4864 那个分桶是平的，而 151936 那个在整个扫描里反向动了 30%：它要的是 128 线程的 threadgroup，而窄分桶用的 1024 线程设置会拿走它三分之一的带宽。所以占用率是个真实效应，但它不是区分这两个分桶的原因。

第二个是 MLX 自己的归约。`mx.sum` 读同样的字节，走的是同一条曲线，而且在下面：

| 读取字节数 | `mx.sum` | 本项目的 GEMV |
| --- | ---: | ---: |
| 8 MB | 82 GB/s | 268 GB/s（8.7 MB） |
| 17.4 MB | 180 GB/s | 279 GB/s |
| 35 MB | 225 GB/s | — |
| 140 MB | 295 GB/s | — |
| 280 MB | 296 GB/s | 412 GB/s（272 MB） |

这个投影在每个尺寸上都快过 MLX 的归约，所以这不是本 kernel 实现上的偶然。

第三个是上面同一个表里的受控对比：同一份 warp-per-column kernel，在 272 MB 矩阵上跑到 412 GB/s，在 17.4 MB 上只有 279。差别只有字节数。

三者合起来说明：一次 launch 能达到的速率取决于它读多少字节，而不是它的网格怎么划分。要让整个 step 都跑在 412 GB/s，那 991 MB 必须以几百 MB 一次的 launch 到达，但每一层都被切成五个严格串行的投影 —— `qkv`、`o_proj`、成对 gate/up、`down` —— 而最大的只有 17.4 MB，因为中间的非线性是真屏障。把 `gate` 和 `up` 配对正是利用了这一点，这也是为什么那一次改动测到 1.03x，而删掉 24 次 1 KB 的 launch 什么也没测到。要明显更快只能少读字节，也就是量化权重，而不是写更好的 kernel。

## 权重 int4，以及它到底买到了什么

`compile_qwen2_static_cache(..., quantize=64)` 把每个投影和 LM head 都打包 —— `qkv`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`、`lm_head` —— 而嵌入和 norm 保持原精度。token embedding 是一次只读一行的查表，每个元素 4 bit 在这里既省不到带宽又要损失精度；norm 每层只有一个向量。

格式只在一处定义，`aotolith/quantize.py`，因为有三份代码必须对它有同一个理解。权重仍然存成 `[out, in]`，所以打包矩阵的一行就是一个输出单元。reduction 轴按 `group_size` 切成共享一个 scale 的组，对称量化，scale 取 `max|w| / 7`，四舍五入 —— 组内绝对值最大的元素正好落在 `+7`，因此取负是精确的。每个 `int32` 字装 8 个 nibble，nibble 0 在最低位，两补码，于是这个字就是 kernel 要读的那个 `uchar4` 当成一个整数。scale 放在字旁边，用计算 dtype：bf16 的 scale 与 f32 的差别比一个 int4 步长小三个数量级，而带 f32 会逼 kernel 同时持两种精度。

编译器怎么处理一个打包权重，取决于它乘多少行。单行 matmul —— decode —— *就是* 流式 kernel：`fold_quantized_weights` 把 `matmul(x, dequantize_i4(packed, scales))` 改写成带 `weight_group_size` 的单个 `matmul`，GEMV prologue 读字、用两条指令符号扩展每个 nibble（`(int)(word << 28) >> 28`）、乘 `x` 再乘组 scale、用 float 累加。整个过程中不存在解包后的权重。多行 matmul —— prefill —— 保留解包，因为 128 行时权重流只是乘法里便宜的那一半，int4 只会给算力受限的 pass 加解码工作；此时显式的 `dequantize_i4` 在 `codegen/quantize.py` 里 lower 成一个流式 kernel，而不是八个逐元素 MLX pass 加一次 stack、每层写出好几个全尺寸临时张量。

step 的耗时更跟着 launch 数走，而不是字节数 —— [下一节](#从-decode-step-里砍掉-245-个-op)砍掉 245 个 op，并测出这一项买到了什么。它留下的是一个「每次 launch 约 14 µs 固定成本」的 decode step：把这种 kernel 的单次 launch 隔离出来，得到 `t ≈ 14 µs + 字节 / 373 GB/s`，于是 2.3 MB 的 gate/up launch 花 20.8 µs（110 GB/s），72 MB 的 LM head 花 288 µs（252 GB/s）。threadgroup 尺寸只改几个百分点，更省的 nibble 解码连噪声都没超过，所以下一步是把一次 launch 用来验证多个 token，而不是把单行 kernel 做快。四个测量点见[架构文档](architecture.md#小-launch-的地板是固定的-14-µs不是带宽)。

精度：int4 把每个权重四舍五入到组内 15 个取值之一，logits 因此会移动，而 greedy decode 是这个移动的混沌读数。以同一份取整后的权重跑 float32 CPU forward 作为基准，运行时偏离 0.51 —— 是 18.5 的 logit 尺度的 2.7% —— 而 torch 自己的 bf16 forward 偏离 0.41，即 2.2%。两者都是 bf16 噪声，都不是解码错误。greedy 于是能跟住取整后的 checkpoint 前 13 个 token，然后在一次接近平手的选择上翻转，64 个 token 里对上 16 个。所以 CI 不使用这个模型上的 token 相等：`tests/test_quantized_qwen2.py` 跑一个 2 层模型，那里检查是精确的 —— 每个 decode token 都必须与取整后的 reference 相同 —— 而 `--quantize` 会同时报告 bf16 reference 和取整后的 reference，任何一次运行都不会含糊它跟的是哪一个。

自动调优也不是出路，而且在此之前它对这种模型连跑都跑不起来。`--autotune` 会给每个 op 都求一次 MLX fallback 当参考，但归一化、旋转和 attention 只降级到生成 kernel，所以编译一个 Transformer 会在第一个独立 `rms_norm` 上抛 `NotImplementedError`。现在它对这类 op 只测量自己，记录单条 benchmark，并在 selection reason 里说明这是唯一有效候选。

## 从 decode step 里砍掉 245 个 op

这个模型的 decode step trace 出来是 1909 个 op —— 147 个 cast、339 个 reshape、265 个 slice、169 个 multiply、49 个归一化 —— 而其中真正的 GPU 工作只占 5.7 ms 里的 2.5 ms。其余都是 launch 之间的间隙，平均每次约 4.6 µs，摊在 24 层、每层约十次严格串行的 launch 上：这样一条依赖链，靠一个微秒级提交的 host 线程是盖不住的，GPU 在命令之间空转。把 launch 数压下来比把任何一个 kernel 做快都更值。

最大的一笔来自 PyTorch 写 RMSNorm 的方式。展开写它是 `(x.to(f32).pow(2).mean().rsqrt() * x.to(f32) * weight.to(f32)).to(x.dtype)`，前端忠实记录了五个 op：一次加宽 cast、一次不带 weight 的归一化、为 weight 的第二次加宽 cast、一次 multiply、一次收窄 cast。编译器早就有带 weight 的 `rms_norm`（它降级到同一个 Metal kernel），但没有任何规则把这种五 op 写法匹配上去，于是图里 49 个归一化各自变成五次 launch 和四个全宽临时张量。

`fuse_norm_scale` 匹配这个序列，最后发出一个带 weight 的 `rms_norm`。每 token 因此从图里少掉 245 个 op——49 个被折叠的 multiply 在内，其余是只为喂它而存在的 cast 和 layout 工作。这 245 个由两个 pass 分摊，值得说清楚：196 个在折叠自己发生的位置消失，剩下 49 个是 `fuse_rms_norm_epilogue` 只有在归一化不再挡在乘法前面之后才够得着的 multiply。端到端量下来，把 `fuse_norm_scale` 从管线里去掉，两个变体都是 245 个 op——bf16 从 1909 到 634（lowering 后 683），int4 从 2030 到 682。这个折叠不需要容差，因为它是逐位一致的：输入行和 weight 都以收窄后的 dtype 到达，融合 kernel 用 fp32 做乘积并只舍入一次，正是原来五 op 链做的事。pass 检查这个前提，不满足时不生效——真要一个独立 fp32 行的图保留它——`tests/test_passes_norm.py` 断言的是与未折叠图的逐位相等，而不是容差。

在同一个交替会话里只对这一个 pass 做消融 —— 关、开、关、开 —— 所以每一对数字共享同一份机器热状态与负载：

| | bf16 | int4 g64 |
| --- | ---: | ---: |
| 这一折净减的图 op 数 | 245 | 245 |
| decode（每步同步） | 155.4 → 167.2 token/s（6.44 → 5.98 ms） | 179.0 → 228.3 token/s（5.59 → 4.38 ms） |
| decode（异步） | 199.2 → 232.2 token/s（5.02 → 4.31 ms） | 195.3 → 272.8 token/s（5.12 → 3.67 ms） |
| prefill，128 token | 11 208 → 12 298 token/s | 9 494 → 10 375 token/s |
| 峰值内存 | 4.26 GiB，不变 | 1.99 → 1.89 GiB |

注：op 数这一行引用的是「这一折去掉了多少」，不是 step 的绝对规模。这些计时测完之后，被 trace 出来的基座变大了 —— 今天 bf16 是 1909 个 op、int4 是 2030 个 —— 因为前端记录的 view 算子数量跟着装上的 transformers 版本走；而每一折的净值没变，仍是 245。两列都能用 `examples/qwen_op_counts.py` 打出来。


结论落在前后配对值上：bf16 是 1.08x 和 1.17x，int4 是 1.28x 和 1.40x，prefill 是 1.09-1.10x，代价是每 token 少 245 个 op，没有任何 kernel 改动。折叠之后异步仍然领先同步 —— 4.31 ms 对 5.98 ms —— 说明 host 循环还盖在队列后面，这次折叠并没有把 launch 地板推到台前。每个 op 平摊的成本折叠前是 4.51 µs、折叠后是 4.41 µs：这就是这条发现的形状——这个 step 的耗时等于 op 数乘以一个常数，而常数没有随数量变。

同一个 pass 还暴露出第二个问题，值得记下来，因为修法是收窄的而不是通用的。`_match_rms_norm_consumers` 会把归一化折进读它的那个投影的 *prologue*，让 GEMV 在加载输入行的同时把它们缩放掉。对打包投影这是免费的，它的 prologue 是生成的 Metal。但对稠密投影不免费：它退回 MLX，会为 *每个* 读者重新推导一遍归一化，比只算一次更慢。原先这个匹配器只在投影带打包权重时才吸收归一化，稠密图保持原样——这保住了稠密图的正确取舍，却把同一笔交易在「投影其实会被生成」时让掉了。现在 `rewrite` 多接一个 `native_matmul`，由 `compile_graph` 从 `codegen_matmul` 填进去：编译器会为这个稠密单行投影生成 GEMV 时才折，不会生成时保持独立的归一化。在 Qwen2.5-0.5B decode 上这是 683 个 op 降到 634 个、异步 227.8 升到 233.0 tok/s、同步 171.8 升到 177.8 tok/s，64 个 token 不变；而不问谁 lower 这个乘法直接折，读数是 189.6 降到 138.8，这就是那个条件值 27% 的原因。

## 把 KV-cache 滚动链收成一个算子

decode 每步要往一个固定容量的窗口里写每层的一个 key 和一个 value，而 torch 能表达这件事的写法都比这件事本身长。adapter 把该层 pair 最旧的那个 slot 切掉、把这一步的两行 stack 起来、cat 到窗口上，再把 pair 切开、squeeze 成两半，因为 attention 要分开读。每层十一个 op，再加上把 value 从刚算出的投影行里读回来所需的三个 view —— slice、reshape、transpose。24 层加起来是 312 个 op，占它生效前那个 step 946 个 op 的三分之一，而它们说的其实是一句话：窗口往前挪一格，把新的 key 和 value 写进腾出来的那一格，然后把窗口和它的两半交出来。

`kv_cache_append` 就是这一句话。它吃整个分层 cache、这一步的 key，和打包的投影行，吐出新的 pair 和两个窗口。它的约定写在 `ir/ops/kernels.py` 里，语义就在那：cache 是 `[layers, 2, batch, heads, slots, dim]`，pair 在轴 1 上，`layer` 说明这一步拥有它的哪一层，`value_start` 说明 value 那一半从投影行的哪个位置开始 —— 算子直接在那里读，所以它的形状副本从不需要存在。重写在 `passes/cache.py` 的 `fuse_kv_cache_update`，它刻意很窄：concat 必须把 pair 造得比一个窗口正好长一个 slot，而这个窗口必须是 *同一个* pair 的前缀切片；两个半边必须由一个 slice 和一个 squeeze 解出、形状与算子输出完全一致；重写要删掉的任何值都不能被链外的算子读到。窗口少切两个 slot、value 不是投影来的、pair 还有第三个读者 —— 每一种都保持通用写法。它在流水线里排第一个，因为后面的 pass 会很高兴地把这些必须被看见的 view 整理掉。

后端每层发一个 kernel。对每个 `(half, batch, head)`，所有 slot 是连续的一段，所以「丢掉最旧的一个再追加一个」等价于这段往后挪一行再读：除了最后一格是向量拷贝，末格从 key 张量或投影行里挑。每线程四个元素，bf16 窗口就是八个字节，唯一的分支只判最后一行。它仍然是把整个窗口拷一遍 —— 每层不到一百 KB —— 所以搬运的字节没有变，收益全部来自消失掉的那些 launch。`runtime/mlx_ops` 里的 MLX fallback 为自动调优和 kernel 拒绝的形状算出同样的三个输出，展开式 plan runner 也学会了绑定多输出算子，因此两条路径都不会退回解释执行。

和上一节同样的测法 —— 同一个会话里关、开、关、开 —— 在 0.5B 模型上，bf16 与 int4：

| | bf16 | int4 g64 |
| --- | ---: | ---: |
| 这一折净减的图 op 数 | 312 | 312 |
| decode（每步同步） | 147.9 → 154.6 token/s（6.76 → 6.47 ms） | 228.2 → 251.5 token/s（4.38 → 3.98 ms） |
| decode（异步） | 178.8 → 188.5 token/s（5.59 → 5.30 ms） | 277.2 → 308.7 token/s（3.61 → 3.24 ms） |
| greedy token | 不变，64 个全同 | 不变，64 个全同 |

三分之一图换来了 bf16 的 1.045x / 1.054x 和 int4 的 1.10x / 1.11x，而不是三分之一的吞吐，这就是诚实的读法：这些是整个 step 里最便宜的算子，每个搬的窗口比最小的投影行还小。int4 的比例更大，因为它的 step 本来更短，同样 312 次 launch 占的比例更高。它真正确认的是成本确实按 launch 计 —— 一个只拷 90 kB 的算子和一个搬 2 MB 的算子花费相近，删掉 312 个换来的大致就是 launch 地板预言的那么多。

顺理成章的下一步是这次重写铺好却没有走的那一步：如果写入按 slot 计数寻址，窗口根本不必挪动，kernel 就只剩一次单行写，attention 直接读一个环。这一步在动手之前就被测掉并放弃了。把发出的 kernel 缩到只写末行 —— 完全不拷窗口，所以数值是错的，但 launch 数、grid 和访存形状都与环形版本一致 —— 异步读数是 183.2 tok/s，而真 kernel 是 188.3，各跑两轮（186.4-190.2 与 182.9-183.4）。这里没有可赢的东西：拷贝是 90 kB，而 step 要流 991 MB 权重，它被队列吸收了，重写真正会删掉的是一次 barrier 而不是带宽。拷贝留着，环形属于那个本来就不在乎窗口顺序的 attention kernel。

## 桶边界要付多少

尺寸桶能服务编译器没见过的请求，代价是请求宽度到桶宽之间那些行。这不是一个数，因为要看这次 launch 读的是权重还是激活。M5 Pro 上 `1024 x 1024` 投影，参照是「按请求宽度直接编译」的同一张计划，15 次调用取最小、重复 7 轮：

| 行数 | 桶宽 | 浪费行数 | 直接编译 | 走桶集合 | 比值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 96 | 0% | 0.159–0.281 ms | 0.151–0.275 ms | 0.95–1.03x |
| 96 | 128 | 33% | 0.159–0.281 ms | 0.157–0.244 ms | 0.87–1.06x |
| 4096 | 8192 | 100% | 0.572–0.588 ms | 1.241–1.251 ms | 2.16–2.18x |

第一行是派发本身：挑桶、核对宽度、调用计划，全部落在直接调用时进程内的抖动范围内——正好命中桶宽的请求不需要填充也不需要切输出。第二行是桶集合设计的场景：96 行交给 128 的桶多算 33% 的行，但这次 launch 并不会慢 33%，因为它无论哪种情况都在读 4 MB 的权重，多出来的 32 行 `1024` 宽激活相对它很便宜。第三行不是它设计的场景：几千行之后这条投影是激活受限的，填充出来的行就要按份付钱，还略多一点。

由此得到的规则和 Qwen2 那套桶一致：边界放在流量的聚集处，最宽的桶贴着 host 允许的最大宽度，因为请求与它的桶之间的每一行都算了又丢掉。`plans.stats.snapshot()` 把这笔交易的两半都报出来：每次请求被哪个桶接走，以及多少行被填充掉。

## 峰值内存，以及释放它的东西

plan 的 storage 布局一直知道哪个值死在哪一层。在这个版本之前 runner 用不上这个信息：展开后的函数体是一张惰性图，每个中间值都被引用到 caller evaluate 输出为止，而 MLX 的 evaluator 在丢掉用完的节点之前，是先给图里每个节点分配了输出。于是峰值是所有中间值之和，而不是最大活集。

在一条"64 x 1024 fp32 输入、八段 concat"的链上测得：storage plan 的活集峰值是 134.2 MB，所有中间值加起来是 200.8 MB。

| 策略 | 峰值 | 同步次数 |
| --- | ---: | ---: |
| `latency` | 200.8 MB | 0 |
| `balanced` | 134.5 MB | 4 |
| `minimal` | 134.5 MB | 9 |

机制必须是同步，不能只是 `del`。删掉 Python 名字本身什么都释放不了——节点还在图里；`mx.async_eval` 也释放不了，因为 buffer 在编码命令时就分配好了：同一条链用 `mx.async_eval` 量到 200.8 MB，用 `mx.eval` 量到 134.5 MB。所以 runner 在释放点 evaluate 还活着的值，再删掉刚死掉的名字，这才让 storage plan 的释放从建议变成事实。

这也正是策略要可选的缘故：一个整图只要 0.6 ms 的 decode step 上，同步太贵。`minimal` 在 `qwen2_decode_128` 上量到 5.3 ms，而 `latency` 是 0.6 ms——九倍的工作，换来的只是一个 transient 只有 0.4 MB 的 plan 的界。`balanced` 的阈值（`max(8 MiB, transient/8)`）是为了让策略在合成负载上免费：这些 workload 的 transient 都小于 8 MiB，所以 `balanced` 一次同步都不发，测出来与 `latency` 在噪声内一致。

但它并没有让策略在真实模型上免费。0.5B 的一个 decode step 持有 8.46 MiB 的 transient —— 刚好越过那道地板 —— 于是 `balanced` 每步发一次屏障，而每步一次屏障就足以拿走两成的步频：它冲掉了调用方正用来重叠下一步 host 端建图的流水线。实测 `latency` 是 240.9 token/s，`balanced` 是 196；prefill 发 7 次屏障，是 11 660 对 8 707。阈值自己的估算 `transient/8` 在这里只有 1.06 MB，所以决定结果的是 8 MiB 那道地板，不是那个比例。正是这个测量让 decoder runtime 默认用 `latency`，见[内存](usage.zh-CN.md#内存)。

这个保证付了两笔代价。第一是每个 decode step 多一次"和容量等长"的比较：`decode_mask` 现算窗口，而不是去查一张 `capacity x capacity` 的布尔表。那张表在 4096 容量时是 16 MiB，到 32k 会是 1 GiB，省下的只是几微秒。第二是单桶入口与 decode plan 共享权重；`compile_qwen2_bucketed` 一直跨桶共享，但 `compile_qwen2_static_cache` 把 prefill 和 decode 分开 trace，于是留了两份 checkpoint。`memory_footprint()` 两个都报，`check_memory_budget()` 则拿预算去比"一个 decode step"的占用——一步里有两份 cache，因为它先分配要返回的 cache，才丢掉传进来的那份。

`aoto bench` 现在每个 workload 都记下这些：`planned_peak_bytes` 与 `peak_bound_bytes`（确定性，在任何机器上精确卡）、`peak_extra_bytes`（实测，只在 baseline 同型号设备上按时间容差卡）、`first_call_ms` 与 `compile_ms`（允许 2 倍，因为未改动代码的两次运行之间 shader compiler 会差 1.5 倍）。
