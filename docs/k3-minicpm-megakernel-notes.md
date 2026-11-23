# K3 MiniCPM5-2B megakernel — 2026-11-21 起点与地基层

目标：prefill ≥ 200 tok/s，decode ≥ 20 tok/s（greedy，与 llama.cpp 同量化输出对齐）。
允许特化，不必须用 llama.cpp。

模型（`/dev/shm/mc-q4o.gguf`，GGUF 实测）：
- 42 层，d=2048，16 q 头 / 2 kv 头，hd=128，ff=6144，vocab=130560
- 权重 Q4_0 1415.6 MB；`output.weight` 已是 Q4_0（150 MB，未量化前的 Q6_K 增益已吃到）
- 每 token：2.29 GMAC（层）+ 0.267 GMAC（lm_head）
- decode 每 token 必读 1265 MB（不含 token_embd）

## 基线（llama.cpp 31953bc + spacemit IME2 后端）

运行：`SPACEMIT_DISABLE_TCM=1 GGML_TILED_KPACK=1`，`-fa` 默认。
（`/dev/tcm_sync_mem` 不存在，TCM 不可用。）

| 配置 | t/s | ms/token |
|---|---|---|
| pp128 t4 / t8 | 81.6 / 154.4 | |
| pp512 t4 / t8 | 75.6 / **144.0** | 6.94 |
| pp2048 t4 / t8 | 57.7 / **109.7** | 9.12 |
| tg128 d0 t4 / t8 | **17.35** / 15.77 | 57.6 / 63.4 |
| tg64 d4096 t4 / t8 | **13.75** / 11.69 | 72.7 / 85.5 |

线程数超过 8 无收益：t10 120.1、t12 106.2、t16 138.5（pp512）。原因见下面「核可用性」。

pp512 profile（t8，perf）：GEMM 79%（`_KsubBLK_LPST703` 69.6 + `END20019` 9.4）、
repack 4%（一次性）、A 量化 3.2%、swiglu 1.9%、FA 1.8%、rope 1.7%、binary 1.2%、rms_norm 0.9%。

## 硬件事实（实测，非推测）

1. **只有 8 个核能跑 IME2。** 进程默认 cpuset 是 0-7；写 `/proc/set_ai_thread`
   会把**该线程**永久挪到 8-15（之后设回 0-7 得 EINVAL）。所以 0-7 不能跑 `vmadot`，
   但可以跑普通 RVV。两个簇无法同时做 IME2。
2. **DRAM 流式读上限**：bwtest（单累加链）t1 5.0 / t4 14.7 / t8 18.3 GB/s；
   多 load 在飞时约 33-34 GB/s（历史探针）。decode 端到端已到 22 GB/s。
3. **IME2 m4 GEMM 单核 33-42 GMAC/s**（gemmbench，FFN 形状 M=512 N=6144 K=2048）：
   - m-part t8: 207-217 GMAC/s，n-part t8: **271-277 GMAC/s**（n 划分比 m 划分快 28%）
   - 与 vmadot 微探针的 245 GMAC/s/核相比，差 6 倍 → 卡在喂数据，不在算力。
4. 模型内有效 GEMM 吞吐更高：pp512 t8 → 1.17 TMAC / (3.56 s × 0.79) ≈ 417 GMAC/s。

## 结论：差多少

- **decode 20 t/s** ⇔ 50 ms。权重 1265 MB 在 33 GB/s 下 38 ms，即非 GEMM 开销必须 ≤12 ms
  （现在约 19 ms）。d4096 再加 172 MB KV，需要 KV 单遍流式（现在两遍）且 ~20 GB/s。
- **prefill 200 t/s@pp512** ⇔ 2.56 s，需要 GEMM ≈ 585 GMAC/s（现在 417）+ 其余不退化。
  pp2048 还要额外把 attention 从 2.9 ms/token 压到 1 ms 以下。
- 现有 llama.cpp 后端调参已到顶：GEMM_M_STRIDE 16/32/64/128/512 → 144/143/139/138/134（pp512），
  更大反而更慢；ubatch 512/1024/2048 见 logs/ub.log。

## kernel 上限定位（gemmbench，FFN 形状 M=512 N=6144 K=2048）

| 测试 | 结果 | 含义 |
|---|---|---|
| int4 HP m4，m 划分 t8 | 207-217 GMAC/s | 现网路径 |
| int4 HP m4，n 划分 t8 | 271-290 GMAC/s | 调度上 n 划分更优 |
| 同上 + B 全部指向同一 tile（L1 常驻） | 292 GMAC/s | **不是 B 带宽受限**，只 +9% |
| 同上 + A 也常驻 | 314 GMAC/s | A 也不是瓶颈 |
| int8×int8 m4（`gemm_kernel_i8i8_m4`，K32 正确分块） | 206 GMAC/s | 比 int4 差，字节翻倍没换来速度 |
| vmadot 纯指令探针（4 条独立链） | 245 GMAC/s/核 | 硬件单元远没吃满 |

结论修正：**GEMM 不是 DRAM 受限，是 kernel 内 K32 子块的「vmadot → 打包/转 f32/乘 scale」依赖链
在顺序核上串行停顿**。所以方向不是加行数、也不是加带宽，而是：
1. K 子块软件流水（double buffer：i+1 的 B/A 预取 + i-1 的 scale 合并，与 i 的 vmadot 交错）；
2. 或者 int32 全程累加、只在 superblock 末尾做一次 scale（需要把 per-32 scale 改成 per-256）。

预期：单核 36-52 → 100+ GMAC/s，8 核 800+ GMAC/s，prefill GEMM 1.17 TMAC → ~1.4 s，
加 attention/glue ≈ 2.0 s，pp512 ≈ 250 tok/s。

另外：pp2048 的 attention 已占 2.9 ms/token（≈32%）。长 prompt 要达标，QK/PV 也得走
int8 vmadot 或更好的分块，否则 pp2048 只能到 ~150。

## 下一步（按杠杆排序）

1. 自写调度层（仍是 ggml-free）：n 划分、8 线程绑 8-15、每层一次 barrier、融合 rope/norm/quantize。
2. decode 单遍 KV（K/V 交错布局）+ 融合 attention，目标非 GEMM ≤12 ms。
3. prefill 的 IME2 kernel 加行数（4→8 行/次 B 载入）：B 字节/MAC 减半，是唯一能把 GEMM
   从 417 推到 585+ 的硬改动。
4. 可选：0-7 簇跑 RVV f16 GEMM 做 n 分片（需先把 f16 GEMM 写对，现有探针 0.1 GMAC/s 是布局错）。

## 已做改动

- `llama-cpp-31953bc/ggml/src/ggml-cpu/spacemit/ime.cpp`：新增 `GEMM_M_STRIDE` 环境变量
  （备份在 `/dev/shm/ime.cpp.orig`）。
- 复现命令见 `bench/`，原始数据见 `logs/`。

---

# 第二轮：TCM 实测 + 排除 plain-RVV 路线

## TCM 是当前最大的免费收益（pp512 +21%）

`/dev/shm/tcm.log`（llama-bench -t 8，r=2，Q4_0 `/dev/shm/mc-q4o.gguf`）：

| 配置 | pp512 | pp2048 | tg128 d0 | tg64 d4096 |
|---|---|---|---|---|
| TCM off | 144.94 | 109.56 | 15.53 | 11.55 |
| TCM on | 167.05 ±11.14 | 133.12 | 15.77 | 12.44 |
| **TCM on + `SPACEMIT_TCM_HOLD=1`** | **175.50 ±0.06** | **132.54** | 15.70 | 12.38 |

`SPACEMIT_TCM_HOLD` 跳过的是 `spine_mem_pool_tcm_mem_wait/release` 握手（ime.cpp:1956），
不是 cache 效果；所以 TCM 的真实收益 ≈ 167/144.9 = **+15%**，剩下的 +5% 是省掉握手。

## TCM 的 prefill 路径选择（关键）

`ime.cpp` 里有两个 TCM 路径：

- **Path A**（当前生效）：条件 `gemm_n_stride == gemm_n && per_mb_rows_wsize <= tcm_size`。
  只把**一个 m 块（默认 16 行）的 A** 放 TCM，然后把 96 个 n-tile 全跑一遍。
  → B 每个 layer 被读 `gemm_m/16 = 32` 遍（L2 常驻，但仍是 32× L2 流量）。
- **Path B**：条件 `gemm_n_stride != gemm_n`。把 **B 的 32 列分块（38.9 KB）放 TCM**，
  n 外层、m 内层，每个线程只搬自己需要的 n-tile。
  → **B 每个 layer 只读一遍**，A（594 KB）L2 常驻。

要激活 Path B 必须让 `gemm_m_blocked < nth`，即 `GEMM_M_STRIDE >= 128`（nth=8）。
这和之前 kernel 单独测到的「n 划分比 m 划分 +28%」是同一件事。**这是 prefill 的下一个待测杠杆。**

## 已排除：用 0-7 号核跑 plain RVV int8 GEMM

先纠正一个硬件事实：**这块板子 `vlenb = 32`，即 VLEN = 256**（不是我先前假设的 1024）。
所以 `i8m2 = 64 lane`、`i16m4 = 64`、`i32m8 = 64`。
`vmadot` 之所以强，是因为 `.hp/i4` 一条指令 512 MAC；plain RVV 一条 `vwmacc` 只有 64 MAC。

微基准（`/dev/shm/rawprobe.c`，实测）：

| 探针 | 结果 |
|---|---|
| `vle8.v` + `vadd` | 1.85 ns/iter |
| `vle8.v` + 1 条 `vwmacc.vx`（单累加器） | 3.02 ns/iter → 串行链延迟 ≈ 6.6 cyc |
| 4 组 (vle+vwmacc) 各自独立累加器 | 0.455 ns/组 → **1 cyc/指令，64 MAC/cyc = 141 GMAC/s/核 上限** |
| 1 vle + 8 vwmacc（共享 load） | 3.65 ns / 8 条 → 仍然 ≈1 cyc/条 |

**结论：这套核的向量单元 IPC 瓶颈是 1 条向量指令/周期**，所以吞吐只取决于 MAC/指令。

实际 GEMM（`/dev/shm/i8rvv7.c`，VLEN 已按 256 修正、布局正确、数值校验通过）：

| 线程 | GMAC/s | 每核 |
|---|---|---|
| 1（core 0 @2.2GHz） | 26.8 | 26.8 |
| 2 | 51.2 | 25.6 |
| 4 | 69.1 | 17.3 |
| 8 | 77.3 | 9.7 |
| 16（0-15 全用） | 94.6 | 5.9 |

卡在 27 GMAC/s/核的原因：**每条 `vwmacc.vx` 都要一个 `lbu` 标量（A[m][k]）**，
顺序核上 load-use 停顿吃掉了 5×。而 `lbu` 数量与 `vwmacc` 数量是 1:1，无法摊薄
（把 A 预广播成向量要额外 64 MB/层 内存流量，更差）。

对比：现成 IME2 kernel 在 8 核上**有效 467 GMAC/s**（144 tok/s 基线反推）。
即使把 16 核全给 plain RVV 也只有 94.6 GMAC/s —— 差 5 倍。
**所以 prefill 必须留在 IME2/vmadot 上，X100 簇不参与 GEMM。**

顺带两个硬件事实：

- **X100(0-7) 主频 2.2 GHz，A100(8-15) 只有 1.8 GHz**。同一份 plain-RVV kernel
  单核 26.8 vs 17.6 GMAC/s。
- AI 锁（`/proc/set_ai_thread`）是**按线程**的，plain RVV 不需要它，
  所以理论上 16 核可以同时干活 —— 但如上，GEMM 上不划算。

## 复现命令

```sh
# TCM 三档对比
/dev/shm/tcm.sh
# m_stride 扫描
/dev/shm/mstride.sh
# decode 线程数扫描
/dev/shm/dec.sh
```

---

# 第三轮：目标改为 10K 上下文 —— 瓶颈彻底重定位

用户指定：**所有优化按 10K 上下文做**。这改变了瓶颈结构，结论也随之改写。

## 硬件事实修正（先纠正上一轮的错误）

逐核实测 `__riscv_vlenb()`（`/dev/shm/vlenprobe.c`）：

| 核 | vlenb | VLEN | 频率 |
|---|---|---|---|
| 0-7 (X100) | 32 | **256 bit** | 2200 MHz |
| 8-15 (A100) | 128 | **1024 bit** | 1800 MHz |

上一轮我说"板子 VLEN=256"，那是因为探针没绑核、跑在 X100 上。**A100 确实是 1024**。
这也说明上一轮 plain-RVV 的 A100 数据（写死 `lanes=64`）只用了 1/4 向量宽度，数字无效；
但"IME2 比 plain RVV 强"的结论不变（见下）。

`ime.cpp:1300` 的 `supported_vlen = (__riscv_vlenb() == 128)` 正好只在 A100 成立，
所以 attention 快路径确实生效了 —— 它慢的原因不是走错分支。

## 10K 基线（`SPACEMIT_TCM_HOLD` 已默认开启）

| 指标 | 值 |
|---|---|
| pp512 | 175.4 tok/s |
| pp1024 | 158.7 tok/s |
| pp2048 | 132.6 tok/s |
| **pp10240（外推）** | **≈ 15 tok/s** |

`IME_PROF` 剖析（thread-sum，2 个 rep）：

| | gemm | fa | fa 占比 |
|---|---|---|---|
| pp512 | 36.9 s | 6.0 s | 14% |
| pp2048 | 147.2 s | 83.7 s | **36%** |

gemm 随 n 严格线性（4.29x / 4x），fa 随 n² 增长。`FA_PROF` 分节（pp2048）：

```
qk 47.3%  pv 27.1%  mask 11.1%  sm 9.7%  kx 4.8%   ->  28.1 ns per (row x kv)
```

折合 attention 只有 **~9 GMAC/s/核**，是 vmadot 峰值（245）的 1.5%。
外推到 10240：attention ≈ 64 ms/token，占 prefill 的 93%。

反推 GEMM 真实吞吐：pp2048 的 gemm 147.2 s / 8 线程 / 4096 token = 4.49 ms/token，
2.29 GMAC/token → **63.8 GMAC/s/核**（8 核 510 GMAC/s），是 vmadot 峰值的 26%。

## 根因：f16→f32 扩宽 FMA 是模拟指令

`rvv_qk_dot_tile_f16_x4` 等内层循环全部建立在 `vfwmacc`（f16×f16→f32）上。
逐条微基准（`/dev/shm/fmaprobe.c`，绑核 8）：

| 指令 | ns/指令 | MAC/周期 | GMAC/s/核 |
|---|---|---|---|
| `vfmacc.vf` f32 m4 | 2.25 | 31.6 | 56.9 |
| `vfmacc.vf` f32 m8 | 4.45 | 31.9 | 57.5 |
| **`vfwmacc.vf` f16→f32 m2** | **17.80** | **4.0** | **7.2** |
| **`vfwmacc.vv` f16→f32 m4** | **17.81** | **8.0** | **14.4** |
| **`vfwmacc.vf` f16→f32 m4** | **72.50** | **2.0** | **3.5** |

**扩宽 FMA 比原生 f32 FMA 慢 8-20 倍**，且越宽越慢 —— 这只能是软件模拟。
整个 attention 就建在这个指令上，所有分块/流水优化都救不了。这就是 9 GMAC/s 的来源。

## 第二个根因：向量 load 是延迟受限，不是带宽受限

`/dev/shm/vbw.c`：

| 形状 | GB/s | 周期/指令 | B/周期 |
|---|---|---|---|
| `vle32.v` m1 (128 B) + 依赖 add | 10.7 | 21.6 | 6 |
| `vle32.v` m2 (256 B) | 21.4 | 21.5 | 12 |
| `vle32.v` m4 (512 B) | 39.7 | 23.2 | 22 |
| `vle32.v` m8 (1024 B) | 49.6 | 37.2 | 28 |
| **4 条独立 `vle32.v` m4 并发** | **168.0** | — | **93** |

单条向量 load 的**延迟 ~21 周期**，与宽度无关。只有 ≥4 条在飞才能到 93 B/周期（168 GB/s）。
所以任何"每步一条 load 然后立刻用"的循环都会浪费 20 周期。

## 第三个事实：这台核对依赖链/调度极端敏感

同一个 QK 循环，只是排布不同（`/dev/shm/isolate.c`，4 个 f32m4 累加器，16 条 FMA 一组）：

| 变体 | GMAC/s | MAC/周期 |
|---|---|---|
| 无 load（k 常量） | 25.6 | 14.2 |
| 1 条 load（每 FMA 一条，同地址） | **68.3** | **37.9** |
| 4 条 load | 19.1 | 10.6 |

纯向量 FMA（`vfmacc.vv`，无 load，6 条独立链）可达 **84.9 GMAC/s/核**。
但一旦和 load 混排，同一份逻辑可在 19-68 之间摆动 3.5 倍。
**结论：靠 intrinsics + GCC 拿不到稳定峰值，必须手写调度（inline asm 或定制汇编）。**

## 结论：10K 上下文下各路径的天花板

以 8 核 A100、18 亿周期/秒计，pp10240 要 200 tok/s 就是 **5.0 ms/token 总预算**：

| 组件 | 当前 | 预算 | 需要的倍数 |
|---|---|---|---|
| GEMM（26.2 TMAC） | 4.49 ms | ~2.5 ms | 1.8x |
| attention（9.0 TMAC） | 63.9 ms | ~1.5 ms | **42x** |
| glue | ~0.5 ms | ~1 ms | — |

注意 attention 在 10K 是 **9.0 TMAC**（= n²·d_model·n_layers，已含 causal 折半），
所以 attention 不再是配角而是主角。

各方案叠加后的可达上限：

| 方案 | attention | GEMM | 合计 | pp10240 |
|---|---|---|---|---|
| 现状 | 9 GMAC/s/核 | 64 | | ~15 tok/s |
| 换 f32 FMA（保留现有结构） | ~25 | 64 | | ~40 tok/s |
| f32 + 手写调度 | ~60 | 64 | | ~90 tok/s |
| **全 int8 vmadot** | 245 | 150+ | 35 TMAC / 1.96 TMAC/s ≈ 18 s | **~500 tok/s 理论上限** |

**只有全 int8 vmadot 能到 200。** f32 路线（就算手写到 60 GMAC/s/核）上限约 90 tok/s。

## 下一步的确定设计（int8 attention）

QK 用 `vmadot ... i8`（int8×int8，`ime2_kernels.cpp:4993 gemm_kernel_i8i8_m4` 里可见），
PV 同样。要点：

1. **Q/K 量化**：per-32-block f16 scale，与 GEMM 的 A 布局复用（`quantize_a_4row_i8` 已存在）。
   V 需要 per-row scale。attention 对量化比 GEMM 敏感，K 用 int8、V 可保留 f16 或 int8。
2. **GQA 分块**：`rk2 = 8`（16 q heads / 2 kv heads）。一个 tile 取 8 个 q head × P 个位置，
   K/V tile 只载入一次 → 复用 8×P。这是把 KV 流量从 1.1 TB/层降到 ~140 GB/pass 的关键。
   （现有 kernel 已有 `fa_kv_group`，但它按 row 分，64 row = 4 位置 × 16 head，复用只有 4。）
3. **load 流水**：每步至少 4 条独立向量 load 在飞。
4. **causal 跳块**：整块被 mask 的 KV tile 直接跳过；整块无 mask 的不做 mask 运算（可省 mask 的 11%）。
5. `GGML_FA_TILE_KV` 从 64 提到 128（scratch 已按 `GGML_FA_TILE_KV_MAX=128` 分配）。

## 本轮落地的改动

- `ime.cpp`：`SPACEMIT_TCM_HOLD` **改为默认开启**（`=0` 可关）。
  实测默认配置 pp512 = **175.40 ± 0.08**，而 `SPACEMIT_DISABLE_TCM=1` 是 144.94。
  TCM 的真实收益 ≈ +21%（上一轮把 ±11 噪声的 167 当成分界，这里更正：
  收益来自 TCM 本身，`hold` 只省握手，单独对比是 175.0 vs 175.4）。
- 备份：`/dev/shm/ime.cpp.orig`、`/dev/shm/ime.cpp.bak2`。

## 复现命令

```sh
# 每核 VLEN
/dev/shm/vlenprobe
# f16 扩宽 FMA 与 f32 FMA 的对比
/dev/shm/fmaprobe
# 向量 load 带宽
/dev/shm/vbw
# load/FMA 争用隔离
/dev/shm/isolate
# 10K 基线（含 IME_PROF / FA_PROF）
cd /home/bianbu/workspace/llama-cpp-31953bc/build-smt/bin
export LD_LIBRARY_PATH=$PWD GGML_TILED_KPACK=1
IME_PROF=1 ./llama-bench -m /dev/shm/mc-q4o.gguf -p 2048 -n 0 -t 8 -r 1
FA_PROF=1  ./llama-bench -m /dev/shm/mc-q4o.gguf -p 2048 -n 0 -t 8 -r 1
```

## 附：decode 线程数扫描（d=0，TCM HOLD 默认开启，n=32，r=2）

| 线程 | tg32 tok/s |
|---|---|
| 3 | 14.45 |
| 4 | 16.01 |
| 5 | 16.02 |
| 6 | 16.22 |
| 8 | 16.04 |
| 10 | Aborted |
| 12 | Aborted |

两点：

- d=0 时 t4-t8 基本持平（16.0-16.2），t6 略高但差异在噪声边缘。
  之前某轮测到 t4 明显好于 t8，是旧代码态，现已被 TCM HOLD 抹平。
- **t>8 直接 Abort**，印证可用 AI 核只有 8 个（8-15），decode 无法靠加线程扩展。

注意这是 **d=0**；10K 上下文下 KV cache 读取主导，数字会显著更低，
decode@10K 的优化目标（≥20 tok/s）与这里不是同一问题。

---

# 第四轮：10K 真基线 + 可信的分项剖析

## 纠正两处错误

1. **pp10240 = 58.10 tok/s（实测）**，不是我前面外推的 15 tok/s。之前那个 15 是
   由一次被中途杀掉的运行估的，作废。外推链 `pp2048=132.87 → pp4096=100.16` 再按
   gemm∝n、fa∝n² 推到 10240 得 57.3，实测 58.10，误差 1.4%。
2. **`IME_PROF` 的 thread-sum 不是墙钟**。它把 8 个线程都累加、还包含 barrier 等待，
   除以 8 后甚至超过总时间（自相矛盾）。已加 `IME_PROF_WALL=1`：只计线程 0。

注意 llama-bench 每个配置会跑 **warmup + 正式**两遍，所以 [IME_PROF] 数字要除以 2。

## 可信的墙钟分项（thread0 wall，除以 2）

| | GEMM | attention | 合计 | 实测墙钟 | GEMM | FA |
|---|---|---|---|---|---|---|
| pp2048 | 9.20 s | 5.23 s | 14.43 s | 15.41 s | 60% | 34% |
| pp4096 | 18.41 s | 20.51 s | 38.92 s | 40.89 s | 45% | 50% |

GEMM 严格 ∝n（2.00x），FA ∝n²（3.92x）。线性外推到 10240：

| 组件 | 10240 预估 | 占 176 s 的 |
|---|---|---|
| GEMM | 46.0 s | 26% |
| attention | 130.8 s | **74%** |
| glue | ~2 s | 1% |

合计 178.8 s，实测 176.2 s —— 模型自洽。

## 真实吞吐（每线程）

- GEMM：26.2 TMAC / 46.0 s = **570 GMAC/s 聚合 = 71 GMAC/s/线程**（vmadot 峰值 245 的 29%）
- attention：9.02 TMAC / 130.8 s = **69 GMAC/s 聚合 = 8.6 GMAC/s/线程**（峰值的 3.5%）

## 已否掉的方案（有实测证据）

| 方案 | 结果 | 结论 |
|---|---|---|
| B 分块进 TCM（Path B） | pp2048 132.7→98.2，pp4096 100.1→76.3 | **-26%，否掉** |
| m_stride 16/32/64 | pp2048 与 pp4096 三者完全相同 | 都走 Path A，无收益 |
| n 外层 m 内层（仅改循环序） | 同上 | 复用已在 L2 发生，无收益 |
| 2D (m,n) 分块（合成基准，忠实 count_n） | 276.4 → 344.9 GMAC/s 聚合 | 只 **1.25x**，不足以解决问题 |
| plain RVV 跑 GEMM | 16 核 94.6 GMAC/s | 比 IME2 差 5x，否掉 |

2D 分块只有 1.25x，说明 GEMM **不是简单 DRAM 受限**——真实路径把 A 放 TCM 后已达
71 GMAC/s/线程，反而比我的合成基准（无 TCM）快 1.85x。GEMM 要再上台阶得改 kernel
（4 行 → 8 行，B 字节/MAC 减半），不是改调度。

## 关键发现：`vmadot ... i8` 与 int4 版同为 512 MAC/指令

`ime2_kernels.cpp:4993 gemm_kernel_i8i8_m4` 里：

```
vmadot v16, v2, v24, i8      // 8 条覆盖 4 行 x 32 列 x 32 k = 4096 MAC
```

即 int8 x int8 的 `vmadot` 也是 **512 MAC/指令**，和 `vmadotsu.hp ... i4` 一样。
而 attention 现在用的 `vfwmacc`（f16→f32）只有 64-128 MAC/指令且每次 17.8 ns。

所以 FA 要达标不需要发明新东西：**把 QK 和 PV 都表达成 int8 GEMM**，
直接复用现成的 `vmadot i8`：

- QK：`S[rows][kv] = Q[rows][128] x K^T[128][kv]` —— 就是 M=rows, N=kv, K=128 的 GEMM
- PV：`O[rows][128] += P[rows][kv] x V[kv][128]` —— M=rows, N=128, K=kv 的 GEMM

A 侧用现成的 `quantize_a_4row_i8`（Q、P 都按 4 行量化，per-32-f16-scale），
B 侧按 `q8_0_32x256` 的 repack 布局打包 K、V。

预期：dot 吞吐从 8.6 提到 100+ GMAC/s/线程，FA 130.8 s → ~20 s。

## 达标所需的倍数（200 tok/s @ 10240 = 51.2 s 预算）

| 项 | 现在 | 需要 | 路线 |
|---|---|---|---|
| GEMM | 46.0 s | ~25 s (1.84x) | 4 行 → 8 行 kernel，B 字节/MAC 减半 |
| attention | 130.8 s | ~22 s (5.9x) | 换成 int8 `vmadot i8` |
| glue | 2 s | 2 s | — |

只做 FA：176→68 s ≈ 150 tok/s。
只做 GEMM：176→155 s ≈ 66 tok/s。
**两个都要做才能过 200。**

## 本轮落地的改动（ime.cpp）

1. `SPACEMIT_TCM_HOLD` 默认开启（前一轮）。
2. `IME_OP_PROF=1`：按算子名累加胶水算子耗时（按实测只有 ADD 0.09 s/rep，可忽略）。
3. `IME_PROF_WALL=1`：只计线程 0，得到真墙钟。
4. 备份：`/dev/shm/ime.cpp.orig`、`.bak2`、`.bak3`、`.bak4`。

## 复现命令

```sh
cd /home/bianbu/workspace/llama-cpp-31953bc/build-smt/bin
export LD_LIBRARY_PATH=$PWD GGML_TILED_KPACK=1
M=/dev/shm/mc-q4o.gguf
# 10K 基线
./llama-bench -m $M -p 10240 -n 0 -t 8 -r 1
# 真墙钟分项
IME_PROF=1 IME_PROF_WALL=1 ./llama-bench -m $M -p 4096 -n 0 -t 8 -r 1
# 按算子
IME_PROF=1 IME_OP_PROF=1 ./llama-bench -m $M -p 512 -n 0 -t 8 -r 1
```

---

# 第五轮：`vmadot i8` 实测 —— attention 的 17x 找到了

## 纯指令吞吐（`/dev/shm/vmi8.c`，钉核 8，vlenb=128）

```
raw vmadot i8     :   457.9 GMAC/s (254.4 MAC/cyc)   <- 每 2 周期发一条，512 MAC/条
raw vmadot i8     :   459.8 GMAC/s (255.4 MAC/cyc)
qk_i8 64 rows/tile:   149.0 GMAC/s ( 82.8 MAC/cyc)   <- 含 vle32/vl8r 的真实 QK 循环
qk_i8 64 rows/tile:   149.1 GMAC/s ( 82.9 MAC/cyc)
```

**修正前几轮的一个低估值**：我记的「vmadot 纯指令 245 GMAC/s」是测低了。真实是
**457.9 GMAC/s/核 = 254 MAC/cyc**。这意味着：
- int4 GEMM 的硬件峰值也是 457 GMAC/s/核，现网 71 GMAC/s/核只用了 **15.5%**；
- attention 现在 8.6 GMAC/s/核，是峰值的 **3.4%**。

## FA 改用 int8 的收益（按实测数）

| 方案 | 单核吞吐 | 8 核聚合 | FA @10240 | 10K 总时间 | prefill |
|---|---|---|---|---|---|
| 现在（vfwmacc f16→f32） | 8.6 | 69 | 130.8 s | 176 s | **58.1**（实测） |
| 自写 qk_i8 循环 | 149 | 1192 | 7.6 s | 53.6 s | ~191 |
| 库函数 `gemm_kernel_i8i8_m4` | **206** | 1648 | **5.5 s** | 51.5 s | **~199** |

注意最后一行：**光把 attention 换成现成的 `gemm_kernel_i8i8_m4`，10K prefill 就基本到 200**。
GEMM 那 1.84x 是「越过 200 并留余量」用的，不是前提。

## 为什么用库函数而不是自己写 vmadot

`vmadot` 的 A/B 寄存器打包格式不直观。用全 1 的 B 探语义
（`/dev/shm/vmi8sem.c`）得到的结果只有 lane 0,1,8,9,16,17,24,25 非零：

```
d[ 0]=1280  d[ 1]=5120  d[ 8]=2560  d[ 9]=14592
d[16]=9280  d[17]=64320 d[24]=-6848 d[25]=23616
```

说明它不是「32 列一一对应 32 个 lane」，中间还有 `vupack`/`vpack` 的重排。
`gemm_kernel_i8i8_m4` 已经把这套重排做对且实测 206 GMAC/s，**直接复用**，
不要重新发明。Q 和 P 走现成的 `quantize_a_4row_i8`（per-32 f16 scale + i16 sum），
K 和 V 走 `q8_0_32x256` 同款 repack。

## 形状映射

- QK：`S[rows][kv] = Q[rows][128] × K^T[128][kv]`
  → `count_m=rows(4的倍数), count_n=kv_tile, k_blks=128/256`
- PV：`O[rows][128] += P[rows][kv] × V[kv][128]`
  → `count_m=rows, count_n=128, k_blks=kv_tile/256`

kv 分块取 256（一个 superblock）最省重排开销；rows 取 32-64 让 A 的量化开销摊销。

## 注意：本机 vmadot 语义探针未完成

`vmi8sem.c` 的 A 布局假设（`a[row*32+k]`）是错的，所以那一组数字只能用来证明
「不是简单一一对应」，不能用来反推格式。后续一律以库函数行为为准。

---

# 第六轮：纠正 vfwmacc 的倍率，锁定 int8 是唯一出路

## 重要更正：vfwmacc 只比 vfmacc 慢 2 倍，不是 8-20 倍

前几轮我用 `/dev/shm/fmaprobe.c` 得到 `vfwmacc m2->m4 = 17.8 ns`，据此判断「慢 8 倍」。
那个数字被**单累加器的依赖停顿**污染了。加 4 条独立累加链重测（`/dev/shm/fma2.c`）：

| 形式 | ns/指令 | 吞吐 | MAC/cycle |
|---|---|---|---|
| `vfwmacc.vf` e16m1→f32m2 (vl=64) ← QK 现用 | 4.45 | 14.4 GMAC/s | 8.0 |
| `vfwmacc.vf` e16m2→f32m4 (vl=128) | 8.91 | 14.4 GMAC/s | 8.0 |
| `vfmacc.vf` f32m2 (vl=64) | 2.23 | 28.7 GMAC/s | 16.0 |
| `vfmacc.vf` f32m4 (vl=128) ← PV 现用 | 4.45 | 28.8 GMAC/s | 16.0 |

结论：**扩宽 FMA 是原生 f32 FMA 的一半速度**（8 vs 16 MAC/cyc），不是模拟陷阱。
8.0 MAC/cyc 正好是 512-bit 输入带宽的极限（f16 输入每 cycle 只能喂 8 个 MAC）。

## 由此推出：QK 的 f16 路线已经到顶

- QK 实测 9.2 GMAC/s/核，vfwmacc 上限 14.4 → **已达 64%**。
- 把 K 转 f32 换成 `vfmacc` 试过（`GGML_FA_QK_F32=1`）：pp2048 **132.76 → 123.44**，
  **反而慢 7%**。原因：凭空多出一份 64 KB/线程的 f32 K 副本（8 线程 = 512 KB），
  挤掉 L2，而 FMA 只快 2x、还被放不进缓存的代价吃掉。
- 该改动已回滚（`/dev/shm/rvk.bak5`）。

**所以 QK 只能换 int8 vmadot。** PV 用的是 `vfmacc`（28.8 上限），实测 16 = 56%，
int8 同样能让它到 65。

## int8 B 布局已完全测通（不用再猜）

用 1024 次单字节探针（`/dev/shm/k4/i8map.c`）把 `ime2::gemm_kernel_i8i8_m4` 的
B 字节→列映射整个测出来，结论极其简单：**列主序**，`col = j/32`，`k = j%32`。
每 32 列 × 32 k 一块：

```
[ 32 × f16 scale (64 B) ][ 32 列 × 32 k int8，col*32 + k 排布 (1024 B) ] = 1088 B
```

A 侧只有 `spacemit_kernels::rvv::quantize_a_4row_i8` 对（152 B/块 = 4×f32 scale +
4×i16 sum + 4行×32k int8）。**用 `ime1::quantize_a_4row_i8` 会全错** —— 它是
144 B 的交错布局（`offset = (4-row)*4 + row*8`），那是给 ime1 vmadot 用的。

验证（`/dev/shm/k4/i8au.cpp`）：与 int8 反量化参考的误差 **2.97e-6**（纯 float 舍入），
与 f64 真值差 0.197（这就是 int8 量化误差本身）。

## 库函数 m4 的吞吐天花板：65 GMAC/s

`gemm_kernel_i8i8_m4` 的 `count_m` 被忽略，一次只算 4 行；循环是 `for ni` 外层、
asm 一次处理 4 行。所以 B 块每 4 行就要重载一遍：

| 一次调用形状 | 吞吐 |
|---|---|
| 4 行 × 32 列 × 128k | 31.8 GMAC/s |
| 4 行 × 64 列 × 128k | 46.9 GMAC/s |
| 4 行 × 128 列 × 128k | **65.4 GMAC/s** |

而手写循环（B 载入一次、复用 16 个行组）在纯指令下能到 254 MAC/cyc = 457 GMAC/s。
所以自写 kernel 的收益主要是 **B 载入复用**，不是指令本身。

## 修正后的预期（10K，pp10240 基线 58.10）

| | 现在 | int8 后 | 说明 |
|---|---|---|---|
| QK (47%) | 61.5 s | 8.7 s | 9.2 → 65 GMAC/s/核 |
| PV (27%) | 35.3 s | 8.7 s | 16 → 65 GMAC/s/核 |
| mask (13%) | 17.0 s | 17.0 s | 与 dot 无关 |
| softmax (9.4%) | 12.3 s | 12.3 s | 与 dot 无关 |
| pack (3.1%) | 4.1 s | 4.1 s | |
| FA 合计 | **130.8 s** | **50.9 s** | |
| GEMM | 46.0 s | 46.0 s | 未动 |
| **总计** | 176 s (58.1 t/s) | **99 s (~103 t/s)** | |

**必须更正我上一轮说的「光换 FA 就到 199」** —— 那是假设整个 FA 都能按 MAC 加速，
但 mask+softmax+pack 占 26%，不会跟着变快。

要真到 200 tok/s（51.2 s 预算），还需要：
- GEMM 46 → ≤25 s（4 行 → 8 行 kernel，或消掉 K32 依赖链）
- mask+softmax 从 29.3 s 压到 ~10 s（现在只有 0.14 elem/cycle，有 50-100x 空间）

## 复现命令

```sh
# int8 B 布局验证 + 吞吐
ssh spacemit-k3 'cd /dev/shm/k4 && ./i8au && ./i8map | head -5'
# FMA 形式对比
ssh spacemit-k3 '/dev/shm/fma2'
```

---

# 第七轮：int8 QK 接进真 kernel —— 负结果

## 做了什么

按布局结论把 QK 换成了 `ime2::gemm_kernel_i8i8_m4`（`GGML_FA_ATTN_I8=1` 开启）：
Q 用 `rvv::quantize_a_4row_i8` 量化；K 用 `rvv_transposed_s16_mn_to_nm` 把
`[k][col]` 转成 `[col][k]` 再逐列 amax 量化。改动默认关闭，管线保留。

## 结果：qk 段变慢 25%

```
F16  : [FA_PROF] qk=47.3%  合计 723 ms   pp2048 132.87
INT8 : [FA_PROF] qk=52.2%  合计 866 ms   pp2048 126.89
```

qk 段的绝对时间 361 ms → 452 ms。**库函数在真实 tile 里只有约 8 GMAC/s**，
比微基准里的 28-31 低得多。原因是它 `count_m` 被忽略、一次只算 4 行，
每个 (4 行 × 32 列 × 32k) 都要重跑一遍 prologue + `vupack` + `vpack` + `vfcvt`
+ `vfmul`，而 8 条 `vmadot` 只占 16 cycle。在 `ldc=KVT=64` 的真实布局下
（KQ 16 KB、再加上 K_f16/Q_f16/att_k_i8/att_q_i8）L1 压力也让重载变贵。

## 这一轮的净结论

- 基线未变：**pp512 175.59 / pp2048 132.78 / pp10240 58.10**（int8 默认关）。
- `vmadot i8` 纯指令 457 GMAC/s、无 epilogue 的手写循环 149 GMAC/s，
  但**这两者都不是数值正确的实现**（没有 vupack/scale）。正确实现只能在
  库函数（8 GMAC/s 实际）和自写完整 kernel 之间选。
- 所以 int8 要真正生效，必须自写 kernel，核心是：
  1. 把 `vupack` 的 B 去交错**提前到打包阶段**（内存里就存成 vmadot 可直接吃的格式）；
  2. k-block 累加留在寄存器里（库函数已是如此），但把 B 的 load+vupack 提到
     行组循环外面，靠 4 行 × 多个行组共享；
  3. `vpack/vfcvt/vfmul` 每 (行组, n 块) 只做一次，不是每 k-block 一次 ——
     需要把 per-32 scale 改成 per-token scale（精度略降，但省掉 3/4 的 epilogue）。

## 下一个更高性价比的目标：mask + softmax（占 FA 的 26%）

10K 下 mask 17.0 s + softmax 12.3 s = **29.3 s**，而这两段是纯 f32 向量活，
理论该有 30+ elem/cycle。实测：

- mask：67.1 G pairs / 17.0 s / 8 核 = **0.27 elem/cycle**
- softmax：同量级

这是 50-100 倍的差距，且**没有任何布局或数值风险**。修好它就是 -26 s，
比 int8 QK 的自写 kernel 便宜得多。

---

# 第八轮：mask+softmax 融合生效，10K 真数字确定

## 改了什么

把 tiled FA 里 mask/softmax 的 4-5 遍扫描压到 2 遍，并把每行的串行归约改成 4 行交叉发射：

- 删除 `mask_all_zero`（它自己的 min+max 扫描比省下的 add 更贵）。
- mask 直接从 f16 读，删掉 16 KB 的 f32 `mask32` 中转缓冲。
- `rvv_mask_add_max_x4`：mask 加 + softmax 行 max 合成一遍，4 行 ILP。
- `rvv_exp_sum_x4`：exp + 行求和合成一遍，4 行 ILP。
- `can_skip` 改用 `rvv_f16_rowmax_x4` 直接扫 f16，4 行 ILP。
- 所有消费者从 `KVT` 改成 `kv_tile`，删掉 -inf 填充的标量循环。

## 效果（实测）

| | 之前 | 之后 |
|---|---|---|
| pp512 | 175.59 | **176.19** |
| pp2048 | 132.78 | **135.19** |
| pp4096 | 100.16 | **103.30** |
| pp10240 | 58.10 | **60.66** |
| tg64 @10K | — | **15.44** |

FA_PROF 里 mask 段从 49.1 ns/(tile) 降到 29.4（**-40%**）。pp10240 +4.4%。

`sm` 段反而 77.3 → 110.0，因为 `rvv_expf_approx_f32m2` 内部有分支（`vcpop_m`），
4 份内联展开后编译器无法交叉调度，反而串行化。这一处还需再改（见下）。

## 10K 目标差距

| | 实测 | 目标 | 需要 |
|---|---|---|---|
| pp10240 | 60.66 | 200 | 3.30x |
| tg64 @10K | 15.44 | 20 | **1.30x** |

**decode 只差 1.3x，是真正够得着的那个。**

## decode @10K 的账（为什么只差 1.3x）

- 权重 1.32 GiB / token，@33 GB/s = **40 ms**
- KV cache：28 层 × 2 × 4 kv头 × 128 × 2 B = 57 KB/token → @10240 = **583 MB**，
  单遍 @33 GB/s = **17.7 ms**；若当前读两遍 = 35 ms
- 实测 64.8 ms/token = 1000/15.44

即 40（权重）+ 35（KV 两遍）= 75 ms 的量级。若把 KV 改成单遍流式
（纯布局改动、**不改数值**）→ 40 + 17.7 = 57.7 ms ≈ **17.3 tok/s**；
再叠 int8 KV（减半到 8.9 ms）→ 49 ms ≈ **20.4 tok/s**。

## prefill 200 的硬约束（必须写清楚）

数值保真的前提下，qk+pv 只能走 f16/f32：

| 段 | 实测 | f16/f32 硬件上限 | 最好情况 |
|---|---|---|---|
| qk (vfwmacc) | 9.2 GMAC/s/核 | 14.4 | 61.5 → 39.4 s |
| pv (vfmacc) | 16 GMAC/s/核 | 28.8 | 35.3 → 19.6 s |
| mask+sm | ~29 s | — | ~10 s |
| GEMM (int4) | 71 GMAC/s/核 | 457 | 46 → ~22 s |
| **合计** | 176 s | | **~93 s ≈ 110 tok/s** |

**所以要 200 tok/s 必须用 int8 attention（QK/PV 走 vmadot），而那会改变 logits，
greedy 输出不再与 llama.cpp 逐 token 相同。** 这是目标内部的一个真实取舍，
需要明确决定，不能默默降精度。

---

# 第九轮：KVT=128 拿到向量宽度，10K prefill 68.56；decode 天花板被 DRAM 钉死

## 1. 核心洞察：向量指令代价与宽度无关

之前两轮的微基准里我记录过一组数字，但没往这个方向想：

| 指令 | vl | 实测 |
|---|---|---|
| `vfwmacc.vf` e16m1→f32m2 | 64 | 4.45 ns/instr |
| `vfmacc.vf` f32m4 | 128 | 4.45 ns/instr |

**同样的 4.45 ns，但一条算 128 lane、一条算 64 lane。** 也就是每条向量指令是固定
8 周期，跟宽度无关。那么把 kv tile 从 64 拉到 128，QK/mask/softmax 就直接多一倍算力，
纯属白送。

`rvv_qk_dot_tile_f16_x4` 里本来就有 `kv_tile > 64` 的 e16m2→f32m4 宽路径，
只是 `KVT` 被 `max_rows_here <= 32` 这个条件挡在 prefill 外面。放开即可。

## 2. 结果

| | 第八轮 | 第九轮 | 对 58.10 原始基线 |
|---|---|---|---|
| pp512 | 176.19 | 174.79 | +0.6% |
| pp2048 | 135.19 | **140.63** | +5.9% |
| pp4096 | 103.87 | **110.69** | +10.5% |
| pp10240 | 60.66 | **68.56** | **+18.0%** |
| tg64 @10K | 15.44 | 15.36 | — |

`ns/(row*kv)` 从 26.68 降到 22.20（-17%）。分段看：
mask 逐元素快 **2.7x**，sm **1.85x**，qk 13%，pv 无变化（PV 本来就走 f32m4 满宽）。

生成文本在 `GGML_FA_KVT64=1`（旧 64 宽）与默认（128 宽）下**逐字一致**，平铺改动数值中性。

## 3. 试过但无效的

- **branchless 无分支 exp**：把 `rvv_expf_approx_f32m2` 里的 `vcpop_m` 分支去掉。
  实测 sm 从 77.3 → 91.6 ms（更差）。该分支几乎总是走快路径，去掉反而只多了寄存器压力。
- **4 行批量 exp**：4 份内联的 exp 各约 12 个 m2 寄存器 = 48 个，超过 32 个向量寄存器，
  全部溢出。sm 77.3 → 103。改成 2 行仍 91.6。**已全部回退到原实现。**
- **m4 宽 exp**：sm 143.5，最差。
- **fp8 (e4m3) KV cache**（kernel 已内置，`-ctk f8_e4m3 -ctv f8_e4m3`）：
  pp10240 68.56 → **52.04（大幅变差）**，tg64 15.44 → 15.30（无变化）。
  pack 阶段把 e4m3 展开成 f16 的开销超过了省下的带宽。**结论：decode 不是 KV 带宽瓶颈。**

## 4. decode @10K 的真实分解（实测，纯 decode `-p 0`，无 prefill 污染）

`IME_PROF_WALL=1`，128 tokens：

| 段 | 每 token |
|---|---|
| GEMM | **50.5 ms** |
| FA | 5.7 ms |
| 其他（glue/norm/rope） | 7.1 ms |
| **合计** | **63.3 ms** → 15.80 tok/s |

关键对照：**纯 decode（无上下文）15.87 tok/s，10K 上下文 15.44。**
attention 在 10K 只增加 1.8 ms。**decode 完全被权重流带宽决定，跟上下文长度几乎无关。**

## 5. 板上 DRAM 带宽实测（`/dev/shm/bw3.c`）

8 线程流式读，2 GB buffer，slab 分区，m4/m8 × 1/4 路展开全部试过：

| 线程 | 带宽 |
|---|---|
| 1 | 8.7 GB/s |
| 4 | 25.7 GB/s |
| 8 | **32.5 GB/s** |

加宽到 m8、4 路展开都**没有提升**。8 线程相对单线程只线性了 4x → 有共享上限，
**32.5 GB/s 就是这块板的 DRAM 读上限。**

decode GEMM 已经在 28.1 GB/s = **峰值的 86%**。

## 6. 两个目标的硬天花板（这是结论）

### decode ≥20 tok/s

权重 1.42 GB（Q4_0，2.52B 参数）：

- 绝对地板：1.42 GB ÷ 32.5 GB/s = **43.7 ms** → 22.9 tok/s（假设 FA 和开销全为 0）
- 现实地板：43.7 + 5.7(FA) = 49.4 ms → **20.2 tok/s**
- 现在：63.3 ms → 15.4 tok/s

所以 20 tok/s **只在一种情况下可达**：GEMM 带宽从 86% 打到 ~100%（省 6.8 ms）
**并且**把那 7.1 ms 的非 GEMM 开销压到 ~1 ms。两个都得做到，正好卡在 20。

只要权重是 1.42 GB，就**没有余量**。

### prefill ≥200 tok/s

qk 现在 11.5 GMAC/s/核，`vfwmacc` 的硬件上限是 14.4 → **已经在 80%**。
pv 17.3 GMAC/s/核 vs `vfmacc` 上限 28.8 → 60%。

f16/f32 最好情况：
- qk 4.53 TMAC ÷ (8×14.4G) = 39.3 s
- pv 4.52 TMAC ÷ (8×28.8G) = 19.6 s
- mask+sm+pack ≈ 15 s
- GEMM 46 s（int4，可优化到 ~25 s）
- glue 2 s
- 合计 ≈ **101 s ≈ 101 tok/s**

**200 tok/s 在 f16/f32 下不可能。必须让 QK/PV 走 int8 `vmadot`**（上限 457 GMAC/s/核），
而这会改变 logits。

## 7. 结论：目标内部有矛盾

- prefill 200 tok/s ⟹ 必须 int8 attention ⟹ logits 变化 ⟹ 无法与 llama.cpp 逐 token 一致
- decode 20 tok/s ⟹ 必须把 1.42 GB 权重读满 DRAM 且零开销，余量为 0；
  真要留余量就得减权重字节（更激进的量化）

两个都撞在"输出与 llama.cpp 对齐"这条上。需要明确取舍。

## 8. 第十轮：查清 in-situ 与微基准的 30% 缺口，并修正 §6 的前提

### 8.1 微基准：行宽已到顶，K 足迹无关

`xqk6`（NROW=48、REPS=8、NT=24，8 线程绑 8-15，先写 `/proc/set_ai_thread`）：

| 变体 | agg GMAC/s |
|---|---|
| x8 ku2（对照） | ~306 |
| x10 ku2 | 309–312 |
| x12 ku2 | ~298 |
| x12 ku1 | ~300 |
| x16 ku2 | ~150 |

x16 掉一半是 f16m2 累加器把向量寄存器组挤爆后的 spill；x10 的 +1% 在噪声内。**行宽这条路到此为止。**

`xqk8` 把每线程 K 足迹从 32 KB 扫到 512 KB（轮转 1/2/4/8/16 块），两轮：

```
32 KB: 292.9  64 KB: 297.1  128 KB: 297.5  256 KB: 296.5  512 KB: 296.9
32 KB: 300.8  64 KB: 298.9  128 KB: 296.9  256 KB: 296.1  512 KB: 296.9
```

**完全平**。所以"in-situ 慢是因为 K/V/KQ/VKQ32 共 ~145 KB 挤爆 L1/L2"这个假设被否定：
这个内核在 512 KB/线程下仍能跑满 ~297 GMAC/s，它不缺带宽也不缺容量。

### 8.2 微基准自身踩的坑（记录，避免重犯）

`xqk7` 一上来就 `*** stack smashing detected ***`（关掉栈保护后变 segfault）。隔离实验
（4 MB 对齐 static 数组 + 8 线程）正常，排除 BSS/线程栈大小。真正原因是 **GCC 把 f16m2
向量内核内联进了线程 lambda**，生成的栈帧被破坏。修法：内核加
`__attribute__((noinline))`，并强制经函数指针 `(*(fn_t)qk_x8_ku2)(...)` 调用。之后全部正常。
板上没有 gdb，这类问题只能靠最小差分定位。

### 8.3 in-situ 的真实速率（thread-0 窗口口径）

`FA_PROF` 每累计 6720 个 tile 打印一次并把计数器清零，计时器只在 `ith==0` 打开，所以
窗口数字是 **thread-0** 的。thread-0 每窗口 6720 tile、全程约 77 个窗口（520k tile/线程），
77 × 0.77 s ≈ 58.5 s，与 `IME_PROF` 的 fa 58.535 s 对得上——这个口径可信。

一个 tile = 64 行 × 128 kv × 128 d = 1.048 MMAC：

- 纯 QK 点积（已扣掉 pack）：241 ms / 6720 = **35.9 µs/tile → 29.2 GMAC/s/线程**
- PV：281 ms / 6720 = **41.8 µs/tile → 25.1 GMAC/s/线程**
- 微基准同内核：37 GMAC/s/线程 → QK 差 **1.27×**、PV 差 **1.47×**
- 每 tile 固定开销：pack（kx）7.4 µs + mask 全掩码扫描 2.6 µs ≈ 10 µs，占 41 µs 的 24%

### 8.4 修掉 kpack inplace 的死分支

`GGML_KPACK_INPLACE` 的设计是让 QK/PV 直接读共享的 packed K/V（布局本来就是
`K[DK][KVT]` 接 `V[KVT][DV]`，正是内核要的），省掉每次 (Q tile, KV tile) 访问的
2×32 KB memcpy。但原代码在 3254 行设好 `k_src=pdst; v_src=pdst+DK*KVT` 之后，
3263–3264 行**无条件**又覆盖回 `K_f16/V_f16`：分支成了死代码，而且一旦真跳过拷贝就会读到
陈旧 K/V。也就是说这条优化**从未生效**，每次访问都在白拷 64 KB/线程。

修复（把 reset 挪进 else，只有走 scratch 时才回退）：

```cpp
if (kpack_inplace && kv_tile == KVT) {
    k_src = pdst;  v_src = pdst + DK * KVT;
} else {
    k_src = K_f16; v_src = V_f16;
}
```

10K A/B（其余 env 与 111.77 s 基线完全相同）：

| 版本 | 10K prefill | t/s | TOKENS |
|---|---|---|---|
| 基线（inplace 关） | 111.77 s | 89.46 | 220 5298 242 8647 |
| 修复 + `GGML_KPACK_INPLACE=1` | **110.97 s** | 90.11 | 220 5298 242 8647 |

只值 0.7%——与剖面一致（kx 仅 4.6–6.4%），拷贝不是那 30% 缺口。但它是白工 + 一个潜在
读陈旧数据的 bug，净收益为正且逐位一致，保留。

### 8.5 对 §6 "硬天花板"的修正（重要）

§6 的 `qk 14.4 GMAC/s/核` 是**未开 x8 f16 内核**那版（x4 加宽、甚至 f32 路径）测出来的。
开启 `GGML_FA_QK_F16=1 GGML_FA_PV_F16=1` 后 qk 段从 725 ms 降到 263 ms（2.76×），
实测内核 37 GMAC/s/核、in-situ 29。因此 §6 里"f16 下 200 tok/s 不可能"的算术前提不成立，
必须按新数重算（同样是 f16、同样逐位一致）：

| 项 | 当前 | 恢复 in-situ 缺口后 |
|---|---|---|
| QK 4.36 TMAC | 18.7 s @29.2 | 14.7 s @37 |
| PV 4.36 TMAC | 21.7 s @25.1 | 14.7 s @37 |
| sm+mask+pack | ~14 s | ~8 s（Q_TILE 翻倍摊薄） |
| GEMM 19.8 TMAC | 46.5 s @426 | 需 ≥800 GMAC/s → 25 s |
| glue | ~7 s | ~5 s |
| 合计 | 111.8 s | ~67 s ≈ 150 t/s |

结论改成可操作的形式：**200 t/s 需要 FA 侧把 QK/PV 的 in-situ 速率拉回微基准并摊薄固定开销
（1.5–1.7×），同时把 IME GEMM 从 426 GMAC/s 提到 800+ GMAC/s（1.9×）**。两项都是
"同精度、逐位一致"前提下的工程问题，不是精度取舍问题。

### 8.6 Q_TILE 64→128：否掉

§8.5 表里"sm+mask+pack 摊薄"这一格的本意是把 Q tile 从 64 翻到 128，让每 tile 的 KV 装载
服务两倍的行。10K A/B（只改 `GGML_FA_TILE_Q`，代码路径自洽：`work_size()` 与 1474 行用的是
同一个常量）：

| 版本 | 10K prefill | t/s | TOKENS |
|---|---|---|---|
| Q_TILE 64（基线） | 110.97 s | 90.11 | 220 5298 242 8647 |
| Q_TILE 128 | **126.31 s** | 79.17 | 220 5298 242 8647 |

逐位一致，但慢 13.8%。原因两条：每线程 FA scratch 从 145 KB 涨到 320 KB（8 线程 2.5 MB，
L2 里放不下别的了），且对角 tile 的掩码浪费翻倍（128 行里靠后的行有一半被因果掩码掉）。
已回滚到 64 并重建。

### 8.7 那 30% 缺口：外部解释逐个排除

在改代码之前先把"是不是环境/基准失真"排干净，三个实验都是同一块板、8 线程钉 8–15、
`set_ai_thread` 写 0：

| 实验 | 假设 | 结果 |
|---|---|---|
| xqk8（K 足迹 32→512 KB/线程） | 缓存容量/带宽 | 292.9/297.1/297.5/296.5/296.9 GMAC/s，**平的** |
| xqk9（同内核持续 18 s，6 窗口） | DVFS / 热降频 | 304.7–305.1 GMAC/s，cpu8/12 前后都是 1.8 GHz |
| xqk10（精确结构模拟：Q 块外/KV 内、因果、共享 K、线程按 Q 块轮转） | 冷 K / 访问模式 | 共享 arena 2.5 MB（10K 规模）：300.4/301.0/299.9/300.0；10 MB（4 倍于 10K）：298.5–299.0 GMAC/s |
| 现场采样（跑 10K 时每 2 s 读 scaling_cur_freq） | megakernel 里 IME+向量交替导致降频 | cpu8/12/15 **全程 1.800 GHz** |

也就是：微基准 37 GMAC/s/线程在任何足迹、任何时长、任何共享方式下都成立，in-situ 的
29.2 不是环境。只剩两个候选：(a) 调用组合不是"全满 tile x8"，(b) 内核被内联进 24 KB 的
巨型函数后生成的代码更差。

先查 (b)：反汇编部署版 `forward_flash_attn_ext_f16_tiled_vlen1024_vf16`，x8 内层循环
（`.L2666`）是干净的 `1×vle16.v + 8×flh + 8×vfmacc.vf + 10 条标量 + 分支`，8 个累加器
v4/v6/…/v18 全程在寄存器里，没有 `vl*r/vs*r` 整寄存器 spill。所以不是溢出。

### 8.8 下一步：在部署路径里按线程埋点

既然外部原因和代码生成都排除，必须在 in-situ 直接测"纯 x8 内核时间 / 调用数 / 实际
kv_tile 与 tile_rows / 各线程 vlenb"。在 `FA_PROF` 里加一组每线程计数器：

- `fa_x8_ns[ith]`：8 次 x8 调用块前后的 `rdtime` 差（仅该分支，不含 pack/mask/softmax）
- `fa_x8_nt/kv/rows[ith]`：调用次数、`kv_tile` 累计、`tile_rows` 累计
- `fa_x8_other[ith]`：落到非 x8 分支（`kv_tile<=64` 或未开 f16）的次数
- `fa_vlenb`：本线程 `csrr vlenb`

窗口打印时输出 8 个线程一行。判据很明确：若纯 x8 时间就是 35.9 µs/tile，则内核在
in-situ 真的慢（继续查 K/V 排序与 PV 交替的缓存交互）；若纯 x8 只有 28–30 µs，则
`qk` 段里混进了别的活，按段补计时。


## 9. QK 尾声重做、TCM 累加器串组、PV 循环序（2026-11-23）

### 9.1 先把"QK 慢"定位到尾声，而不是点积

`probe_qk_epi.cpp` 在板子上单独跑 8 组 `vmadot` 步（8 累加器 x 32 lane），只换尾声写法：

| 变体 | GMAC/s | 时间 |
|---|---|---|
| A 原始（f32 → f32 缩放 → i16 + 3 次 slidedown + 每行 4 次 8-lane store） | 48.3 | 1.74 ms |
| B `vnsra` + slidedown | 35.1 | 2.39 ms |
| C 交错 4 行 x 8 位置块，单次 store | **93.8** | 0.89 ms |
| D 无尾声（裸 store） | 101.4 | 0.83 ms |
| E 交错 + 缩放 | 79.1 | 1.06 ms |
| F `vrgather` 转成按行 | 71.9 | 1.17 ms |

结论：点子积没问题（裸 store 101.4 就是单核天花板），是**尾声的转置**吃掉了 50%+。
于是 `scores` 改成 `vmadot` 直出的交错布局（lane = 4 行 x 8 位置，块 64 B），
尾声只剩一条 `vnsra.wx` + 一条 `vse16`：

- `SC_NBLK = MAXCTX/8`，`SC_GSTRIDE = SC_NBLK*32`，`SC_TSTRIDE = I8_G*SC_GSTRIDE`，`SC_SHIFT = 6`
- 软最大化改成按块 `vmax` 后 3 次 slidedown 折叠（只读 lane 0/8/16/24），exp 用 `exp2f_poly_m4`
  一次算 4 行 x 8 位置
- 清零只需从该组的因果边界 `(L+4g+1)/16` 开始，不再 memset 整块

10K 因果 bench（`-A 10240 -x 0 -J -I -K`）：QK 128.7 → **101.8 core-s**。

### 9.2 `-Q`（TCM staging）一直是错的

`pfa_pv_i8` 的 TCM 分支把 16 个 row group 的 `vmadot` 累加器指到同一组 `praw`，
于是每个输出都是 16 个组之和。bench 校验 `worst at dim 0: mine 3900.72`（参考 0.997），
而**文件路径强制 `g_tcmstage = 1`**，所以每次真 prefill 的 attention 都是错的。
改成每组一套 `pacc[I8_G][8][32]`（16 KB，线程私有）后 `worst at dim 84: mine 0.81087`，
与非 TCM 路径逐位一致。

代价：PV 从 44 → 58 core-s（累加器 1 KB → 16 KB 的 L1 往返）。这笔账是必须付的。

### 9.3 PV 的循环顺序：按 K/V 块外、行组内

原来 `for gg { pv4(整段 V) }`，即每个 tile 把 V 的范围走 16 遍。10K 上 V 是 1.28 MB/tile，
16 遍 = 20 MB/tile，直接吃满 DRAM：PV 131 core-s。
改成"V 块外、行组内"（16 组累加器落在 L1），并用每组自己的因果上界截断点积后：
**PV 53.9 core-s**（`-y` 保留旧序做对照）。
TCM staging（`-Q`）再加一次纯 memcpy 的流式搬运，从 130.9 降到 56.8；去掉 staging
只改循环序是 130.9（说明问题在"点积里夹 1 KB 随机读"而不是读的总量）。

### 9.4 两个非 attention 的坑

- `pfa_pack_q` 是**单线程**，2.43 s/次 prefill。按 (row group, head) 用 `parallel_for`
  切开后 0.34 s。10K prefill 直接少 2.1 s。
- `fscanf(f, "%d", &v)` 在这块板子 + 这版 glibc 上**返回 1 但不写回**：解析 10000 个
  token 全部拿到栈上残值（每次运行不同，例如 `1383359744`），第一次 embedding 查表就
  SIGSEGV。改用自写的 strtol 式扫描后正常。任何以后新增的 prompt 解析都不要用 fscanf。

### 9.5 现在的账（10K，`-F p10k.txt -G 10240`）

| 项目 | 时间 |
|---|---|
| gemm | 25.56 s（qkv 2.99 / o 3.25 / gate+up 10.55 / down 8.76） |
| attn | 22.08 s（qk 100.8 + pv 53.9 core-s；X100 softmax 只受等值约束） |
| conv(K/V→int8) | 2.35 s |
| misc | 3.08 s |
| quant / silu / norm | 1.66 / 1.08 / 0.93 s |
| packq | 0.34 s |
| **合计** | **57.26 s → 174.6 tok/s**（改动前 165.5） |

剩下的差距（到 200 tok/s 还差 7.3 s）：
- 约 362 GB 的 attention DRAM 流量 / 实测单核 memcpy 只有 6.8 GB/s（读写合计 13 GB/s），
  聚合到 16–17 GB/s 就顶住了。分数缓冲占其中 57%（写 68 GB + 读两遍 137 GB）。
  要再快只能少搬字节：把 QK+max 融合（每个组块多 14 条向量指令，QK 会涨 70%，不划算）、
  或在线最大值 + int32 累加器重缩放（要重写 PV）。
- QK 还可以按"2 个 query tile 共享一次 K 读"把 K 流量砍半（16 个累加器 v16–v31，
  2 个 Q 寄存器），预期再省 2 s 量级。
- gemm 25.56 s 已经是 846 GMAC/s 聚合（106 GMAC/s/核），A 簇基本到顶；X100 在这段时间
  空转，把 FFN 的一小部分行分给 X100（实测 625–652 GMAC/s）是唯一还没试的大块。

## 10. prefill→decode 串成一条链，以及 10K decode 的天花板（2026-11-23 下午）

### 10.1 链条

`-F p10k.txt` 之前 prefill 完、导出 KV 就 `return 0`，decode 根本没跑。现在：

- 加 `pos0`：prefill 处理了 `0..nt-2`，decode 从最后一个 prompt token 在位置 `nt-1` 开始，
  直接复用 prefill 写进同一份 f16 KV 的缓存（`g_vpos` / `ktile`），不重新加载。
- `prefill_release()` 释放 prefill 专用的大缓冲（10K chunk 下几百 MB），
  避免把 decode 的attention 工作集挤到 DRAM 之外。
- 三个 bench 分支加 `!pos0` 守卫，否则 `real_prefill` 会把 `g_pfa_rows` 置上、
  串完链又跑一遍 241 s 的合成 attention bench。

实测（`-F p10k.txt -G 10240 -n 8`，单进程）：

| 阶段 | 结果 |
|---|---|
| prefill | 9999 tokens / 57.0 s = **175.4 tok/s** |
| decode（10K，贪心） | 8 tokens / 0.794 s = **10.1 tok/s** |

### 10.2 板子的内存天花板（实测，决定了 decode 的上限）

| 线程数 | 聚合读带宽 |
|---|---|
| 1 | 6.8 GB/s |
| 4 | 12.5 GB/s |
| 8 | 17.5 GB/s |
| 12 | 19.0 GB/s |
| 16 | 19.3 GB/s |

即 12 核之后就饱和。10K decode 每 token 必须搬：权重 1265 MB（Q4_0）+ KV 440 MB（f16）
= 1705 MB → 88 ms/token 的下限，也就是 **11.6 tok/s**。实测 10.1 tok/s，已经贴着这条线。

也就是说：q4 权重 + f16 KV 时，**10K decode 的 20 tok/s 在物理上不可能**。要到 50 ms/token，
总流量必须 ≤ 965 MB。可行的组合只有：

- KV 改 int8：440 → 220 MB
- 权重改 2 bit（厂商已有 `gemm_kernel_i8i2k`）：1265 → ~720 MB

两者合计 ~940 MB → ~49 ms/token → 20 tok/s 量级。只做其中一项都不够
（int8 KV 单独：1445 MB → 15.8 tok/s；2 bit 单独：1160 MB → 12.6 tok/s）。

### 10.3 decode attention 的实测分解（10K，ms/token）

`qk 15.1  exp 2.0  pv 22.9  fin 0.6`，合计 40.6 ms，占每 token 时间的 41%。
qk 读 K 220 MB 用 15.1 ms（14.6 GB/s），pv 读 V 220 MB 用 22.9 ms（9.6 GB/s）——
都没到 19.3 GB/s，说明这两段还有 1.3–2 倍的向量化空间，改完可以把 attention 压到 ~20 ms。

### 10.4 `-q3`（3 bit 权重）能跑，但不能用于对齐

`-q3` 把 Q4_0 反量化后按每 k16 重新量化成 3 bit（`repack_b_q3`，`gemm_kernel_i8i3k`）：

- 权重 1265.2 → **1001.6 MB**（0.79 倍，因为 Q3_K 要存每 k16 的 int8 scale + fp16 + mask）
- 短上下文 decode 16.45 → **20.37 tok/s**
- 但贪心输出在**第 1 个 token** 就和 Q4_0 分叉（`5 2104 394…` vs `5 367 22179…`），
  因为这是自定的 per-k16 缩放，不是 llama.cpp 的 Q3_K。

所以 `-q3` 只能当速度探针。要既压字节又保持"与 llama.cpp 对齐"，必须用 llama.cpp
自己的量化器产出 Q3_K/Q2_K，再喂给厂商的 i8i3k/i8i2k 内核。

### 10.5 一个和数值无关、但会打断脚本的退出崩溃

`-O2` 构建在 main 返回时报 `stack smashing detected`。已确认：

- `-n 0`（不生成任何 token）也会触发，所以发生在加载/repack 阶段，与 decode 无关；
- `-O1` 或 `-fno-stack-protector` 不触发，ASan 全程无报告；
- 早于本次改动（`76a8862`）就存在，`-t 1`、加大栈都照样触发；
- 数值不受影响（prefill 的 `worst at dim 84` 与非崩溃构建逐位一致）。

判断是 GCC 15 在 RVV 动态 `vlenb` 帧 + 32 个向量寄存器 spill 上的代码生成问题，
不是源码越界。当前测量构建统一加 `-fno-stack-protector` 绕开，另用 `-O1` 交叉验证数值。

## 11. exp 三指令近似与 flash attention：都试了，一个有保留、一个被数据否掉

### 11.1 exp：三指令成立，但只在"融合进点积循环"时才有价值

`probe_exp3.cpp`（X100 单核，VLEN=256，每次 8 lane）：

| 版本 | 指令数 | 相对误差峰值 | 周期/向量 |
|---|---|---|---|
| 六项多项式（原） | 12 | 1.8e-7 | 43.4 |
| 尾数技巧（mul + cvt + add） | 3 | **6.15e-2** | **13.9** |
| 尾数技巧 + 8 桶 vrgather 修正 | 7 | 1.7e-2 | 45.8 |
| 尾数技巧 + 32 桶（A100） | 7 | 4e-3 | 实测 864 core-s，**vrgather 在这颗核上极慢** |

尾数技巧：`vfcvt.x.f(t·2^23)` 把 `floor(t)` 落进指数域、小数落进尾数域，
再 `+127<<23` 重新解释就是 `2^floor(t)·(1+f)`。误差恒为正、峰值在 f=0.5288（6.15%）。
两桶修正全部失败：8 桶残留 1.7%（g(f)=2^f/(1+f) 本身有 6% 起伏），
32 桶在 A100 上要靠 vrgather，实测把 softmax 拖到 864 core-s。

**实测影响**（10K，`-A 10240 -x 0 -J -I -Q -V`，softmax 在 A100 上）：

| | qk | sm | pv | 总 |
|---|---|---|---|---|
| poly | 78.5 | 260.5 | 123.6 | 60.6 s |
| 三指令 | 76.7 | **143.5** | 91.7 | **41.9 s** |

单看这一段是 1.8 倍。但**生产路径不受影响**：10K 全链 prefill
`-e 0` 57.88 s / `-e 1` 57.81 s，decode exp 16.8 → 14.2 ms（每 token 少 0.3 ms）。
原因是 softmax 早就被搬到空闲的 X100 上，它的耗时藏在 A100 的点积后面；
而 decode 的 exp 只占每 token 的 2%。

结论：`-e 1` 保留为开关（默认仍用多项式），它的价值在于**允许把 exp 融进点积循环**
——也就是下一条要讨论的 FA——而不是单独替换。

### 11.2 FA：单遍融合在这个板子上是负收益，已用数字排除

把 QK、softmax、PV 融成一个循环（不留 score 缓冲）意味着 softmax 回到 A100 上。
按 11.1 的表，那就是把 143 core-s（三指令）或 260 core-s（多项式）重新压到
A100 的关键路径上：

| 方案 | A100 core-s | 8 核墙钟 | X100 |
|---|---|---|---|
| 现在：QK+PV 在 A100，softmax 在 X100 | 149 | **21.7 s** | 15 s（藏住） |
| FA 融合，全在 A100（三指令 exp） | 312 | 41.9 s | 空闲 |
| FA 融合，全在 A100（多项式 exp） | 462 | 60.6 s | 空闲 |

FA 省下的只有 score 缓冲：`-D` 探针量到 QK 的 store 值 20 core-s。用 20 core-s 去换
143 core-s 的 exp 回迁，账面上不可能赢。所以**没有采用教科书式 FA**。

真正该做的是"FA 的局部版"：保留 A100 做点积、X100 做 exp 的分工，但把 score 缓冲
从"整段 context"改成"按 KV 块环形缓冲"。现在的 score 缓冲是每 head-tile 1.3 MB，
10K 全层合计约 137 GB 的写 + 137 GB 的读，按 19 GB/s 折算是 14 s 的纯搬运，
刚好是 attention 墙钟（21.7 s）与 A100 计算下界（18.6 s）之间的差。
块环形缓冲（8 块 × 64 位 × 64 行 × 2 B = 64 KB）能把它压进 L2，预计省 3–4 s。
