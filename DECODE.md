# Decode: measured state

Numbers below are from `./bench_decode.sh` on the board (16.4 GB RAM, `/dev/shm`
7.9 GB), Q4_0 weights, greedy, `-O1`.

## What the target needs

    decode >= 20 tok/s  ==  50.0 ms/token at any context up to 10K

## Where it is

| item | i8xi4 (Q4_0) | i8xi3k (Q3) |
|---|---|---|
| weight blob | 1265.2 MB | 1001.6 MB |
| GEMM per token | 50.15 ms | 38.90 ms |
| attention (ctx <= 64) | 2.56 ms | 2.56 ms |
| everything else | 4.7-5.2 ms | 4.6-5.1 ms |
| **tok/s** | **17.26** | **20.52** |

GEMM sustains 24.0 GB/s against a measured raw-stream ceiling of 27.5 GB/s, so the
vendor kernel's serial load->use chain is leaving ~13% of the bandwidth on the
table. Everything else in the token costs ~7.3 ms and is flat across the stages.

## Two things this measurement does NOT cover

1. **No real prompt.** `-p` is a single starting token id and the loop sets
   `g_pos = step`, so a run labelled `prompt=10240` has 32 tokens of context, not
   10240. The attention figures above are for short context.
2. **Attention does not scale.** It costs 2.56 ms at ctx 32 and 4.92 ms at ctx 64,
   i.e. 0.074 ms per position. Extrapolated, 10240 positions would be ~755 ms per
   token. The scalar per-position loop has to be replaced with the vectorised
   kernel before a 10K decode number means anything.

## A bug this measurement found

`llama-quantize` leaves `output.weight` at Q6_K by default. `mega_decode` repacked
it as Q4_0 like the other 294 matrices, so 210-byte Q6_K blocks were parsed as
144-byte Q4_0 blocks: the fp16 scales became random bit patterns and the int32
accumulators overflowed. Every logit was `inf` and argmax picked garbage. Both
mega3 and mega_decode had this. Quantize with

    llama-quantize --pure --output-tensor-type q4_0 --token-embedding-type q4_0 \
        MiniCPM5-2B-F16.gguf mc-q4o.gguf Q4_0 8

which puts all 296 tensors at Q4_0, and delete the repack caches afterwards: the
cache is validated by byte size only, and the Q6_K-as-Q4_0 blob has the same size,
so a stale cache loads silently and reproduces the bug.

## Why 3-bit is not the answer on its own

It clears 20 tok/s (20.52), but it changes the output: greedy tokens diverge from
the Q4_0 run at step 1 with a margin of 0.67. The objective requires the output to
match llama.cpp, so 3-bit cannot be the final answer unless the whole comparison
is redefined against a 3-bit reference.

## Storage

The model, the repack caches and the binaries were in `/dev/shm`, i.e. RAM: about
3.6 GB of it, wiped on every recovery, and a plausible contributor to the board
dropping off the network. They now live on UFS (`/home/bianbu/workspace/mega`) and
survive reboots. The weights are still *streamed from RAM* at runtime, because
decode needs 24+ GB/s and UFS is an order of magnitude short; they are read from
UFS once at startup into an aligned huge-page blob.
