#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Rebuild ggml-cpu with the decode split-KV fall-through, then re-measure 10K.
cd /home/bianbu/workspace/llama-cpp-31953bc || exit 1
LOG=$WORK/splitkv_10k.log
: > "$LOG"
echo "=== BUILD" >> "$LOG"
cmake --build build-smt --target ggml-cpu -j 16 >> "$LOG" 2>&1
echo "EXIT_BUILD=$?" >> "$LOG"

cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt

echo "=== MEGA splitkv t=8 n=64" >> "$LOG"
timeout 2400 env FA_TRACE=1 ./mega_minicpm_pf -m "$M" -f "$P" -n 64 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_SKV_T8=$?" >> "$LOG"

echo "=== MEGA splitkv t=4 n=64" >> "$LOG"
timeout 2400 ./mega_minicpm_pf -m "$M" -f "$P" -n 64 -c 12288 -t 4 >> "$LOG" 2>&1
echo "EXIT_SKV_T4=$?" >> "$LOG"

echo "=== MEGA splitkv t=8 n=16 i8qk (prefill attn int8)" >> "$LOG"
timeout 2400 env GGML_FA_ATTN_I8=1 ./mega_minicpm_pf -m "$M" -f "$P" -n 16 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_SKV_T8_I8=$?" >> "$LOG"

echo "=== REF t=8 n=4 token check" >> "$LOG"
timeout 2400 ./mega_minicpm_pf --ref -m "$M" -f "$P" -n 4 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_REF=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
