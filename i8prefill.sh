#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# 10K prefill with the int8 QK path (GGML_FA_ATTN_I8=1): speed + token identity.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/i8_prefill.log
: > "$LOG"

echo "=== I8 t=8 n=16" >> "$LOG"
timeout 2400 env GGML_FA_ATTN_I8=1 ./mega_minicpm_pf -m "$M" -f "$P" -n 16 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_I8=$?" >> "$LOG"

echo "=== I8 prof t=8 n=4" >> "$LOG"
timeout 2400 env GGML_FA_ATTN_I8=1 FA_PROF=1 IME_PROF=1 ./mega_minicpm_pf -m "$M" -f "$P" -n 4 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_I8PROF=$?" >> "$LOG"

echo "=== I8 t=4 n=16" >> "$LOG"
timeout 2400 env GGML_FA_ATTN_I8=1 ./mega_minicpm_pf -m "$M" -f "$P" -n 16 -c 12288 -t 4 >> "$LOG" 2>&1
echo "EXIT_I8_T4=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
