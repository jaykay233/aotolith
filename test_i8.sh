#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Does the int8 QK path (GGML_FA_ATTN_I8=1) speed up 10K prefill, and do tokens stay identical?
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/i8_10k.log
: > "$LOG"

echo "=== MEGA t=8 n=32 i8=1 GGML_FA_ATTN_I8=1" >> "$LOG"
timeout 2400 env GGML_FA_ATTN_I8=1 ./mega_minicpm_pf -m "$M" -f "$P" -n 32 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_I8_T8=$?" >> "$LOG"

echo "=== MEGA t=4 n=32 i8=1" >> "$LOG"
timeout 2400 env GGML_FA_ATTN_I8=1 ./mega_minicpm_pf -m "$M" -f "$P" -n 32 -c 12288 -t 4 >> "$LOG" 2>&1
echo "EXIT_I8_T4=$?" >> "$LOG"

echo "=== MEGA t=8 n=4 short ctx 512 sanity (i8 on)" >> "$LOG"
head -c 1800 /home/bianbu/opt10k/ctx10k.txt > /dev/shm/short1.txt
timeout 600 env GGML_FA_ATTN_I8=1 ./mega_minicpm_pf -m "$M" -f /dev/shm/short1.txt -n 8 -c 2048 -t 8 >> "$LOG" 2>&1
echo "EXIT_I8_SHORT=$?" >> "$LOG"
echo "=== MEGA t=8 n=4 short ctx 512 sanity (i8 off)" >> "$LOG"
timeout 600 ./mega_minicpm_pf -m "$M" -f /dev/shm/short1.txt -n 8 -c 2048 -t 8 >> "$LOG" 2>&1
echo "EXIT_OFF_SHORT=$?" >> "$LOG"
echo "=== DONE" >> "$LOG"
