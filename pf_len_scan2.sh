#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Where does megakernel batched prefill start diverging from llama.cpp? Single chunk forced.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
SRC=/dev/shm/p1200.txt
LOG=$WORK/pf_len_scan2.log
: > "$LOG"

for BYTES in 40 130 260 400 520 600 640 700; do
  head -c "$BYTES" "$SRC" > "/dev/shm/pl2_$BYTES.txt"
  echo "### bytes=$BYTES" >> "$LOG"
  timeout 300 ./mega_minicpm_pf --ref -m "$M" -f "/dev/shm/pl2_$BYTES.txt" -n 4 -c 2048 -t 8 2>&1 \
      | grep -E "prompt tokens|^TOKENS" | sed 's/^/REF   /' >> "$LOG"
  timeout 300 env MEGA_PF_BATCH=1024 ./mega_minicpm_pf -m "$M" -f "/dev/shm/pl2_$BYTES.txt" -n 4 -c 2048 -t 8 2>&1 \
      | grep -E "prompt tokens|pf_batch|^TOKENS" | sed 's/^/MEGA  /' >> "$LOG"
  timeout 300 ./mega_minicpm_pf -m "$M" -f "/dev/shm/pl2_$BYTES.txt" -n 4 -c 2048 -t 8 2>&1 \
      | grep -E "pf_batch|^TOKENS" | sed 's/^/DEF   /' >> "$LOG"
done
echo "=== DONE" >> "$LOG"
