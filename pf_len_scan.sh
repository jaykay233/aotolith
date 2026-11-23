#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Is the megakernel error a hard cliff at ~256 keys, or a gradual drift with length?
# Compares --ref (llama.cpp) vs megakernel at increasing prompt lengths, n=4.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
SRC=/dev/shm/p1200.txt
LOG=$WORK/pf_len_scan.log
: > "$LOG"

for BYTES in 700 900 1050 1150 1200; do
  head -c "$BYTES" "$SRC" > "/dev/shm/plen_$BYTES.txt"
  echo "### bytes=$BYTES" >> "$LOG"
  timeout 300 ./mega_minicpm_pf --ref -m "$M" -f "/dev/shm/plen_$BYTES.txt" -n 4 -c 2048 -t 8 2>&1 \
      | grep -E "prompt tokens|^TOKENS" | sed 's/^/REF   /' >> "$LOG"
  timeout 300 env MEGA_PF_BATCH=1024 ./mega_minicpm_pf -m "$M" -f "/dev/shm/plen_$BYTES.txt" -n 4 -c 2048 -t 8 2>&1 \
      | grep -E "prompt tokens|^TOKENS" | sed 's/^/MEGA  /' >> "$LOG"
done
echo "=== DONE" >> "$LOG"
