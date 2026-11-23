#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
LOG=$WORK/pf_chunk_test.log
: > "$LOG"
for P in /dev/shm/p600.txt /dev/shm/p1200.txt; do
  for R in ref mega; do
    echo "=== $P $R" >> "$LOG"
    if [ "$R" = ref ]; then F=--ref; else F=; fi
    timeout 600 ./mega_minicpm_pf -m "$M" -f "$P" -n 8 -c 4096 -t 8 $F >> "$LOG" 2>&1
    echo "EXIT=$?" >> "$LOG"
  done
done
echo "=== DONE" >> "$LOG"
