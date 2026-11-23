#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# 10K speculative decode with prompt-lookup drafts (k=3 vs k=7), t=8, n=64.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/spec10k.log
: > "$LOG"

run() { # label k threads n
  local label=$1 k=$2 t=$3 n=$4
  echo "=== $label k=$k t=$t n=$n" >> "$LOG"
  timeout 2400 env MEGA_DRAFT=$k ./mega_minicpm_spec -m "$M" -f "$P" -n "$n" -c 12288 -t "$t" >> "$LOG" 2>&1
  echo "EXIT_$label=$?" >> "$LOG"
}

run SPEC_K3_T8 3 8 64
run SPEC_K7_T8 7 8 64
run SPEC_K3_T4 3 4 64
echo "=== DONE" >> "$LOG"
