#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Decode attention partition sweep at 10K: FA_NCHUNK is one thread per whole kv-head group,
# which cuts KV re-reads (each thread scans its kv head once) at the cost of fewer active threads.
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/fanchunk_10k.log
: > "$LOG"

run() { # label threads n extra_env...
  local label=$1 t=$2 n=$3; shift 3
  echo "=== $label t=$t n=$n env=$*" >> "$LOG"
  timeout 2400 env "$@" ./mega_minicpm_pf -m "$M" -f "$P" -n "$n" -c 12288 -t "$t" >> "$LOG" 2>&1
  echo "EXIT_$label=$?" >> "$LOG"
}

run SKV_NC2_T8  8 32 FA_NCHUNK=2
run SKV_NC4_T8  8 32 FA_NCHUNK=4
run SKV_NC2_T4  4 32 FA_NCHUNK=2
run I8_T8       8 16 GGML_FA_ATTN_I8=1
echo "=== DONE" >> "$LOG"
