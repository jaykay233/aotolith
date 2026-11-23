#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Build the split-KV tiled FA path and validate it, then measure the 10K decode gain.
#   build   : only libggml-cpu changes; the mega binaries load it via RUNPATH
#   V4K_REF : single-thread reference token stream (no split possible, nth == 1)
#   V4K_*   : 4K prompt token match, split off vs forced on (FA_SPLITKV_MIN=0)
#   S4K_*   : same for the speculative verify path (4-token verify -> nr=64)
#   V10K_*  : 10K timing; prefill rows (nr=256) take the normal tiled path, decode splits
cd /home/bianbu/workspace/llama-cpp-31953bc || exit 1
LOG=$WORK/splitkv_tiled.log
: > "$LOG"
echo "=== BUILD" >> "$LOG"
cmake --build build-smt --target ggml-cpu -j 16 >> "$LOG" 2>&1
rc=$?
echo "EXIT_BUILD=$rc" >> "$LOG"
if [ $rc -ne 0 ]; then echo "=== DONE" >> "$LOG"; exit 1; fi

cd "$WORK" || exit 1
set -o pipefail
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P4=/home/bianbu/opt10k/ctx4k.txt
P10=/home/bianbu/opt10k/ctx10k.txt

run() { # label binary prompt n threads extra_env...
  local label=$1 bin=$2 p=$3 n=$4 t=$5; shift 5
  echo "=== $label bin=$bin n=$n t=$t env=$*" >> "$LOG"
  timeout 2400 env FA_TRACE=1 "$@" "./$bin" -m "$M" -f "$p" -n "$n" -c 12288 -t "$t" 2>&1 \
    | grep -E '^\[mega\]|^TOKENS|^FA_PROF|^FA_TRACE' >> "$LOG"
  echo "EXIT_$label=$?" >> "$LOG"
}

run V4K_REF      mega_minicpm_pf   "$P4"  24 1 FA_SPLITKV_TILED=0
run V4K_OFF      mega_minicpm_pf   "$P4"  24 8 FA_SPLITKV_TILED=0
run V4K_ON       mega_minicpm_pf   "$P4"  24 8 FA_SPLITKV_MIN=0
run S4K_OFF      mega_minicpm_spec "$P4"  24 8 MEGA_DRAFT=3 FA_SPLITKV_TILED=0
run S4K_ON       mega_minicpm_spec "$P4"  24 8 MEGA_DRAFT=3 FA_SPLITKV_MIN=0
run V10K_OFF     mega_minicpm_pf   "$P10" 32 8 FA_SPLITKV_TILED=0
run V10K_ON      mega_minicpm_pf   "$P10" 32 8
echo "=== DONE" >> "$LOG"
