#!/bin/bash
. "$(dirname "$0")/board_env.sh"   # WORK = UFS work dir; tmpfs is RAM
# Where does 10K time go?
# 1) profiled megakernel pass at 10K (IME gemm/fa section timers)
# 2) llama-bench tg at depth 0 and 10000, FA off vs on -> is the 18 t/s record the non-FA path?
cd "$WORK" || exit 1
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export GGML_TILED_KPACK=1
M=$WORK/mc-q4o.gguf
P=/home/bianbu/opt10k/ctx10k.txt
LOG=$WORK/prof10k.log
B=/home/bianbu/workspace/llama-cpp-31953bc/build-smt/bin
: > "$LOG"

echo "=== MEGA prof t=8 n=4 (prefill 10000 once, then 4 decode steps)" >> "$LOG"
env IME_PROF=1 IME_PROF_WALL=1 IME_OP_PROF=1 FA_PROF=1 \
  timeout 2400 ./mega_minicpm_pf -m "$M" -f "$P" -n 4 -c 12288 -t 8 >> "$LOG" 2>&1
echo "EXIT_PROF=$?" >> "$LOG"

bench() { # d fa t
  echo "=== llama-bench d=$1 fa=$2 t=$3" >> "$LOG"
  ( cd "$B" && LD_LIBRARY_PATH="$B" timeout 1800 ./llama-bench -m "$M" -fa "$2" -p 0 -n 16 -d "$1" -t "$3" -r 1 >> "$LOG" 2>&1 )
  echo "EXIT_D$1_FA$2_T$3=$?" >> "$LOG"
}

bench 0     0 8
bench 0     1 8
bench 10000 0 8
bench 10000 1 8
bench 10000 0 4
echo "=== DONE" >> "$LOG"
