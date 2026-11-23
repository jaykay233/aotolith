#!/bin/sh
# Decode benchmark: plain greedy, stage breakdown, and the 3-bit variant.
#
# Decode is DRAM-bound, so the question is never "is the kernel fast" but "how
# close is the weight stream to what the board delivers". Two ceilings matter and
# they are far apart: a bare streaming read hits ~56 GB/s with 8 threads, but the
# vendor GEMM's serial load->use chain over 32-k blocks sustains only ~24 GB/s.
# Against 24 GB/s the i8xi4 blob costs 1.265 GB / 24 = 52.7 ms, i.e. 19.0 tok/s --
# short of the target. At i8xi3k it is 0.98 GB / 24 = 40.8 ms, i.e. 24.5 tok/s,
# which leaves ~9 ms for attention and the elementwise stages.
#
# Everything runs in one board session, because board sessions are the scarce
# resource and each must answer more than one question:
#   MEGA_RAW   bare streaming read        -> the memory ceiling
#   MEGA_BW    per-matrix GEMM bandwidth  -> what the vendor kernel actually gets
#   decode 0   weight stream alone        -> tok/s ceiling with no KV walk
#   decode 10k adds the 10K-context KV    -> attention's share
#
#   ./bench_decode.sh           # plain (i8xi4) only
#   ./bench_decode.sh -q3       # 3-bit (i8xi3k) as well
#   ./bench_decode.sh -b        # build only
set -u
HERE=$(dirname "$0")
. "$HERE/board_env.sh"
BUILD_ONLY=0
DO_Q3=0
for a in "$@"; do
    [ "$a" = "-b" ] && BUILD_ONLY=1
    [ "$a" = "-q3" ] && DO_Q3=1
done
G="g++ -O1 -std=c++20 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba_xsmtvdotii"

# A compile failure here previously went unnoticed because the pipeline ended in
# grep: the exit status came from grep, not the compiler, and the assembler says
# "Error:" with a capital E. Both are fixed -- no pipe on the compile, and the
# error scan is case-insensitive.
ensure_work || exit 1
$SSH "
    cd '$WORK' || exit 1
    if [ ! -f mega_decode.cpp ]; then echo 'missing mega_decode.cpp'; exit 1; fi
    for o in ime2.o rvv.o; do
        [ -f \$o ] || { echo \"missing \$o -- run recover_board.sh\"; exit 1; }
    done
    $G -o mega_decode mega_decode.cpp ime2.o rvv.o -lpthread -lm > bld.log 2>&1
    if [ \$? -ne 0 ]; then echo BUILD FAILED; head -20 bld.log; exit 1; fi
    grep -iE 'error|illegal' bld.log | head -5
    ls -l mega_decode > /dev/null || exit 1
    [ -f mc-q4o.gguf ] || { echo 'missing mc-q4o.gguf'; exit 1; }
    echo built
"
[ $? -ne 0 ] && exit 1
[ "$BUILD_ONLY" = "1" ] && exit 0

# $1 = extra env, $2 = extra args, $3 = label
run() {
    envs=$1; args=$2; label=$3
    echo
    echo "########## $label"
    $SSH "
        cd '$WORK'
        setsid sh -c 'env MEGA_WORK=$WORK $envs timeout -s KILL 900 ./mega_decode -m mc-q4o.gguf $args' </dev/null
    " 2>&1 | tail -30
}

run "MEGA_RAW=1"  "-p 1 -n 1"        "raw streaming ceiling"
run "MEGA_BW=1"   "-p 1 -n 1"        "per-matrix GEMM bandwidth (i8xi4)"
run ""            "-p 1 -n 32"       "decode prompt=1     i8xi4"
run ""            "-p 10240 -n 32"   "decode prompt=10240 i8xi4"

if [ "$DO_Q3" = "1" ]; then
    run "MEGA_BW=1" "-p 1 -n 1 -q3"      "per-matrix GEMM bandwidth (i8xi3k)"
    run ""          "-p 1 -n 32 -q3"     "decode prompt=1     i8xi3k"
    run ""          "-p 10240 -n 32 -q3" "decode prompt=10240 i8xi3k"
fi
