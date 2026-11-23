#!/bin/sh
# Speed matrix for the 10K chain: same prompt, same everything, only the decode
# weight format and attention path change. Run from the board's mega directory.
#
#   -q2   weights repacked to the vendor 2-bit layout (738 MB from 1416 MB)
#   -X    prefill K256 hp vendor kernel
#   -H    decode on the one-row hp kernel llama.cpp uses (alignment probe, slower)
#
# The int8 prefill copy is built from whichever source the run loads, so a -q2
# run prefills through 2-bit-derived int8 weights; expect its tokens to differ
# from the Q4_0 runs, which is why every row prints the first id and the margin.
set -e
export SPACEMIT_TCM_HOLD=1
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/p10k.txt}
N=${N:-4}
B=${B:-./m3c}

run() {
    tag=$1; shift
    echo "== $tag"
    $B -m "$M" -F "$P" -n "$N" "$@" 2>&1 |
        grep -E "prefill:|step 0 ->|step 1 ->|decode:.*tok/s|gemm  " || true
}

run "q4o int8kv"        -X
run "q2  int8kv"        -q2 -X
run "q2  hp decode"     -q2 -H -X
