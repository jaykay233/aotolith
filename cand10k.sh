#!/bin/sh
# 10K decode candidate configs, one after another on a quiet board.
# Every run prints the "decode path:" banner first, so a flag the parser ate
# cannot be mistaken for "this optimisation does nothing".
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/p10k.txt}
N=${N:-12}
B=${B:-./m3c.new}

run() {
    tag=$1; shift
    echo "== $tag"
    $B -m "$M" -F "$P" -n "$N" "$@" 2>&1 |
        grep -E "decode path:|prefill: [0-9]+ tokens|attn split|gemm  |attn  |decode: .*tok/s|step 0 ->" || true
}

run "q4o+K8"          -K8 -X
run "q3+K8"           -K8 -q3 -X
run "q3+K8+win4096"   -K8 -q3 -X -Z 4096
run "q2+K8"           -K8 -q2 -X
