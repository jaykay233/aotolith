#!/bin/sh
# 10K chain A/B after the int8 weight copy is released before decode.
# Each row pays the ~76 s prefill; decode is n=12 so the number is the steady
# state and not a first-touch artefact. Run from the board's mega directory.
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export MEGA_STEPS=1
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/p10k.txt}
N=${N:-12}
B=${B:-./m3c.new}

run() {
    tag=$1; shift
    echo "== $tag"
    $B -m "$M" -F "$P" -n "$N" "$@" 2>&1 |
        grep -E "prefill:|released|\[step|gemm  |attn  |decode:|step 0 ->|step 1 ->" || true
}

run "q4o K8"  -X -K8
run "q3  noK8" -X -q3
