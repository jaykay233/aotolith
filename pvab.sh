#!/bin/sh
# PV: one pass per kv at m4 versus four passes at m1, with f16 and int8 V.
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/p10k.txt}
N=${N:-12}
B=${B:-./m3c.new}
for cfg in "-X" "-P4 -X" "-K8 -X" "-P4 -K8 -X"; do
    echo "== $cfg"
    $B -m "$M" -F "$P" -n "$N" $cfg 2>&1 |
        grep -E "decode path:|attn split|gemm  |attn  |decode: .*tok/s|step 0 ->" || true
done
