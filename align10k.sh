#!/bin/sh
# End-to-end alignment at 10K: the same 10000 ids the greedy reference
# (refgreedy, Q4_0 exact weights) read, then eight greedy tokens out of the
# megakernel, compared token by token.
#
# The reference prints 220 5298 242 8647 1808 20974 5760 47.
#   -X        Q4_0 exact weights, vendor K256 hp prefill, f16 KV   (alignment anchor)
#   -X -K8    same with the int8 decode KV cache                    (speed path)
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/p10k.txt}
B=${B:-./m3c.new}

for cfg in "-X" "-X -K8"; do
    echo "== $cfg"
    $B -m "$M" -F "$P" -n 8 $cfg 2>&1 |
        grep -E "prefill:|step [0-9] ->|gemm  |attn  |decode: .*tok/s" || true
done
