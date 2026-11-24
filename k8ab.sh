#!/bin/sh
# Decode attention with and without the int8 KV cache, at the top-level context
# sizes that matter. The two configs must be written so the parser cannot eat a
# flag: put -K8 before -X, and read the "decode path:" banner to confirm.
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
M=${M:-/dev/shm/mc-q4o.gguf}
B=${B:-./m3c.new}
N=${N:-12}
for P in "$@"; do
    for cfg in "-X" "-K8 -X"; do
        echo "== $P [$cfg]"
        $B -m "$M" -F "$P" -n "$N" $cfg 2>&1 |
            grep -E "decode path:|attn split|gemm  |attn  |decode: .*tok/s|step 0 ->" || true
    done
done
