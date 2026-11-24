#!/bin/sh
# Is the 10K prefill deterministic? Same ids, same flags, twice, with the int8 KV
# hash and the TCM attach line in the log. If the hash or the first decode token
# moves between runs there is a race in the pipelined prefill, not a rounding
# difference.
export SPACEMIT_TCM_HOLD=1
export SPACEMIT_DECODE_CORES=8-15
export MEGA_KVHASH=1
M=${M:-/dev/shm/mc-q4o.gguf}
P=${P:-/tmp/p10k.txt}
B=${B:-./m3c.new}
for i in 1 2; do
    echo "== run $i"
    $B -m "$M" -F "$P" -n 2 -X 2>&1 |
        grep -E "\[tcm\]|kvhash|AnonHuge|prefill: [0-9]+ tokens|step 0 ->|step 1 ->" || true
done
