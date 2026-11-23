#!/bin/sh
# Shared board layout, sourced by the benchmark and recovery scripts.
#
# /dev/shm is tmpfs, i.e. RAM. The model, the repacked weight caches, the binaries
# and the vendor objects all used to live there: about 3.6 GB of it, and every
# recovery wiped the lot, so a reboot cost a 13.6 s re-quantize plus a full repack
# before a single measurement could run. They live on UFS now and survive reboots.
#
# Only storage moves. The weights are still streamed from RAM at runtime -- decode
# reads 1.26 GB per token against a ~24 GB/s ceiling, and UFS is an order of
# magnitude short of that -- so the board reads them once at startup into an
# aligned, huge-page-backed blob and never touches UFS again in the hot loop.
#
# Override with MEGA_WORK=/some/dir; the C++ side reads the same variable.

WORK=${MEGA_WORK:-/home/bianbu/workspace/mega}
BOARD=${BOARD:-spacemit-k3}
export WORK BOARD

SSH="ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 $BOARD"

# Reports what the board has, and whether the UFS work directory exists yet.
board_layout() {
    $SSH "
        echo \"work=\$HOME\"
        df -h /home /dev/shm 2>/dev/null | sed 's/^/  /'
        grep -E '^(MemTotal|MemAvailable):' /proc/meminfo | sed 's/^/  /'
        [ -d '$WORK' ] && ls -l '$WORK' | head -12 || echo '  $WORK missing'
    "
}

ensure_work() {
    $SSH "mkdir -p '$WORK' && echo '$WORK ready'" || return 1
}
