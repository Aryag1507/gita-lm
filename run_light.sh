#!/usr/bin/env bash
#
# Low-resource overnight/daytime run — keeps the laptop usable while training.
#
# Layers three kinds of throttling:
#   1. light mode in the app  : CPU device, capped torch threads, max_length=256
#   2. nice -n 19             : lowest OS scheduler priority, so interactive apps
#                               always win contention for the CPU
#   3. thread env caps        : stop BLAS/OMP libraries from grabbing every core
#
# Usage:
#   ./run_light.sh
#
# Runs in the foreground with logging. Safe to run while you keep working.

set -euo pipefail

PYTHON="${PYTHON:-/Users/aryagupta/anaconda3/bin/python3}"

# Keep math libraries from spawning a thread per core.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/light_${TIMESTAMP}.log"

echo "=== gita-lm LIGHT run @ ${TIMESTAMP} (nice -n 19, CPU, 4 threads) ===" | tee "$LOG"

run_step () {
    local name="$1"; shift
    echo ">>> [$(date +%H:%M:%S)] START: ${name}" | tee -a "$LOG"
    if nice -n 19 "$@" 2>&1 | tee -a "$LOG"; then
        echo ">>> [$(date +%H:%M:%S)] DONE:  ${name}" | tee -a "$LOG"
    else
        echo "!!! [$(date +%H:%M:%S)] FAILED: ${name} — aborting" | tee -a "$LOG"
        exit 1
    fi
    echo "" | tee -a "$LOG"
}

run_step "LoRA fine-tuning (light)"   "$PYTHON" train.py --light
run_step "Reward model train (light)" "$PYTHON" train_reward.py --light
run_step "Benchmark"                  "$PYTHON" benchmark.py

echo "=== Light run complete. Results in results/, log in ${LOG} ===" | tee -a "$LOG"
