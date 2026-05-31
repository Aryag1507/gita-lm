#!/usr/bin/env bash
#
# Overnight pipeline: train the LoRA adapter, train the reward model, then run
# the full benchmark. Kick this off before bed and review results in the morning.
#
# Usage:
#   ./run_overnight.sh              # default: 10 epochs
#   EPOCHS=5 ./run_overnight.sh     # override epoch count
#
# All stdout/stderr is tee'd to logs/run_<timestamp>.log so you can review what
# happened even if the terminal closed.

set -euo pipefail

# Use the Python that has the project dependencies installed.
PYTHON="${PYTHON:-/Users/aryagupta/anaconda3/bin/python3}"
EPOCHS="${EPOCHS:-10}"

mkdir -p logs results
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/run_${TIMESTAMP}.log"

echo "=== gita-lm overnight run @ ${TIMESTAMP} ===" | tee "$LOG"
echo "Python : ${PYTHON}" | tee -a "$LOG"
echo "Epochs : ${EPOCHS}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_step () {
    local name="$1"; shift
    echo ">>> [$(date +%H:%M:%S)] START: ${name}" | tee -a "$LOG"
    if "$@" 2>&1 | tee -a "$LOG"; then
        echo ">>> [$(date +%H:%M:%S)] DONE:  ${name}" | tee -a "$LOG"
    else
        echo "!!! [$(date +%H:%M:%S)] FAILED: ${name} — aborting" | tee -a "$LOG"
        exit 1
    fi
    echo "" | tee -a "$LOG"
}

run_step "LoRA fine-tuning"   "$PYTHON" train.py --epochs "$EPOCHS"
run_step "Reward model train" "$PYTHON" train_reward.py
run_step "Benchmark"          "$PYTHON" benchmark.py

echo "=== All steps complete. Results in results/, full log in ${LOG} ===" | tee -a "$LOG"
