#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Ensemble experiment runner for Stokes flow training
# Runs multiple trials of different training configurations in parallel

set -e

# Configuration
NUM_TRIALS=10           # Number of trials per experiment type
MAX_PARALLEL=3          # Maximum parallel jobs

# Create logs directory
LOGS_DIR="experiment_logs"
mkdir -p "$LOGS_DIR"

# Timestamp for this experiment batch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Starting experiment ensemble at $TIMESTAMP"
echo "Trials per config: $NUM_TRIALS"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Logs directory: $LOGS_DIR"
echo "========================================"

# Function to wait for background jobs if we've hit the limit
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

# Function to run a single experiment
run_experiment() {
    local exp_type=$1
    local trial=$2
    local script=$3
    local physics=$4
    local bc=$5
    local log_file="$LOGS_DIR/${exp_type}_${TIMESTAMP}_trial${trial}.log"
    
    echo "[Starting] $exp_type trial $trial -> $log_file"
    
    python "$script" \
        ++add_physics_loss=$physics \
        ++add_bc_loss=$bc \
        ++validation_image="${exp_type}_trial${trial}.png" \
        > "$log_file" 2>&1
    
    echo "[Finished] $exp_type trial $trial"
}

# Export function for parallel execution
export -f run_experiment
export LOGS_DIR
export TIMESTAMP

echo ""
echo "Phase 1: Data-only experiments (no physics, no BC)"
echo "---------------------------------------------------"
for trial in $(seq 1 $NUM_TRIALS); do
    wait_for_jobs
    run_experiment "data-only" $trial "train_domino.py" "False" "False" &
done

echo ""
echo "Phase 2: Data + Physics experiments"
echo "------------------------------------"
for trial in $(seq 1 $NUM_TRIALS); do
    wait_for_jobs
    run_experiment "data-physics" $trial "train_domino.py" "True" "True" &
done

echo ""
echo "Phase 3: Staged training experiments"
echo "-------------------------------------"
for trial in $(seq 1 $NUM_TRIALS); do
    wait_for_jobs
    run_experiment "staged" $trial "train_domino_staged.py" "True" "True" &
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Logs saved to: $LOGS_DIR/"
echo "========================================"
echo ""
echo "To analyze results, run:"
echo "  python parse_experiment_logs.py --logs_dir $LOGS_DIR --timestamp $TIMESTAMP"
