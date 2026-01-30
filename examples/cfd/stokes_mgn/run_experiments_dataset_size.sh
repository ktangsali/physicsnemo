#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset size ablation experiment runner for Stokes flow training
# Varies num_training_samples and compares data-only vs data+physics training

# Don't use set -e as it interferes with background jobs and pgrep

# Configuration
NUM_TRIALS=5                          # Number of trials per configuration
MAX_PARALLEL=5                        # Maximum parallel jobs (16 cores / 2 num_workers)
DATASET_SIZES=(16 32) # Training sample sizes to test

# Create logs directory
LOGS_DIR="experiment_logs_dataset_size"
mkdir -p "$LOGS_DIR"

# Timestamp for this experiment batch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Calculate total experiments
NUM_SIZES=${#DATASET_SIZES[@]}
TOTAL_EXPERIMENTS=$((NUM_SIZES * 2 * NUM_TRIALS))  # sizes × (data-only + data-physics) × trials

echo "========================================"
echo "Dataset Size Ablation Experiments"
echo "========================================"
echo "Started at: $TIMESTAMP"
echo "Dataset sizes: ${DATASET_SIZES[*]}"
echo "Trials per config: $NUM_TRIALS"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Logs directory: $LOGS_DIR"
echo "========================================"
echo ""

# Function to count running experiment processes
count_running_jobs() {
    pgrep -f "train_domino.py.*num_training_samples" | wc -l
}

# Function to wait for background jobs if we've hit the limit
wait_for_jobs() {
    while [ $(count_running_jobs) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

# Function to run a single experiment
run_experiment() {
    local exp_type=$1
    local dataset_size=$2
    local trial=$3
    local physics=$4
    local bc=$5
    local log_file="$LOGS_DIR/${exp_type}_n${dataset_size}_${TIMESTAMP}_trial${trial}.log"
    
    echo "[Starting] $exp_type | n=$dataset_size | trial $trial"
    
    python train_domino.py \
        ++add_physics_loss=$physics \
        ++add_bc_loss=$bc \
        ++num_training_samples=$dataset_size \
        ++validation_image="${exp_type}_n${dataset_size}_trial${trial}.png" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[Finished] $exp_type | n=$dataset_size | trial $trial ✓"
    else
        echo "[FAILED]   $exp_type | n=$dataset_size | trial $trial (exit code: $exit_code)"
    fi
}

# Export function and variables for subshells
export -f run_experiment
export LOGS_DIR
export TIMESTAMP

# Track job count
job_count=0

echo "Launching experiments (interleaved by dataset size)..."
echo "-------------------------------------------------------"
echo "Order: For each size -> data-only (all trials) -> data-physics (all trials) -> next size"
echo ""

# Loop over dataset sizes - complete each size before moving to next
for size in "${DATASET_SIZES[@]}"; do
    
    echo ""
    echo "========================================"
    echo "Dataset size: n=$size"
    echo "========================================"
    
    # Phase 1: Data-only experiments for this size
    echo ""
    echo "--- Launching data-only (n=$size) ---"
    for trial in $(seq 1 $NUM_TRIALS); do
        wait_for_jobs
        run_experiment "data-only" $size $trial "False" "False" &
        ((job_count++))
    done
    
    # Wait for all data-only trials to complete before starting data-physics
    echo "Waiting for data-only n=$size to complete..."
    wait
    echo "✓ Data-only n=$size complete!"
    
    # Phase 2: Data + Physics experiments for this size
    echo ""
    echo "--- Launching data-physics (n=$size) ---"
    for trial in $(seq 1 $NUM_TRIALS); do
        wait_for_jobs
        run_experiment "data-physics" $size $trial "True" "True" &
        ((job_count++))
    done
    
    # Wait for all data-physics trials to complete before moving to next size
    echo "Waiting for data-physics n=$size to complete..."
    wait
    echo "✓ Data-physics n=$size complete!"
    
    echo ""
    echo "========================================"
    echo "✓ All experiments for n=$size COMPLETE"
    echo "  You can now analyze results for this size!"
    echo "========================================"
    
done

echo ""
echo "All $job_count experiments finished!"

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo "Logs saved to: $LOGS_DIR/"
echo ""
echo "Log file naming convention:"
echo "  {exp_type}_n{dataset_size}_{timestamp}_trial{N}.log"
echo ""
echo "Examples:"
echo "  data-only_n32_${TIMESTAMP}_trial1.log"
echo "  data-physics_n800_${TIMESTAMP}_trial5.log"
echo ""
echo "To analyze results, run:"
echo "  python parse_dataset_size_logs.py --logs_dir $LOGS_DIR --timestamp $TIMESTAMP"
echo "========================================"
