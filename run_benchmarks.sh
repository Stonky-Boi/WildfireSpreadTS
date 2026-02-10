#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
EPOCHS=100
SEED=0
BASE_LOG_DIR="$HOME/wildfire_logs"
mkdir -p "$BASE_LOG_DIR"

# --------- ENV ---------
REPO_ROOT="$HOME/WildfireSpreadTS"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"
echo "Starting benchmarks..."

# Helper function to run a benchmark
run_benchmark()
{
    local model_name=$1
    shift
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local model_log_dir="$BASE_LOG_DIR/$model_name/$timestamp"
    mkdir -p "$model_log_dir"
    local log_file="$model_log_dir/run.log"
    
    echo "--- Starting $model_name Benchmark ---"
    echo "Logging to $log_file"
    
    # Run the command and log both stdout and stderr
    "$@" 2>&1 | tee -a "$log_file"
}

# --------- UNet ---------
run_benchmark "unet" python3 src/train.py \
  --config=cfgs/unet/res18_monotemporal.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --seed_everything=${SEED} \
  --trainer.max_epochs=${EPOCHS} \
  --do_test=True

# --------- Logistic Regression ---------
run_benchmark "logistic_regression" python3 src/train.py \
  --config=cfgs/LogisticRegression/full_run.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --seed_everything=${SEED} \
  --trainer.max_epochs=${EPOCHS} \
  --do_test=True

# --------- UTAE ---------
run_benchmark "utae" python3 src/train.py \
  --config=cfgs/UTAE/all_features.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --seed_everything=${SEED} \
  --trainer.max_epochs=${EPOCHS} \
  --do_test=True

# --------- ConvLSTM ---------
run_benchmark "convlstm" python3 src/train.py \
  --config=cfgs/convlstm/full_run.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --seed_everything=${SEED} \
  --trainer.max_epochs=${EPOCHS} \
  --do_test=True

echo "All benchmarks complete."