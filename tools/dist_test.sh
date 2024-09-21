#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
VISIBLE_GPUS=$4  # Add an argument for visible GPUs

# Set which GPUs are visible
export CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS


MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py $CONFIG -C $CHECKPOINT --launcher pytorch ${@:5}
