#!/bin/bash
cd ..
set -e

STUFF="--config_file configs/san.yaml --mode run --group test --batch_size 64"
torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py \
        --name test1\
        --actually_stop_at 220000 \
        --n_train_iters 250000 \
        $STUFF