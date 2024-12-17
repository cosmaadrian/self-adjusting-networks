#!/bin/bash
cd ..
set -e

STUFF="--config_file configs/san.yaml --mode run --group explore-decay --batch_size 64"
torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py \
        --name size:256-decay:0.001\
        --optimizer_args.weight_decay 0.001 \
        --model_args.dmodel 256 \
        --actually_stop_at 8192 \
        --n_train_iters 250000 \
        $STUFF

torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py \
        --name size:256-decay:0.01\
        --optimizer_args.weight_decay 0.01 \
        --model_args.dmodel 256 \
        --actually_stop_at 8192 \
        --n_train_iters 250000 \
        $STUFF

torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py \
        --name size:256-decay:0.1\
        --optimizer_args.weight_decay 0.1 \
        --model_args.dmodel 256 \
        --actually_stop_at 8192 \
        --n_train_iters 250000 \
        $STUFF

torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py \
        --name size:256-decay:1\
        --optimizer_args.weight_decay 1.0 \
        --model_args.dmodel 256 \
        --actually_stop_at 8192 \
        --n_train_iters 250000 \
        $STUFF