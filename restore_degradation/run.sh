#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 train.py -train_batch_size=4 -exp_name=dim_test -task=all