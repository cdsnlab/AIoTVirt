#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=1 python  $(pwd)/run.py \
                        --name resnet18 
