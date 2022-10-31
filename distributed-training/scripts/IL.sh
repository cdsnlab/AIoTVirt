#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python  $(pwd)/run.py \
                        --name resnet18 
