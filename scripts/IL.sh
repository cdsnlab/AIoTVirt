#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=1 python  ~/LIGETI/run.py \
                        --name resnet18 \
                        --alpha 0.1 \
