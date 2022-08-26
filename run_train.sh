#!/bin/bash

GPU_INDEX=$1

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
python3 train.py