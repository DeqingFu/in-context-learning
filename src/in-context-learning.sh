#!/bin/bash

source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/transformers

python train.py --config conf/linear_regression.yaml