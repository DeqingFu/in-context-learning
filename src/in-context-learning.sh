#!/bin/bash

source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/in-context-learning

python train.py --config conf/linear_regression.yaml