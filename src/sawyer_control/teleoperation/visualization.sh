#!/bin/bash

#ENV_NAME="sawyer-pickup-banana-v2"
#ENV_NAME="sawyer-drawer-open-v0"
ENV_NAME="sawyer-pick-place-cube-v0"

FILENAME="/home/tung/workspace/rlhf_bench/iql-pytorch-sawyer/datasets/${ENV_NAME}/${ENV_NAME}_episode_${1}.pkl"
# FILENAME="/home/tung/workspace/hrl_bench/preference_rl/sawyer_dataset/${ENV_NAME}/sawyer-drawer-open-v0_episode_${1}.pkl"

python visualize_demonstration.py --file ${FILENAME}
#python visualize_demonstration_in_transition.py --file ${FILENAME}