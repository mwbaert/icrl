#!/bin/bash

echo 'alias python=/usr/bin/python3.8' > ~/.bashrc
. ~/.bashrc
python -V 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jovyan/.mujoco/mujoco210/bin
pip install -e ./custom_envs
pip install torch
wandb login fa44fb586bc0ae1f03502cb1b6268f3b916b163b