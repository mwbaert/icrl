#!/bin/bash

python3.8 -V
pip3.8 install torch pandas tqdm imageio wandb matplotlib mpl-scatter-density glfw mujoco mujoco-py
pip3.8 install -e ./custom_envs

# for safety gridworlds
#export PYTHONPATH=${PYTHONPATH}:~/Documents/research/icrl/ai_safety_gridworlds
#pip install pycolab

wandb login fa44fb586bc0ae1f03502cb1b6268f3b916b163b

#"bash -c \"git clone https://github.com/mwbaert/icrl.git; cd icrl; chmod +x docker_init.sh; ./docker_init.sh; python run_me.py icrl --config_file config_icrl_logic_jtl.json -ws True --n_iters 20 --l2_coeff 1e-5\""