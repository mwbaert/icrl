# train expert agent in LGW environment
#   num_threads should be 1, the true LGW cost function cannot handle multiple trheads for the moment
#   trained of 1e5 timesteps (original was 1e6)
#   after training you should copy the "files" directory to icrl/expert_data/LGW_ from the wandb directory
python run_me.py cpg --config_file config_exp_lgw.json

# generate rollouts from expert policy in LGW environment
python run_me.py run_policy --load_dir icrl/expert_data/LGW_/ --env_id LGW-v0 -nr 20

# train ICRL
    # first you have to copy the rollouts to the EXPERT directory
    # results are stored in icrl/wandb
python run_me.py icrl -p ICRL-FE2 --group LapGrid-ICRL -er 20 -ep icrl/expert_data/LGW_ -tei LGW-v0 -eei CLGW-v0 -tk 0.01 -cl 20 -clr 0.003 -ft 0.5e5 -ni 10 -bi 20 -dno -dnr -dnc
