# train expert agent in the nominal environment but constrained with the ground truth cost function
# the model is evaluated on the constrained environment
#   num_threads should be 1, the true LGW cost function cannot handle multiple trheads for the moment
#   trained of 1e5 timesteps (original was 1e6)
#   after training you should copy the "files" directory to icrl/expert_data/LGW_ from the wandb directory
python run_me.py cpg --config_file config_exp_lgw.json
python run_me.py cpg --config_file config_exp_sw.json
python run_me.py cpg --config_file config_exp_jtl.json
python run_me.py cpg --config_file config_exp_t.json --timesteps 2e5 --learning_rate 0.0003 -dnc -dnr -dno

# generate rollouts from expert policy in the nominal environment
# copy rollouts from run_policy/ to EXPERT/
python run_me.py run_policy --load_dir icrl/expert_data/LGW_/ --env_id LGW-v0 -nr 20
python run_me.py run_policy --load_dir icrl/expert_data/SW/ --env_id SW-v0 -nr 20
# set reward threshold such that no violations will be part of the rollouts
# this requires to evaluate on the constrained environment
# keep this stochastic such that there is variation in the observed trajectories
python run_me.py run_policy --load_dir icrl/expert_data/JTL/ --env_id CJTL-v0 -nr 20 --red_light_prob 0.8
python run_me.py run_policy --load_dir icrl/expert_data/T/ --env_id CT-v0 -nr 20
python run_me.py run_policy --load_dir icrl/expert_data/RG/ --env_id CRG-v0 -nr 20
# train ICRL
    # first you have to copy the rollouts to the EXPERT directory
    # results are stored in icrl/wandb
python run_me.py icrl -p ICRL-FE2 --group LapGrid-ICRL -er 20 -ep icrl/expert_data/LGW_ -tei LGW-v0 -eei CLGW-v0 -tk 0.01 -cl 20 -clr 0.003 -ft 0.5e5 -ni 10 -bi 20 -dno -dnr -dnc
python run_me.py icrl --config_file config_icrl_jtl.json
python run_me.py icrl --config_file config_icrl_logic_jtl.json


# generate rollouts jtl (single goal)
# run_me.py cpg --config_file config_exp_jtl.json -tk 0.01 --timesteps 4e6 --learning_rate 0.0003 -ws True -dnc -dnr -dno
# generate rollouts jtl (multi goal)
# run_me.py cpg --config_file config_exp_jtl.json -tk 0.01 --timesteps 4e6 --learning_rate 0.0003 -ws True -dnc -dnr -dno --num_curriculum_updates 2

# train icrl jtl (single goal)
# run_me.py icrl --config_file config_icrl_jtl.json -ws True -dnc -dno -dnr