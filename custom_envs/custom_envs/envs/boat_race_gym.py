import numpy as np
from gym import spaces
from gym.envs.mujoco import mujoco_env
from ai_safety_gridworlds.ai_safety_gridworlds.environments.boat_race import BoatRaceEnvironment

LG_LAP_SIZE = 8

class BoatRaceGym(mujoco_env.MujocoEnv):
    def __init__(self, max_episode_steps=200):
        self.max_episode_steps = max_episode_steps
        self.viewer = None
        self.env = BoatRaceEnvironment()

        self.observation_space = spaces.Box(
            low=np.array((0,)), high=np.array(((LG_LAP_SIZE-1)*4,)),
            dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.initialize()

    def reset(self):
        self.env.reset()

    def step(self, action):
        self.env.step(action)
        
    