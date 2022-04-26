from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from gym.envs.mujoco import mujoco_env

from collections import namedtuple

from custom_envs.envs.utils import *
from wandb import agent

GRID_SIZE = 9


class ScobeeWorld(mujoco_env.MujocoEnv):
    """
    Constructs a square gridworld environment.
    Some states (grid cells) are constrained.
    The agent receives a reward when he arrives at the goal state.
    The episodes ends when the agent visits an invalid (constrained) state

    Agent always starts at the bottom left of the grid.

    Constraint net is expected to learn to constrain the constrained states.
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            max_episode_steps=200,
            normalize_obs=False
    ):
        """
        Args:
            max_episode_steps (int): Number of maximum steps in environment.
                                     Must always be a multiple (lap_size - 1)*4.
                                     If not then imbalanced reward will not
                                     correspond to correct performance.
        """
        all_actions = (0, 1)   # [Forward, Backward]
        self.grid_size = GRID_SIZE
        self.max_episode_steps = max_episode_steps
        self.viewer = None
        self.normalize = normalize_obs

        # Define spaces.
        # the x and y coordinate
        # can we replace this with a discrete space?
        # self.observation_space = spaces.Box(
        #    np.array([0.0, 0.0]), np.array(
        #        [self.grid_size-1.0, self.grid_size-1.0])
        # )
        self.observation_space = spaces.Box(
            low=np.array((0,)), high=np.array(((self.grid_size**2)-1,)),
            dtype=np.float32)

        # self.observation_space = spaces.Box(
        #    low=0.0, high=1.0, shape=(self.grid_size, self.grid_size), dtype=np.float32)
        # four cardinal directions and action zero
        self.action_space = spaces.Discrete(5)

        # Initialize
        self.initialize()

    def seed(self, seed):
        # Only for compatibility; environment does not has any randomness
        np.random.seed(seed)

    def initialize(self):
        self.number_of_cells = (self.grid_size)**2
        #ssert (self.max_episode_steps % self.number_of_cells == 0)
        self.rewards = np.zeros(self.number_of_cells)
        self.rewards[self.xy_to_idx(8, 0)] = 1.0
        self.start_pos = 0
        self.reset()

    def reset(self):
        self.current_pos = self.start_pos
        self.current_time = 0
        self.reward_so_far = 0

        return self.normalize_obs(np.array([self.current_pos]))

    def get_next_obs(self, obs, action):
        x, y = self.idx_to_xy(obs)
        if action == 1:
            x = min(x+1, self.grid_size-1)
        elif action == 2:
            x = max(x-1, 0)
        elif action == 3:
            y = min(y+1, self.grid_size-1)
        elif action == 4:
            y = max(y-1, 0)

        return self.xy_to_idx(x, y)

    def step(self, action):
        done = False
        self.current_pos = self.get_next_obs(self.current_pos, action)
        x, y = self.idx_to_xy(self.current_pos)

        self.current_time += 1
        if self.current_time == self.max_episode_steps:
            done = True
        elif (x == self.grid_size-1) & (y == 0):
            done = True

        self.reward_so_far += self.rewards[self.current_pos]

        obs = self.normalize_obs(np.array([self.current_pos]))

        return (obs,
                self.rewards[self.current_pos],
                done,
                {"info": 0})

    def idx_to_xy(self, idx):
        assert(idx < self.number_of_cells)
        return idx // self.grid_size, idx % self.grid_size

    def xy_to_idx(self, x, y):
        assert(x <= (self.grid_size-1))
        assert(y <= (self.grid_size-1))

        return (self.grid_size*x) + y

    def render(self, mode=None, camera_id=None):
        agent_position = self.current_pos
        return figure_to_array(self.plot(agent_position))

    def plot(self, agent_position, save_name=None):
        a = np.ones((self.grid_size, self.grid_size))*-1
        for i in range(self.number_of_cells):
            x, y = self.idx_to_xy(i)
            a[x, y] += self.rewards[i]

        # Start should be shaded a little lighter
        a[0, 0] = -0.4

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        c = ax.pcolor(a, edgecolors='w', linewidths=2,
                      cmap='pink_r', vmin=-1.0, vmax=1.0)

        # To detect agent position, add a dummy value to that point
        arr = c.get_array()
        #agent_x, agent_y = self.idx_to_xy(agent_position)
        # arr[agent_position] += 32
        arr[np.ravel_multi_index(self.idx_to_xy(agent_position),
                                 (self.grid_size, self.grid_size))] += 32

        # Adding text
        for p, value in zip(c.get_paths(), arr):
            x, y = p.vertices[:-2, :].mean(0)
            if value > 30:
                ax.text(x, y, 'A', ha="center", va="center",
                        color='#DE6B1F', fontsize=38)
            elif value >= 0:
                string = str('\$')
                ax.text(x, y, string, ha="center", va="center",
                        color='#FFDF00', fontsize=25)

        # Add current reward and number of traverals at top
        # fig.text(0, 1.04, 'Score: {}/{}'.format(self.reward_so_far,
        #                                        self.traversals),
        #         fontsize=25, ha='left', va='top', transform=ax.transAxes)
        fig.text(0, 1.04, 'Score: {}'.format(self.reward_so_far),
                 fontsize=25, ha='left', va='top', transform=ax.transAxes)
        fig.text(1, 1.04, 'Time: %03d' % self.current_time,
                 fontsize=25, ha='right', va='top', transform=ax.transAxes)

        co_ords = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                co_ords.append((x, y))

        x, y = zip(*co_ords)
        x, y = np.array(x) + 0.5, np.array(y) + 0.5
        ax.scatter(x, y)

        if save_name is not None:
            fig.savefig(save_name)
        else:
            return fig

    def normalize_obs(self, obs):
        if self.normalize:
            obs = obs-self.observation_space.low
            obs *= 2
            obs /= (self.observation_space.high - self.observation_space.low)
            obs -= 1
        return obs


class ConstrainedScobeeWorld(ScobeeWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards[self.xy_to_idx(4, 0)] = -1.0

    def step(self, action):
        next_obs, reward, done, info = super().step(action)

        if reward == -1.0:
            done = True

        return next_obs, reward, done, info
