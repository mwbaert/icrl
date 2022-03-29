import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from gym.envs.mujoco import mujoco_env

from collections import namedtuple

from custom_envs.envs.utils import *

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
            normalize_obs=True
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
        self.observation_space = spaces.Box()
        # the x and y coordinate
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.grid_size, self.grid_size), dtype=np.float32)
        # four cardinal directions and action zero
        self.action_space = spaces.Discrete(5)

        # Initialize
        self.initialize()

    def seed(self, seed):
        # Only for compatibility; environment does not has any randomness
        np.random.seed(seed)

    def initialize(self):
        self.rewards = np.zeros((self.grid_size, self.grid_size))
        self.rewards[self.grid_size-1, 0] = 1.0
        self.start_pos = (0, 0)
        self.reset()

    def reset(self):
        self.current_pos = self.start_pos
        self.current_time = 0
        self.reward_so_far = 0

        return self.current_pos

    def get_next_obs(self, obs, action):
        x, y = obs
        if action == 0:
            new_position = obs
        elif action == 1:
            new_position = (x+1, y)
        elif action == 2:
            new_position = (x-1, y)
        elif action == 3:
            new_position = (x, y+1)
        elif action == 4:
            new_position = (x, y-1)

        return new_position

    def step(self, action):
        done = False
        x, y = self.current_pos

        if action == 1:
            x += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            y += 1
        elif action == 4:
            y -= 1

        self.current_time += 1
        if self.current_time == self.max_episode_steps:
            done = True
        elif (x == self.grid_size-1) & (y == 0):
            done = True

        self.reward_so_far += self.rewards[x, y]

        return ((x, y),
                self.rewards[x, y],
                done)

    def _idx_to_xy(self, idx):
        # not needed i think
        if idx < self.lap_size:
            return int(idx), 0
        elif idx < self.lap_size*2-1:
            return self.lap_size-1, int(idx - self.lap_size + 1)
        elif idx < self.lap_size*3-2:
            return int(self.lap_size*3 - 3 - idx), self.lap_size-1
        else:
            return 0, int(self.number_of_cells - idx)

    def render(self, mode=None, camera_id=None):
        # TODO
        agent_position = self.current_pos
        return figure_to_array(self.plot(agent_position))

    def plot(self, agent_position, save_name=None):
        # TODO

        a = np.ones((self.lap_size, self.lap_size))*-1
        b = self.rewards

        a[:, 0] = b[:self.lap_size]
        a[-1, 1:] = b[self.lap_size:self.lap_size*2-1]
        a[1:, -1] = b[self.lap_size*2-2:self.lap_size*3-3][::-1]
        a[0, 1:] = b[self.lap_size*3-3:self.number_of_cells][::-1]

        # Start should be shaded a little lighter
        a[0, 0] = -0.4
        a[:, 0] = b[:self.lap_size]
        a[-1, 1:] = b[self.lap_size:self.lap_size*2-1]
        a[1:, -1] = b[self.lap_size*2-2:self.lap_size*3-3][::-1]
        a[0, 1:] = b[self.lap_size*3-3:self.number_of_cells][::-1]

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        c = ax.pcolor(a, edgecolors='w', linewidths=2,
                      cmap='pink_r', vmin=-1.0, vmax=1.0)

        # To detect agent position, add a dummy value to that point
        arr = c.get_array()
        arr[np.ravel_multi_index(self._idx_to_xy(agent_position),
                                 (self.lap_size, self.lap_size))] += 32

        # Adding text
        for p, value in zip(c.get_paths(), arr):
            x, y = p.vertices[:-2, :].mean(0)
            if value > 31:
                ax.text(x, y, 'A', ha="center", va="center",
                        color='#DE6B1F', fontsize=38)
            # ===
            # If you want coins + agent in coins cells, uncomment the following block
            # ===
            # elif value > 32:
            #     ax.text(0.5*x, 1.05*y, 'A', ha="left", va="top", color='white', fontsize=25)
            #     string = str('\$'*int(value - 32))
            #     print(string)
            #     ax.text(x, 0.95*y, string, ha="center", va="center", color='#FFDF00', fontsize=25)
            elif value > 0:
                string = str('\$'*int(value))
                ax.text(x, y, string, ha="center", va="center",
                        color='#FFDF00', fontsize=25)

        # Add current reward and number of traverals at top
        fig.text(0, 1.04, 'Score: {}/{}'.format(self.reward_so_far,
                                                self.traversals),
                 fontsize=25, ha='left', va='top', transform=ax.transAxes)
        fig.text(1, 1.04, 'Time: %03d' % self.current_time,
                 fontsize=25, ha='right', va='top', transform=ax.transAxes)

        ob = np.arange(0, 40)
        co_ords = []
        for i in ob:
            co_ords.append(self._idx_to_xy(i))
        x, y = zip(*co_ords)
        x, y = np.array(x) + 0.5, np.array(y) + 0.5
        ax.scatter(x, y)

        if save_name is not None:
            fig.savefig(save_name)
        else:
            return fig

    # def normalize_obs(self, obs):
    #    if self.normalize:
    #        obs = obs-self.observation_space.low
    #        obs *= 2
    #        obs /= (self.observation_space.high - self.observation_space.low)
    #        obs -= 1
    #    return obs


class ConstrainedScobeeWorld(ScobeeWorld):
    # TODO
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        done = False
        if action == 0:
            self.current_pos += 1
            if self.current_pos == self.number_of_cells:
                self.traversals += 1
                self.current_pos = 0
            reward = self.rewards[self.current_pos]
        elif action == 1:
            reward = -1   # penalize the backward action
            done = True

        self.current_time += 1
        if self.current_time == self.max_episode_steps:
            done = True

        self.reward_so_far += reward

        obs = self.normalize_obs(np.array([self.current_pos]))

        return (obs,
                reward,
                done,
                {"traversals_so_far": self.traversals})

    def get_next_obs(self, obs, action):
        if action == 0:
            new_position = obs + 1
        elif action == 1:
            new_position = obs
        return new_position
