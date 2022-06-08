import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from gym.envs.mujoco import mujoco_env
import custom_envs.envs.utils as ce_utils

from collections import namedtuple

from custom_envs.envs.utils import *

# Throughout this file, position coordinates are tuples (x,y) where x and y
# are the x and y coordinates. This follows the standard mathematical
# convention for graphs.

BRIDGE_GRID_SIZE = 12
BRIDGE_MAX_TIME_STEPS = 1000


class JunctionTrafficLights(mujoco_env.MujocoEnv):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, constraint_regions=[], track_agent=False,
                 normalize_obs=False):
        # Environment setup.
        self.size = BRIDGE_GRID_SIZE
        self.max_time_steps = BRIDGE_MAX_TIME_STEPS
        self.start_pos = np.array([[6, 0], [6, 12], [0, 6], [12, 6]])
        self.goals = self.start_pos
        self.action_dim = 2
        self.state_dim = 3
        self.track_agent = track_agent
        self.normalize = normalize_obs
        self.constraint_regions = constraint_regions

        # Define spaces.
        self.observation_space = spaces.Box(
            low=np.array((0, 0, 0, 0, 0, 0, 0)), high=np.array((BRIDGE_GRID_SIZE, BRIDGE_GRID_SIZE, 1, 1, 1, 1, 1)),
            dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        scale = 1
        self.action_map_dict = {0: scale*np.array((1, 0)),
                                1: scale*np.array((-1, 0)),
                                2: scale*np.array((0, 1)),
                                3: scale*np.array((0, -1))}

        # Keep track of all visited states.
        self.make_visited_states_plot()

    def reset(self):
        self.start_i = 2  # np.random.randint(0, 4)
        start_x, start_y = self.start_pos[self.start_i][0], self.start_pos[self.start_i][1]
        self.curr_state = np.array(
            [start_x, start_y, 
            self.isNoRoadN(start_x, start_y),
            self.isNoRoadE(start_x, start_y),
            self.isNoRoadS(start_x, start_y),
            self.isNoRoadW(start_x, start_y),
            self.isOffRoad(start_x, start_y)
            ], dtype=np.float32)

        self.goal_i = 0  # np.random.randint(0, 4)
        # while self.goal_i == self.start_i:
        #    self.goal_i = np.random.randint(0, 4)
        self.goal = self.goals[self.goal_i]
        self.done = False
        self.timesteps = 0
        self.score = 0.
        self.add_new_visited_state(self.curr_state)
        return self.normalize_obs(self.curr_state)

    def seed(self, seed):
        pass

    def close(self):
        pass

    def step(self, action):
        assert hasattr(self, 'done'), 'Need to call reset first'
        assert self.done == False, 'Need to call reset first'

        # Get reward, next state.
        action = self.action_map_dict[action]
        self.curr_state, reward, self.done = self.reward(
            self.curr_state, action)
        self.score += reward
        self.timesteps += 1
        self.add_new_visited_state(self.curr_state)

        if self.timesteps > self.spec.max_episode_steps:
            self.done = True

        obs = self.normalize_obs(self.curr_state)

        return obs, reward, self.done, {}

    def reward(self, state, action):
        """
        Calculate reward.
        Done if agent reaches the goal.
        Fixed reward of 50 at goal.
        Reward of -1 - 0.1*p elsewhere where p=|action| if |action| > 6 else 0.
        Penalize agent -5 reward if it tries to move outside grid or through/to
        an invalid state.
        """
        done = False
        next_state = np.around(state[:2]+action, 6)

        # do not move when at a border
        if (np.min(next_state) < 0) or (np.max(next_state) > self.size):
            next_state = state[:2]

        if in_regions(state[:2], next_state, self.constraint_regions):
            reward = -10000
            #done = True
        else:
            reward = -np.sum(np.abs((next_state-self.goal)))/BRIDGE_GRID_SIZE

            if(next_state == self.goal).all():
                # Within 1 unit circle of the goal (states within unit circle
                # but outside grid have already been handled).
                reward = 10000
                done = True

        # add predicates to next_state
        next_state = np.append(next_state, self.isNoRoadN(
            next_state[0], next_state[1]))
        next_state = np.append(next_state, self.isNoRoadE(
            next_state[0], next_state[1]))
        next_state = np.append(next_state, self.isNoRoadS(
            next_state[0], next_state[1]))
        next_state = np.append(next_state, self.isNoRoadW(
            next_state[0], next_state[1]))
        next_state = np.append(next_state, self.isOffRoad(
            next_state[0], next_state[1]))

        return next_state, reward, done

    def template(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20, forward=True)

        # Fill plot with green.
        ax.add_patch(patches.Rectangle(
            xy=(0, 0), width=self.size, height=self.size,
            color='mediumspringgreen', fill=True
        ))

        ax.set_xticks(np.arange(0, self.size, 1))
        ax.set_yticks(np.arange(0, self.size, 1))
        ax.grid()

        # Add constraints.
        for origin, width, height in self.constraint_regions:
            ax.add_patch(patches.Rectangle(
                xy=origin, width=width, height=height,
                linewidth=1, color='#DE6B1F', fill=True
            ))

        # Add goal.
        add_circle(ax, self.goal, 'orange', 0.5, True)

        # Formatting.
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        plt.tick_params(axis='both', which='both', length=0, labelsize=0)

        # To remove the huge white borders.
        ax.margins(0)

        return fig, ax

    def render(self, *args, **kwargs):
        fig, ax = self.template()

        # Add agent at appropriate location.
        if hasattr(self, 'curr_state'):
            add_circle(ax, self.curr_state[:2], 'y', 0.2, False)
        else:
            add_circle(ax, self.start, 'y', 0.2, False)

        # Add score.
        fig.text(0, 1.04, 'Score: %06.2f' % self.score,
                 fontsize=25, ha='left', va='top', transform=ax.transAxes)
        fig.text(1, 1.04, 'Time: %03d' % self.timesteps,
                 fontsize=25, ha='right', va='top', transform=ax.transAxes)

        image = figure_to_array(fig)
        plt.close(fig=fig)

        return image

    def make_visited_states_plot(self):
        """This plots all states visited in this environment."""
        if self.track_agent:
            self.visited_state_fig, self.visited_state_ax = self.template()

    def add_new_visited_state(self, state):
        """Add a new visited state to plot."""
        if self.track_agent:
            add_circle(self.visited_state_ax, state[:2], 'y', 0.02, False)

    def save_visited_states_plot(self, save_name, append_to_title=None):
        """Call this at the very end to save the plot of states visited. This
        also closes the plot.
        """
        if self.track_agent:
            title = 'Visited States'
            if append_to_title is not None:
                title += ' | ' + append_to_title
            self.visited_state_ax.set_title(title, fontsize=25)

            self.visited_state_fig.savefig(save_name)
            plt.close(fig=self.visited_state_fig)

    def normalize_obs(self, obs):
        if self.normalize:
            obs = obs-self.observation_space.low
            obs *= 2
            obs /= (self.observation_space.high - self.observation_space.low)
            obs -= 1
        return obs

    def isNoRoadN(self, x, y):
        return (y == 7) and ((x < 4) or (x > 7))

    def isNoRoadS(self, x, y):
        return (y == 4) and ((x < 4) or (x > 7))

    def isNoRoadE(self, x, y):
        return (x == 7) and ((y < 4) or (y > 7))

    def isNoRoadW(self, x, y):
        return (x == 4) and ((y < 4) or (y > 7))

    def isOffRoad(self, x, y):
        if(y < 4) or (y > 7):
            return (x < 4) or (x > 7)
        return False

class ConstrainedJunctionTrafficLights(JunctionTrafficLights):
    def __init__(self, *args):
        # Lower bridge is constrained. Defined in the same way as
        # water regions were defined.
        constraint_regions = [
            (np.array((0, 0)), 4, 4), (np.array((0, 8)), 4, 4), (np.array((8, 0)), 4, 4), (np.array((8, 8)), 4, 4)]

        super().__init__(constraint_regions, *args)
