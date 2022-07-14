import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from gym.envs.mujoco import mujoco_env
import custom_envs.envs.utils as ce_utils
import random

from collections import namedtuple

from custom_envs.envs.utils import *

# Throughout this file, position coordinates are tuples (x,y) where x and y
# are the x and y coordinates. This follows the standard mathematical
# convention for graphs.

GRID_SIZE = 12
TEST_MAX_TIME_STEPS = 1000
ORIENTATION_N = 0
ORIENTATION_E = 1
ORIENTATION_S = 2
ORIENTATION_W = 3

STATE_X = 0
STATE_Y = 1
STATE_ORIENTATION = 2
STATE_NO_ROAD_LEFT = 3
STATE_NO_ROAD_RIGHT = 4
STATE_NO_ROAD_IN_FRONT = 5
STATE_OFF_ROAD = 6
STATE_SIZE = 7

ACTION_STILL = 0
ACTION_STRAIGHT = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


class TestEnv(mujoco_env.MujocoEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, constraint_regions=[], constraint_state_action_pairs=[], track_agent=False, normalize_obs=False):
        # Environment setup.
        self.size = GRID_SIZE
        self.max_time_steps = TEST_MAX_TIME_STEPS
        self.start_pos = np.array([[0, 0], [0, 5]])
        self.goal_pos = np.array(
            [[11, 0], [11, 4], [6, 11]])
        self.track_agent = track_agent
        self.constraint_regions = constraint_regions
        self.constraint_state_action_pairs = constraint_state_action_pairs
        self.normalize = True  # normalize_obs
        self.goal = self.goal_pos[0]
        self.red_light_prob = 0.5

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(STATE_SIZE,), dtype=np.float64)
        self.obs_min = np.array([0, 0, 0, 0, 0, 0, 0])
        self.obs_max = np.array(
            [GRID_SIZE-1, GRID_SIZE-1, 4, 1, 1, 1, 1])

        # |still|straight|left|right|
        self.action_space = spaces.Discrete(4)
        scale = 1

        # absolute actions
        self.action_map_dict = {0: scale*np.array((0, 1)),  # N
                                1: scale*np.array((1, 0)),  # E
                                2: scale*np.array((0, -1)),  # S
                                3: scale*np.array((-1, 0))}  # W

        # Keep track of all visited states.
        self.make_visited_states_plot()

    def reset(self):
        start_i = 0 #np.random.randint(0, 2)
        start_x, start_y = self.start_pos[start_i][0], self.start_pos[start_i][1]

        self.curr_state = np.array(
            [start_x, start_y,
             ORIENTATION_E,
             self.isNoRoadLeft(start_x, start_y, ORIENTATION_E),
             self.isNoRoadRight(start_x, start_y, ORIENTATION_E),
             self.isNoRoadInFront(start_x, start_y, ORIENTATION_E),
             self.isOffRoad(start_x, start_y)
             ], dtype=np.float32)

        self.done = False
        self.timesteps = 0
        self.score = 0.
        self.add_new_visited_state(self.curr_state)

        return self.normalize_obs(self.curr_state)

    def set_goal(self, goal_i):
        self.goal = self.goal_pos[0]

    def set_red_light_prob(self, prob):
        self.red_light_prob = prob

    def seed(self, seed):
        pass

    def close(self):
        pass

    def step(self, action):
        assert hasattr(self, 'done'), 'Need to call reset first'
        assert self.done == False, 'Need to call reset first'

        # Get reward, next state.
        self.curr_state, reward, self.done = self.reward(
            self.curr_state, action)
        self.score += reward
        self.timesteps += 1
        self.add_new_visited_state(self.curr_state)

        if self.timesteps > self.spec.max_episode_steps:
            self.done = True

        obs = self.curr_state
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
        next_state = self.applyAction(state, action)

        reward = - \
            np.sum(np.abs((next_state[:STATE_Y+1]-self.goal)))/GRID_SIZE

        if(next_state[:STATE_Y+1] == self.goal).all():
            reward = 0  # reward already zero?
            done = True
        # else:
        #    reward = -0.01

        if ce_utils.in_regions(state, next_state, self.constraint_regions):
            reward = -100
        # elif (len(self.constraint_regions) > 0) and (state[-1] == 1) and (action == 1):
        #    reward = -100

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
            add_circle(ax, self.curr_state[:STATE_Y+1], 'y', 0.2, False)
            add_triangle(
                ax, self.curr_state[:STATE_Y+1], self.curr_state[STATE_ORIENTATION], 'r', 0.2, False)

            # add traffic light
            # if self.curr_state[STATE_IN_FRONT_OF_RED_LIGHT]:
            #    add_circle(ax, [2, 2], 'r', 0.4, False)
            # else:
            #    add_circle(ax, [2, 2], 'g', 0.4, False)
        else:
            add_circle(ax, self.start, 'y', 0.2, False)
            add_triangle(ax, self.start, 1, 'r', 0.2, False)
            add_circle(ax, [2, 2], 'g', 0.4, False)

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
            obs = obs-self.obs_min
            obs *= 2
            obs /= (self.obs_max - self.obs_min)
            obs -= 1
        return obs

    def applyAction(self, state, action):
        next_state = [0 for _ in range(STATE_SIZE)]

        next_state = self.updateOrientation(state, action, next_state)
        next_state = self.updatePosition(state, action, next_state)
        next_state = self.updatePropositionals(next_state)
        # if (next_state[STATE_X] == 3) and (state[STATE_X] != 3):
        #    next_state[STATE_IN_FRONT_OF_RED_LIGHT] = 1
        # else:
        #    next_state[STATE_IN_FRONT_OF_RED_LIGHT] = 0

        return next_state

    def updateOrientation(self, state, action, next_state):
        if action == ACTION_LEFT:
            next_state[STATE_ORIENTATION] = (state[STATE_ORIENTATION]+3) % 4
        elif action == ACTION_RIGHT:
            next_state[STATE_ORIENTATION] = (state[STATE_ORIENTATION]+1) % 4
        else:
            next_state[STATE_ORIENTATION] = state[STATE_ORIENTATION]

        return next_state

    def updatePosition(self, state, action, next_state):
        next_state[:STATE_Y+1] = state[:STATE_Y+1]

        if action != ACTION_STILL:
            next_state[:STATE_Y +
                       1] += self.action_map_dict[state[STATE_ORIENTATION]]

            # do not move when at a border
            if (np.min(next_state[:STATE_Y+1]) < 0) or (np.max(next_state[:STATE_Y+1]) >= self.size):
                next_state[:STATE_Y+1] = state[:STATE_Y+1]

        return next_state

    def updatePropositionals(self, next_state):
        next_state[STATE_NO_ROAD_LEFT] = self.isNoRoadLeft(
            next_state[STATE_X], next_state[STATE_Y], next_state[STATE_ORIENTATION])
        next_state[STATE_NO_ROAD_RIGHT] = self.isNoRoadRight(
            next_state[STATE_X], next_state[STATE_Y], next_state[STATE_ORIENTATION])
        next_state[STATE_NO_ROAD_IN_FRONT] = self.isNoRoadInFront(
            next_state[STATE_X], next_state[STATE_Y], next_state[STATE_ORIENTATION])
        next_state[STATE_OFF_ROAD] = self.isOffRoad(
            next_state[STATE_X], next_state[STATE_Y])

        # if (next_state[STATE_X] == 3) and (random.random() < self.red_light_prob):
        #    next_state[STATE_IN_FRONT_OF_RED_LIGHT] = 1
        # else:
        #    next_state[STATE_IN_FRONT_OF_RED_LIGHT] = 0

        return next_state

    def isNoRoadLeft(self, x, y, orientation):
        return self.isNoRoad(x, y, (orientation+3) % 4)

    def isNoRoadRight(self, x, y, orientation):
        return self.isNoRoad(x, y, (orientation+1) % 4)

    def isNoRoadInFront(self, x, y, orientation):
        return self.isNoRoad(x, y, orientation)

    def isNoRoad(self, x, y, orientation):
        if orientation == ORIENTATION_N:
            if(y == 7) and ((x < 4) or (x > 7)):
                return 1.0
            return 0.0
        elif orientation == ORIENTATION_S:
            if(y == 4) and ((x < 4) or (x > 7)):
                return 1.0
            return 0.0
        elif orientation == ORIENTATION_E:
            if(x == 7) and ((y < 4) or (y > 7)):
                return 1.0
            return 0.0
        elif orientation == ORIENTATION_W:
            if(x == 4) and ((y < 4) or (y > 7)):
                return 1.0
            return 0.0

    def isOffRoad(self, x, y):
        if((y < 4) or (y > 7)) and ((x < 4) or (x > 7)):
            return 1.0
        return 0.0


class ConstrainedTestEnv(TestEnv):
    def __init__(self, *args):
        # Lower bridge is constrained. Defined in the same way as
        # water regions were defined.
        # constraint_regions = [
        #    (np.array((0, 0)), 4, 4), (np.array((0, 8)), 4, 4), (np.array((8, 0)), 4, 4), (np.array((8, 8)), 4, 4)]
        constraint_regions = [
            (np.array((4, 0)), 4, 2)
        ]
        # "-1" represent don't cares
        # constraint_state_action_pairs = [
        #    ([-1, -1, -1, -1, -1, -1, -1, 1], ACTION_STRAIGHT)
        # ]

        #super().__init__(constraint_regions, constraint_state_action_pairs, *args)
        super().__init__(constraint_regions, None, *args)
