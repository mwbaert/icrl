import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from gym.envs.mujoco import mujoco_env

from collections import namedtuple

from custom_envs.envs.utils import *

GRID_SIZE = 5
MAX_TIME_STEPS = 50


class SumoTl(mujoco_env.MujocoEnv):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, constraint_regions=[], track_agent=False, normalize_obs=True):
        # Environment setup.
        self.size = GRID_SIZE
        self.max_time_steps = MAX_TIME_STEPS
        self._start = [(0, 2), (4, 2), (2, 0), (2, 4)]
        #self.goal = np.array([self.size, 0])
        self.goal = [(0, 2), (4, 2), (2, 0), (2, 4)]
        self.action_dim = 2
        self.state_dim = 2
        self.track_agent = track_agent
        self.normalize = normalize_obs

        # Water regions. Each tuple contains the bottom left corner's
        # coordinates of a water region, its width and height respectively.
#        self.water_regions = [(np.array((4,0)), 4, 5),
#                              (np.array((4,6.5)), 4, 3),
#                              (np.array((4,10.5)), 4, 3),
#                              (np.array((4,15)), 4, 5)]
#
#        self.water_regions = [(np.array((4,0)), 4, 1),
#                              (np.array((4,2.5)), 4, 7),
#                              (np.array((4,10.5)), 4, 7),
#                              (np.array((4,19)), 4, 1)]
#       self.water_regions = [(np.array((4, 0)), 4, 1),
#                             (np.array((4, 2.5)), 4, 6.5),
#                             (np.array((4, 11)), 4, 6.5),
#                             (np.array((4, 19)), 4, 1)]

        # Constraint regions.
        # self.constraint_regions = constraint_regions

        # Define spaces.
        self.observation_space = spaces.Box(
            low=np.array((0, 0)), high=np.array((GRID_SIZE, GRID_SIZE)),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array((0, 0)), high=np.array((GRID_SIZE, GRID_SIZE)),
            dtype=np.float32)

        # Keep track of all visited states.
        self.make_visited_states_plot()

        self.random_start = True
        self.first = True

    @property
    def start(self):
        if self.random_start:
            return self._start[np.random.randint(len(self._start))]
        else:
            if self.first:
                self.first = False
                self.prev_start = np.random.randint(len(self._start))
            else:
                if self.prev_start == 0:
                    self.prev_start = 1
                elif self.prev_start == 1:
                    self.prev_start = 0
                else:
                    raise ValueError
            return self._start[self.prev_start]

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        pass

    def reset(self):
        self.curr_state = np.array(self.start, dtype=np.float32)
        self.done = False
        self.timesteps = 0
        self.score = 0.
        self.add_new_visited_state(self.curr_state)
        return self.normalize_obs(self.curr_state)

    def step(self, action):
        assert hasattr(self, 'done'), 'Need to call reset first'
        assert self.done == False, 'Need to call reset first'
        assert len(action) == self.action_dim

        # Project action to valid range in high and low is defined.
        try:
            action = np.min([action, self.action_space.high], axis=0)
            action = np.max([action, self.action_space.low], axis=0)
        except:
            pass

        # Get reward, next state.
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
        next_state = np.around(state+action, 6)
        act_mag = np.sum(action**2)**(1/2)
        reward = -1 - 0.1*act_mag * int(act_mag > 6)

        if (np.min(next_state) < 0 or np.max(next_state) > self.size or
            in_regions(state, next_state, self.water_regions) or
                in_regions(state, next_state, self.constraint_regions)):
            # Tried to move out of grid or through/to an invalid state.
            # Back in same spot. Penalize slightly.
            reward -= 5
            next_state = state

        elif np.sum((self.goal-next_state)**2) < 1:
            # Within 1 unit circle of the goal (states within unit circle
            # but outside grid have already been handled).
            reward = 50
            done = True

        return next_state, reward, done

    def template(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20, forward=True)

        # Fill plot with green.
        ax.add_patch(patches.Rectangle(
            xy=(0, 0), width=self.size, height=self.size,
            color='mediumspringgreen', fill=True
        ))

        # Add water.
        for origin, width, height in self.water_regions:
            ax.add_patch(patches.Rectangle(
                xy=origin, width=width, height=height,
                linewidth=1, color='deepskyblue', fill=True
            ))

        # Add constraints.
        for origin, width, height in self.constraint_regions:
            ax.add_patch(patches.Rectangle(
                xy=origin, width=width, height=height,
                linewidth=1, color='#DE6B1F', fill=True
            ))

        # Add goal.
        add_circle(ax, self.goal, 'orange', 1, True)

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
            add_circle(ax, self.curr_state, 'y', 0.2, False)
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
            add_circle(self.visited_state_ax, state, 'y', 0.02, False)

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


class DiscreteSumoTl(SumoTl):
    """Discrete version of bridge environment."""

    def __init__(self, *args):
        super().__init__(*args)
        # Choose scale carefully to ensure that the agent can cross
        # the bridges.
        s = 0.7
        self.action_map_dict = {0: s*np.array((1, 0)),
                                1: s*np.array((-1, 0)),
                                2: s*np.array((0, 1)),
                                3: s*np.array((0, -1))}

        self.action_space = spaces.Discrete(4)

    def step(self, action):
        """Thin wrapper. Calls the parent step class with action
        mapped to the corresponding `continuous' version.
        """
        return super().step(self.action_map_dict[action])


class ConstrainedDiscreteSumoTl(DiscreteSumoTl):
    def __init__(self, *args):
        # Lower bridge is constrained. Defined in the same way as
        # water regions were defined.
        #        constraint_regions = [(np.array((4,5)), 4, 1.5),
        #                              (np.array((4,13.5)), 4, 1.5)]
        constraint_regions = [(np.array((4, 1)), 4, 1.5),
                              (np.array((4, 17.5)), 4, 1.5)]

        super().__init__(constraint_regions, *args)
