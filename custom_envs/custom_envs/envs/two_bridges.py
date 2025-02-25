import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from gym.envs.mujoco import mujoco_env

from collections import namedtuple

from custom_envs.envs.utils import *

# Throughout this file, position coordinates are tuples (x,y) where x and y
# are the x and y coordinates. This follows the standard mathematical
# convention for graphs.

BRIDGE_GRID_SIZE = 20
BRIDGE_MAX_TIME_STEPS = 1000

class TwoBridges(mujoco_env.MujocoEnv):
    """
    Drawn to scale for a 20x20 grid.

    ######################
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #                    #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #                    #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    #        ####        #
    ######################

    """
    metadata = {"render.modes": ["rgb_array"]}
    def __init__(self, constraint_regions=[], start=(0,0), track_agent=False,
                 normalize_obs=False):
        # Environment setup.
        self.size = BRIDGE_GRID_SIZE
        self.max_time_steps = BRIDGE_MAX_TIME_STEPS
        self.start = start
        self.goal = np.array([self.size, 0])
        self.action_dim = 2
        self.state_dim = 2
        self.track_agent = track_agent
        self.normalize = normalize_obs

        # Water regions. Each tuple contains the bottom left corner's
        # coordinates of a water region, its width and height respectively.
        self.water_regions = [(np.array((4,0)), 4, 5),
                              (np.array((4,6)), 4, 8),
                              (np.array((4,15)), 4, 5)]

        # Constraint regions.
        self.constraint_regions = constraint_regions

        # Define spaces.
        self.observation_space = spaces.Box(
                low=np.array((0,0)), high=np.array((BRIDGE_GRID_SIZE,BRIDGE_GRID_SIZE)),
                dtype=np.float32)
        self.action_space = spaces.Box(
                low=np.array((0,0)), high=np.array((BRIDGE_GRID_SIZE,BRIDGE_GRID_SIZE)),
                dtype=np.float32)

        # Keep track of all visited states.
        self.make_visited_states_plot()

    def reset(self):
        self.curr_state = np.array(self.start, dtype=np.float32)
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
        assert len(action) == self.action_dim

        # Project action to valid range in high and low is defined.
        try:
            action = np.min([action, self.action_space.high], axis=0)
            action = np.max([action, self.action_space.low], axis=0)
        except:
            pass

        # Get reward, next state.
        self.curr_state, reward, self.done = self.reward(self.curr_state, action)
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
            xy=(0,0), width=self.size, height=self.size,
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

class DiscreteTwoBridges(TwoBridges):
    """Discrete version of bridge environment."""
    def __init__(self, *args):
        super().__init__(*args)
        # Choose scale carefully to ensure that the agent can cross
        # the bridges.
        scale = 0.7
        self.action_map_dict = {0: scale*np.array((1,0)),
                                1: scale*np.array((-1,0)),
                                2: scale*np.array((0,1)),
                                3: scale*np.array((0,-1))}

        self.action_space = spaces.Discrete(4)

    def step(self, action):
        """Thin wrapper. Calls the parent step class with action
        mapped to the corresponding `continuous' version.
        """
        return super().step(self.action_map_dict[action])


class DenseDiscreteTwoBridges(DiscreteTwoBridges):
    """Here the reward on right half of the environment and the bridges
    is dense. The agent recieves a reward proportional to how close it
    is to the goal. The reward in the left half of the environment is
    still sparse and negative. So the agent is incentized to get to the
    right side as quickly as it can. Since the agent starts closer to
    the lower bridge, so the optimal solution would be to use that.
    """
    def __init__(self, *args):
        super().__init__(*args)

    def reward(self, state, action):
        """
        Calculate reward.
        Done if agent reaches the goal.
        Fixed reward of 100 at goal.
        Penalize agent -2 reward if it tries to move outside grid or go
        through/on constraint or water.
        -1 reward in left half.
        In right half, the reward is inversely proportional to distance
        to the goal (the scale is higher if agent is in lower bottom half).
        """
        done = False
        next_state = np.around(state+action, 6)

        if (np.min(next_state) < 0 or np.max(next_state) > self.size or
            in_regions(state, next_state, self.water_regions) or
            in_regions(state, next_state, self.constraint_regions)):
            # Tried to go out of grid or move through/to an invalid state.
            # Back in same spot. Penalize slightly.
            next_state = state
            reward = -2.

        elif np.sum((self.goal-next_state)**2) < 1:
            # Within 1 unit circle of the goal (states within unit circle
            # but outside grid have already been handled).
            #reward = 250.
            #done = True
            reward = 12

        elif next_state[0] > self.water_regions[0][0][0]:
            # In right half. States in right half but in invalid regions
            # (e.g. water have already been handled).
            reward = 10/(np.sum((self.goal - next_state)**2)**(1/2))

            if next_state[1] < self.water_regions[1][0][1]:
                # Higher reward in bottom half region.
                #reward *= self.size
                reward *= 1
        else:
            # In left half.
            reward = -1.

        return next_state, reward, done


class ConstrainedDenseDiscreteTwoBridges(DenseDiscreteTwoBridges):
    def __init__(self, *args):
        # Lower bridge is constrained. Defined in the same way as
        # water regions were defined.
        constraint_regions = [(np.array((4,5)), 4, 1)]

        super().__init__(constraint_regions, *args)


class DDConstrainedDenseDiscreteTwoBridges(ConstrainedDenseDiscreteTwoBridges):
    def __init__(self, *args):
        #start = (0,0)
        #start = (0,20)
        start = (3,5)
        super().__init__(start, *args)


class ContinuousTwoBridges(TwoBridges):
    def __init__(self, *args):
        super().__init__(*args, normalize_obs=False)

        # Overwrite spaces.
        self.observation_space = spaces.Box(
                low=np.array((0,0,0)), high=np.array((BRIDGE_GRID_SIZE,BRIDGE_GRID_SIZE,np.inf)),
                dtype=np.float32)
        action_lim = 2.
        self.action_space = spaces.Box(
                low=np.array((-action_lim,-action_lim)), high=np.array((action_lim,action_lim)),
                dtype=np.float32)

        self._reset_noise_scale = 0.
        self.init_qpos = np.array([0,0,0])

    def reset(self):
        self.score = 0.
        self.timesteps = 0.
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.qpos = self.init_qpos + np.random.uniform(
            low=noise_low, high=noise_high, size=(3,))

        return self.qpos

    def step(self, action):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        # Compute increment in each direction
        self.qpos[2] += action[1]
        ori = self.qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # Ensure that the robot is within reasonable range
        qpos = self.qpos.copy()
        qpos[0] = np.clip(qpos[0] + dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1] + dy, -self.size, self.size)

        self.qpos[:2], reward, done = self.reward(self.qpos[:2], qpos[:2], action)
        self.score += reward
        self.timesteps += 1

        return self.qpos, reward, done, dict(action_mag=np.linalg.norm(action), ori=ori,
                                             dx=dx, dy=dy)

    def reward(self, state, next_state, action):
        done = False

        if (np.min(next_state) < 0 or np.max(next_state) > self.size or
            in_regions(state, next_state, self.water_regions) or
            in_regions(state, next_state, self.constraint_regions)):
            # Tried to go out of grid or move through/to an invalid state.
            # Back in same spot. Penalize slightly.
            next_state = state
            reward = -2.

        elif np.sum((self.goal-next_state)**2) < 1:
            # Within 1 unit circle of the goal (states within unit circle
            # but outside grid have already been handled).
            reward = 250.
            #done = True
            #reward = 12

        elif next_state[0] > self.water_regions[0][0][0]:
            # In right half. States in right half but in invalid regions
            # (e.g. water have already been handled).
            reward = 10/(np.sum((self.goal - next_state)**2)**(1/2))

            if next_state[1] < self.water_regions[1][0][1]:
                # Higher reward in bottom half region.
                reward *= self.size
                #reward *= 1
        else:
            # In left half.
            reward = -1.

        return next_state, reward, done

class ConstrainedContinuousTwoBridges(ContinuousTwoBridges):
    def __init__(self, *args):
        # Lower bridge is constrained. Defined in the same way as
        # water regions were defined.
        constraint_regions = [(np.array((4,5)), 4, 1)]

        super().__init__(constraint_regions, *args)