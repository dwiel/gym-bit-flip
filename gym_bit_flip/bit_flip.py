import copy
import numpy as np
import gym
from gym import spaces
import random


class BitFlip(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, bit_length=16, max_steps=None, mean_zero=False):
        super(BitFlip, self).__init__()
        if bit_length < 1:
            raise ValueError("bit_length must be >= 1, found {}".format(bit_length))
        self.bit_length = bit_length
        self.mean_zero = mean_zero

        if max_steps is None:
            # default to bit_length
            self.max_steps = bit_length
        elif max_steps == 0:
            self.max_steps = None
        else:
            self.max_steps = max_steps

        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Discrete(bit_length)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=0, high=1, shape=(bit_length,)),
                "goal": spaces.Box(low=0, high=1, shape=(bit_length,)),
            }
        )

        self.reset()

    def _terminate(self):
        return (self.state == self.goal).all() or self.steps >= self.max_steps

    def _reward(self):
        return self.compute_reward(self.state, self.goal)

    def step(self, action):
        # action is an int in the range [0, self.bit_length)
        self.state[action] = int(not self.state[action])
        self.steps += 1

        return (self._get_obs(), self._reward(), self._terminate(), {})

    def reset(self):
        self.steps = 0

        self.state = np.array([random.choice([1, 0]) for _ in range(self.bit_length)])

        # make sure goal is not the initial state
        self.goal = self.state
        while (self.goal == self.state).all():
            self.goal = np.array(
                [random.choice([1, 0]) for _ in range(self.bit_length)]
            )

        return self._get_obs()

    def _mean_zero(self, x):
        if self.mean_zero:
            return (x - 0.5) / 0.5
        else:
            return x

    def _get_obs(self):
        return {
            "state": copy.copy(self._mean_zero(self.state)),
            "goal": copy.copy(self._mean_zero(self.goal)),
        }

    def _render(self, mode="human", close=False):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """
        this function is expected by rlkit if running a her algorithm
        """
        return -1 if (achieved_goal != desired_goal).any() else 0
