""" Setup requirements and import dependencies
"""

import torch  # NN framework
import gym  # RL environment
import ray  # RL algorithms
import time
import utils
from ray.rllib.agents.ppo import PPOTrainer

""" 9. Customised Environment: Exercise 5
By looking at the simple Shower environment (defined below), create your own new environment to reproduce the following MDP: https://en.wikipedia.org/wiki/File:Markov_Decision_Process.svg

"""

import numpy as np
from ray.tune import register_env


class ShowerEnv(gym.Env):
    def __init__(self):
        # 3 discrete actions
        self.action_space = gym.spaces.Discrete(3)
        # single continuous scalar to describe the state
        self.observation_space = gym.spaces.Box(low=0, high=75, shape=(1,))

    def reset(self):
        # initial temperature between 33° and 43°
        self.temperature = np.array([38 + np.random.randint(-5, 5)]).astype(float)
        # number of step in one shower
        self.shower_length = 10

    def step(self, action):
        # the new temperature depends on the action and some random values
        self.temperature += np.random.randint(-3, 2) + action

        # decrease length
        self.shower_length -= 1

        # stop the episode once the length is 0
        done = self.shower_length <= 0

        # compute reward
        if self.temperature >= 37 and self.temperature <= 39:
            reward = 1
        else:
            reward = -1

        return self.temperature, reward, done, []


# register the environment with rllib
register_env('MyShowerEnv-v0', lambda x: ShowerEnv())
# instead of using gym.make use the following
env = ShowerEnv()

env.reset()
env.step(1)



""" 10. Try other algorithms: Exercise 6

Rllib features numerous algorithms. Visit https://docs.ray.io/en/latest/rllib/rllib-algorithms.html, pick one that suits
 you needs and use it to train an agent on your customised environment. 
"""
