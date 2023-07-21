""" 1. Setup requirements and import dependencies
"""

import torch  # NN framework
import gym  # RL environment
import ray  # RL algorithms
import time
import utils
from ray.rllib.agents.ppo import PPOTrainer

""" 7. Evaluation: Exercise 3

Load and evaluate the policy you trained before. To do so, you can reuse code from Exercise 1, but instead of playing random action, use the new policy.
Instead of evaluating the policy on one trajectory, use an average over multiples trajectories.
"""

