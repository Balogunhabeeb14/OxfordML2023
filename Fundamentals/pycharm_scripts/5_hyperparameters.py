""" Setup requirements and import dependencies
"""

import torch  # NN framework
import gym  # RL environment
import ray  # RL algorithms
import time
import utils
from ray.rllib.agents.ppo import PPOTrainer

""" 8. Hyperparameters Optimisation: Exercise 4

Try obtaining higher rewards by retraining with other hyperparameters.
"""

# default values of some hyperparemeters of PPO
default_config = {
    "framework": "torch",
    "num_gpus": 0,
    "num_workers": 1,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    "gamma": 0.99,
    "lr": 0.0001,
    "train_batch_size": 200,
}
