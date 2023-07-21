import os

from ray.rllib.env.wrappers.atari_wrappers import MaxAndSkipEnv, WarpFrame, FrameStack, MonitorEnv, NoopResetEnv
import gym
from ray.tune import register_env
from ray.tune.logger import UnifiedLogger
import pickle


# register Mario environment with the correct wrappers for rllib
try:
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT


    class EpisodicLifeEnv(gym.Wrapper):

        def __init__(self, env):
            """Make end-of-life == end-of-episode, but only reset on true game
            over. Done by DeepMind for the DQN and co. since it helps value
            estimation.
            """
            gym.Wrapper.__init__(self, env)
            self.lives = 0
            self.was_real_done = True

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.was_real_done = done
            # check current lives, make loss of life terminal,
            # then update lives to handle bonus lives
            lives = self.env.unwrapped._life
            if self.lives > lives > 0:
                # for Qbert sometimes we stay in lives == 0 condtion for a few fr
                # so its important to keep lives > 0, so that we only reset once
                # the environment advertises done.
                done = True
            self.lives = lives
            return obs, reward, done, info

        def reset(self, **kwargs):
            """Reset only when lives are exhausted.
            This way all states are still reachable even though lives are episodic,
            and the learner need not know about any of this behind-the-scenes.
            """
            if self.was_real_done:
                obs = self.env.reset(**kwargs)
            else:
                # no-op step to advance from terminal/lost life state
                obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped._life
            return obs


    class MarioWrapper(gym.Wrapper):
        def __init__(self):
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v1")
            env = JoypadSpace(env, RIGHT_ONLY)  # SIMPLE_MOVEMENT
            env = MonitorEnv(env)
            # env = NoopResetEnv(env, noop_max=30)
            # env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            env = WarpFrame(env, 84)
            env = FrameStack(env, 4)
            super().__init__(env)


    env_name = "SuperMarioBrosWrap-1-1-v1"
    gym.envs.register(id=env_name, entry_point='utils:MarioWrapper')
    register_env(env_name, lambda x: MarioWrapper())

finally:
    pass


# used to customize where the logs are saved in rllib
def custom_log(custom_path):
    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        return UnifiedLogger(config, custom_path, loggers=None)

    return logger_creator


# fix a bug when reloading model from rllib when using pytorch
def without_optimizer(path):
    extra_data = pickle.load(open(path, "rb"))
    worker = pickle.loads(extra_data['worker'])
    worker['state']['default_policy'].keys()
    if '_optimizer_variables' in worker['state']['default_policy']:
        del worker['state']['default_policy']['_optimizer_variables']
        extra_data['worker'] = pickle.dumps(worker)
        with open(path, 'wb') as dest:
            pickle.dump(extra_data, dest)
    return path

