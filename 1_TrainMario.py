import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

################################################################################
# CONSTANTS

CHECKPOINT_DIR = "./models"  # Where to save the model
LOG_DIR = "./logs"  # Where to save the tensorboard logs


SAVE_FREQUENCY = 100000  # How many steps should pass before saving the model

LEARNING_RATE = 0.00001  # Learning rate for the model
TOTAL_TIMESTEPS = 20000000  # Total number of steps to train the model
NUMBER_OF_STEPS = 512  # Number of steps to run on each environment per update


################################################################################
# CALLBACK FOR SAVING PROGRESS
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)

        return True


#################################################################################
# CREATE AND PREPROCESS THE ENVIRONMENT
env = gym_super_mario_bros.make(
    "SuperMarioBros-v3", apply_api_compatibility=True, render_mode="human"
)  # Create the environment

env = Monitor(env, LOG_DIR)  # Create a monitor for tensorboard logging

# The following lines reduce the information being fed into the model/reduce the state space
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(
    **kwargs
)  # A fix for the JoypadSpace wrapper
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # Set the joypad space to simple movement
env = GrayScaleObservation(env, keep_dim=True)  # Convert the image to grayscale
env = DummyVecEnv([lambda: env])  # Create a dummy vector environment
env = VecFrameStack(env, 4, channels_order="last")  # Stack the last 4 frames together

#################################################################################
# INITIALISE THE CALLBACK AND PPO MODEL

callback = TrainAndLoggingCallback(check_freq=SAVE_FREQUENCY, save_path=CHECKPOINT_DIR)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=LEARNING_RATE,
    n_steps=NUMBER_OF_STEPS,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
model.save("mario_ppo")
