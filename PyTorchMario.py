import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

################################################################################
"""

PLEASE NOTE IN ITS CURRENT STATE, THIS IS A COPY FROM THE FOLLOWING VIDEO:
https://youtu.be/2eeYqJ0uBKE?si=QAD43UqAAK7tBdD9

MINOR CHANGES HAVE BEEN MADE TO THE CODE TO MAKE IT WORK WITH THE CURRENT
VERSIONS OF THE LIBRARIES USED.


"""
################################################################################
# CREATE THE ENVIRONMENT

# Define the moves for easier understanding later on
moves = [
    "NOTHING",
    "RIGHT",
    "RIGHT_JUMP",
    "RIGHT_RUN",
    "RIGHT_JUMP_RUN",
    "JUMP",
    "LEFT",
    "LEFT_JUMP",
    "LEFT_RUN",
    "LEFT_JUMP_RUN",
    "DOWN",
    "UP",
]

actions = dict(zip(moves, range(12)))


# Create the environment
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)

#################################################################################
# PREPROCESS THE ENVIRONMENT

# Simplify controls for the AI
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = JoypadSpace(env, SIMPLE_MOVEMENT)


# Gray scale the environment
env = GrayScaleObservation(env, keep_dim=True)

# Wrap it inside a dummy environment
env = DummyVecEnv([lambda: env])

# Stack the frames
env = VecFrameStack(env, n_stack=4, channels_order="last")

#################################################################################
# RUN THE ENVIRONMENT

# Reset the environment, and set the done flag to false.
env.reset()

done = False
for step in range(100000):
    # While the game is not done, take a step
    while not done:
        obs, reward, terminated, info = env.step([env.action_space.sample()])
        done = terminated
env.close()
