import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO

#################################################################################
# CONSTANTS
MODEL_PATH = "./models/modelBAD0.00001-finish.zip"  # Path to the saved model

#################################################################################
# PREPROCESS AND SETUP
env = gym_super_mario_bros.make(
    "SuperMarioBros-v3", apply_api_compatibility=True, render_mode="human"
)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

# Load the trained model
model = PPO.load(MODEL_PATH)

# Run the model
for i in range(10):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
env.close()
