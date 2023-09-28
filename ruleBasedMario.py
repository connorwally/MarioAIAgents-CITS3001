from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
import numpy as np

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


# search screen for noteworthy objects
# observation = table of RGB values representing the screen
# theme = level theme determined beforehand (g = ground, u = underground, c = castle, w = water)
def look(observation, theme="g"):
    objects = {}
    for row in range(253):
        for p in range(
            253
        ):  # doesn't go all the way to the end since it can't identify things cut off by the screen border
            if observation[row][p] == [228, 92, 16]:
                if goomba(observation, row, p, theme):
                    objects[(row, p)] = "goomba"

    return


def goomba(observation, row, col, theme):
    if (
        theme == "g"
        and observation[row + 1][col] == [228, 92, 16]
        and observation[row + 2][col] == [228, 92, 16]
        and observation[row + 3][col] == [228, 92, 16]
        and observation[row][col + 1] == [240, 208, 176]
        and observation[row + 1][col + 1] == [240, 208, 176]
        and observation[row + 2][col + 1] == [240, 208, 176]
        and observation[row + 3][col + 1] == [240, 208, 176]
        and observation[row][col + 2] == [0, 0, 0]
        and observation[row + 1][col + 2] == [0, 0, 0]
        and observation[row + 2][col + 2] == [0, 0, 0]
        and observation[row + 3][col + 2] == [240, 208, 176]
        and observation[row][col + 3] == [228, 92, 16]
        and observation[row + 1][col + 3] == [0, 0, 0]
        and observation[row + 2][col + 3] == [240, 208, 176]
        and observation[row + 3][col + 3] == [240, 208, 176]
    ):
        return True


def agent(observation, info, ticks):
    # Get relevant values from info
    xPos = info["x_pos"]
    yPos = info["y_pos"]
    powerup = info["status"]
    level = [info["world"], info["stage"]]

    # Use HUD coin to determine level theme
    # hudPixels = [observation[31][93], observation[32][93]]
    """
	#slowdown for debug purposes
	for i in range(1, 50000):
		print(i)
	"""

    for i in range(1, 250):
        print()

    # print(hudPixels)
    # Determine input to make
    if ticks % 100 > 50 and ticks % 100 < 100:
        return actions["RIGHT_JUMP_RUN"]
    return actions["RIGHT_RUN"]


def main():
    ticks = 34
    env = gym.make(
        "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
    )
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    observation = env.reset()
    rewardTotal = 0
    done = False
    env.reset()
    action = 4
    while not done:
        ticks += 1
        obs, reward, terminated, truncated, info = env.step(action)
        action = agent(obs, info, ticks)
        rewardTotal += reward
        """
		for x in obs:
			for y in x:
				print(y, end=', ')
			print('_______')
		print('=======')
		"""
        done = terminated or truncated
    print("This agent obtained a total reward of " + str(rewardTotal))


if __name__ == "__main__":
    main()
