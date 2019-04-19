import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import gym_minigrid
from gym import wrappers
from time import time
env = gym.make('MiniGrid-GridCity-4S30Static-v0')
env.render()
env = wrappers.Monitor(env, "./gym-results", force=True)
img = env.reset()

for _ in range(1000):
    action = 1
    observation, reward, done, info = env.step(action)
    if done: break
env.close()
