import argparse
import gym
import comet_ml
import random
import numpy as np
try:
    import gym_minigrid
except ImportError:
    pass
import sys
sys.path.append(".")

import utils
from model import DQNModel
import csv
import os

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--num-trajectories", type=int, default=10,
                    help="number of frames of training (default: 10e7)")
args = parser.parse_args()


# Get model directory
model_dir = "storage/Hyule_random"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# Init environment
env = gym.make(args.env)
print("Number of Actions:", env.action_space)

episode_success = []
all_rewards = []
episode_length_list = []

for trajectory in range(args.num_trajectories):
    state = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:
        action = random.randrange(env.action_space.n)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_length += 1

    success = 0.0
    if reward == 2.0:
        success = 1.0
    episode_success.append(success)
    episode_length_list.append(episode_length)
    all_rewards.append(episode_reward)

with open(model_dir + '/rewards.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(all_rewards)

with open(model_dir + '/episode_success.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(episode_success)

with open(model_dir + '/episode_length.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(episode_length_list)

print("Average total reward:", np.mean(all_rewards))
print("Average Success Rate:", np.mean(episode_success))
print("Average Length of Episode:", np.mean(episode_length_list))
