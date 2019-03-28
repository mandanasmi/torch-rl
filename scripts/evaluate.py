#!/usr/bin/env python3

import argparse
import gym
import time
import torch
import sys
import numpy as np
from torch_rl.utils.penv import ParallelEnv
from gym_recording.wrappers import TraceRecordingWrapper

try:
    import gym_minigrid
except ImportError:
    pass
import sys
sys.path.append(".")

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")
args = parser.parse_args()

logs = {"num_frames_per_episode": [], "return_per_episode": []}

start_time = time.time()

print("CUDA available: {}\n".format(torch.cuda.is_available()))

for i in range(args.episodes):
    print("episode: " + str(i))
    episode_return = 0
    episode_num_frames = 0

    env = gym.make(args.env)
    env.random_seed = i
    env.seed(i)
    utils.seed(i)
    env = TraceRecordingWrapper(env, directory="storage/recordings")
    obs = env.reset()

    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(args.env, env.observation_space, model_dir, args.argmax, 1)
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        episode_return += reward
        episode_num_frames += 1

        if done:
            logs["return_per_episode"].append(episode_return)
            logs["num_frames_per_episode"].append(episode_num_frames)
            print(reward)
end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
print(logs)
# return_per_episode = utils.synthesize(logs["return_per_episode"])
# num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
#
# print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
#       .format(num_frames, fps, duration,
#               *return_per_episode.values(),
#               *num_frames_per_episode.values()))
#
# Print worst episodes

# n = args.worst_episodes_to_show
# if n > 0:
#     print("\n{} worst episodes:".format(n))
#
#     indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
#     for i in indexes[:n]:
#         print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
