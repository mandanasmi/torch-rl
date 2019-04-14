#!/usr/bin/env python3

import argparse
import gym
import time
import torch
import sys
import pickle
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
parser.add_argument("--difficulty", required=True,
                    type=int, help="difficulty")
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

logs = []
start_time = time.time()

print("CUDA available: {}\n".format(torch.cuda.is_available()))
for difficulty in range(1, args.difficulty + 1):
    logs_by_difficulty = {"difficulty": difficulty, "actual_difficulty": [], "target_path_length": [], "num_frames_per_episode": [], "return_per_episode": []}
    for i in range(args.episodes):
        episode_return = 0
        episode_num_frames = 0

        env = gym.make(args.env)
        env.random_seed = i
        env.seed(i)
        utils.seed(i)
        env.unwrapped.set_difficulty(difficulty)

        env = TraceRecordingWrapper(env, directory="storage/recordings")
        obs = env.reset()

        model_dir = utils.get_model_dir(args.env, args.model, args.seed)
        agent = utils.Agent(args.env, env.observation_space, model_dir, args.argmax, 1)
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)

            episode_return += reward
            episode_num_frames += 1

            if done:
                logs_by_difficulty["actual_difficulty"].append(env.unwrapped.actual_difficulty)
                logs_by_difficulty["target_path_length"].append(env.unwrapped.target_path_length)
                logs_by_difficulty["return_per_episode"].append(episode_return)
                logs_by_difficulty["num_frames_per_episode"].append(episode_num_frames)
                print("episode: " + str(i) + ", reward: " + str(reward))
    logs.append(logs_by_difficulty)

pickle.dump(logs, open("log.txt", 'wb'))
