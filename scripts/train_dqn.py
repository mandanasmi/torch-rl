#!/usr/bin/env python3

import argparse
import gym
import time
import datetime
import torch
import torch_rl
import sys
import numpy as np
import json

try:
    import gym_minigrid
except ImportError:
    pass
import sys
sys.path.append(".")

import utils
from model import DQNModel

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--difficulty", required=False, default=1, type=int,
                    help="difficulty of the environment")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--replay_capacity", required=False, default=10000, type=int,
                    help="Capacity of Replay Buffer")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--exp", type=int, default=0,
                    help="experiment id number")
parser.add_argument("--text", action="store_true", default=True,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=32,
                    help="batch size for PPO (default: 256)")
args = parser.parse_args()

# Get model directory
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_seed{}_{}".format(args.env, 'dqn', args.seed, suffix)
model_name = args.model or default_model_name
model_dir = utils.get_model_dir(args.env, model_name, args.exp, args.seed)

# Init environment
env = gym.make(args.env)
if "Street" not in args.env:
    env.unwrapped.set_difficulty(args.difficulty, weighted=False)
env.seed(args.seed)

# Get obs space and preprocess function
obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, env.observation_space, model_dir)

# Init Model
print(obs_space)
args.text = False
base_model = DQNModel(obs_space, env.action_space, args.text)
if torch.cuda.is_available():
    base_model.cuda()

# Init Algorithm
algo = torch_rl.DQNAlgo_new(env, base_model, args.frames, args.discount, args.lr, args.optim_eps,
                            args.batch_size, preprocess_obss)

# Train Algoritm
algo.update_parameters()


