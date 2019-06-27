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
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-3,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=64,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--debug", action="store_true", default=False,
                    help="Records Q values during training")
args = parser.parse_args()

# Get model directory
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_seed{}_{}".format(args.env, 'dqn', args.seed, suffix)
model_dir = "storage/" + ((args.model+"_seed_"+str(args.seed)) or default_model_name)
utils.create_folders_if_necessary(model_dir)

# Store json args
logger = utils.get_logger(model_dir)
with open(model_dir + '/args.json', 'w') as outfile:
    json.dump(vars(args), outfile)

# Set seed for all randomness sources
utils.seed(args.seed)

# Load training status
try:
    status = utils.load_status(model_dir)
except OSError:
    status = {"num_frames": 0, "difficulty": args.difficulty}
    utils.save_status(status, model_dir)
print("Status: ", status)

# Init environment
env = gym.make(args.env)
if "Street" not in args.env:
    env.unwrapped.set_difficulty(status["difficulty"], weighted=False)
env.seed(args.seed)

# Get obs space and preprocess function
obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, env.observation_space, model_dir)

# Load model
try:
    base_model = utils.load_model(model_dir)
    print("Model successfully loaded\n")
except OSError:
    base_model = DQNModel(env.action_space, args.text, env=args.env)
    print("Model successfully created\n")

if torch.cuda.is_available():
    base_model.cuda()
print("CUDA available: {}\n".format(torch.cuda.is_available()))

# Init Algorithm
algo = torch_rl.DQNAlgo_new(env, base_model, args.frames, args.discount, args.lr, args.optim_eps, args.batch_size,
                            preprocess_obss, record_qvals=args.debug)

# Train Algoritm
algo.update_parameters(status, model_dir)
