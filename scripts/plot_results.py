import comet_ml
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import utils
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, required=True,
                    help="random seed (default: 1)")
parser.add_argument("--dense-reward", action="store_true", default=False,
                    help="Use dense reward during training.")
args = parser.parse_args()

# Get model directory
model_dir = "storage/" + args.model +"_seed_"+str(args.seed)
if args.dense_reward: model_dir += "_denser"
utils.create_folders_if_necessary(model_dir)

def prep_list(filename):
    with open(model_dir+filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    all_vals = []
    for val in your_list[0]:
        all_vals.append(float(val))

    new_list = []
    for i in range(0, len(all_vals), 100):
        new_list.append(np.mean(all_vals[i:i+100]))

    return new_list


if not os.path.isdir(model_dir+"/plots/"):
    os.mkdir(model_dir+"/plots/")
reward_file = "/rewards.csv"
success_file = "/episode_success.csv"
loss_file = "/losses.csv"

fig = plt.figure()
plt.plot(prep_list(reward_file))
fig.suptitle('Sum of Rewards', fontsize=20)
plt.xlabel('trajectories (x100)', fontsize=16)
plt.ylabel('Reward', fontsize=16)
fig.savefig(model_dir+'/plots/sum_rewards.jpg')

fig = plt.figure()
plt.plot(prep_list(success_file))
fig.suptitle('Success Rate', fontsize=20)
plt.xlabel('trajectories (x100)', fontsize=16)
plt.ylabel('Average final reward', fontsize=16)
fig.savefig(model_dir+'/plots/success_rate.jpg')

fig = plt.figure()
plt.plot(prep_list(loss_file))
fig.suptitle('Q-Value difference', fontsize=20)
plt.xlabel('trajectories (x100)', fontsize=16)
plt.ylabel('Loss', fontsize=16)
fig.savefig(model_dir+'/plots/losses.jpg')

# orig_obs_file = "orig_obs.npy"
# orig_obs = np.load(dir+orig_obs_file)
