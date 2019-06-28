import matplotlib.pyplot as plt
import csv
import numpy as np
import os

dir = "storage/HyruleDQN_seed_1_denser/"

def prep_list(filename):
    with open(dir+filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    all_vals = []
    for val in your_list[0]:
        all_vals.append(float(val))

    new_list = []
    for i in range(0, len(all_vals), 100):
        new_list.append(np.mean(all_vals[i:i+100]))

    return new_list


if not os.path.isdir(dir+"plots/"):
    os.mkdir(dir+"plots/")
reward_file = "rewards.csv"
success_file = "episode_success.csv"
loss_file = "losses.csv"

fig = plt.figure()
plt.plot(prep_list(reward_file))
fig.suptitle('Sum of Rewards', fontsize=20)
plt.xlabel('timesteps (x100)', fontsize=16)
plt.ylabel('Reward', fontsize=16)
fig.savefig(dir+'plots/sum_rewards.jpg')

fig = plt.figure()
plt.plot(prep_list(success_file))
fig.suptitle('Success Rate', fontsize=20)
plt.xlabel('timesteps (/100)', fontsize=16)
plt.ylabel('Average final reward', fontsize=16)
fig.savefig(dir+'plots/success_rate.jpg')

fig = plt.figure()
plt.plot(prep_list(loss_file))
fig.suptitle('Q-Value difference', fontsize=20)
plt.xlabel('timesteps (/100)', fontsize=16)
plt.ylabel('Loss', fontsize=16)
fig.savefig(dir+'plots/losses.jpg')

# orig_obs_file = "orig_obs.npy"
# orig_obs = np.load(dir+orig_obs_file)
