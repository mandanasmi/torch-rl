import numpy as np
import torch
from torch_rl.format import default_preprocess_obss
from abc import ABC
from collections import deque
import random
import math
import json, os, csv


class DQNAlgo_new(ABC):
    """The class for the DQN"""

    def __init__(self, env, base_model, num_frames, discount=0.99, lr=7e-4, adam_eps=1e-5,
                 batch_size=256, preprocess_obss=None, capacity=10000, log_interval=100,
                 save_interval=1000, train_interval=100, record_qvals=False):

        self.env = env
        self.base_model = base_model
        self.base_model.train()
        self.discount = discount
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr, eps=adam_eps)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.batch_num = 0
        self.replay_buffer = ReplayBuffer(capacity)

        self.all_rewards = []
        self.losses = []
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.train_interval = train_interval

        self.curriculum_threshold = 0.75

        self.qvals = []
        self.record_qvals = record_qvals

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 10000
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) \
                                             * math.exp(-1. * frame_idx / epsilon_decay)

    def update_parameters(self, status, model_dir):
        num_frames = status['num_frames']
        episode_reward = 0
        self.obs = self.env.reset()

        if self.record_qvals:
            orig_obs = self.obs

        for frame_idx in range(num_frames, self.num_frames):

            preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
            epsilon = self.epsilon_by_frame(frame_idx)

            action = self.base_model.act(preprocessed_obs, epsilon)
            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.push(self.obs, action, reward, next_state, done)
            self.obs = next_state

            episode_reward += reward

            if len(self.replay_buffer) > self.batch_size and frame_idx % self.train_interval == 0:
                loss = self.compute_td_loss()
                self.losses.append(loss.item())

                if self.record_qvals:
                    self.qvals.append(self.base_model(self.preprocess_obss([orig_obs], device=self.device)))

            if done:
                self.obs = self.env.reset()
                self.all_rewards.append(episode_reward)
                episode_reward = 0

                if len(self.all_rewards) % self.log_interval == 0 and len(self.all_rewards) > 0:
                    print("Number of Trajectories:", len(self.all_rewards),
                          "| Number of Frames:", frame_idx,
                          "| Rewards:", np.mean(self.all_rewards[-100:]),
                          "| Losses:", np.mean(self.losses[-100:]))
                    status["num_frames"] = frame_idx

                    # Curriculum learning
                    if np.mean(self.all_rewards[-100:]) > self.curriculum_threshold:
                        print("empirical_win_rate: " + str(np.mean(self.all_rewards[-100:])))
                        print("Increasing Difficulty by 1!")
                        status["difficulty"] += 1
                        self.env.set_difficulty(status["difficulty"])
                        print(status["difficulty"])

                if len(self.all_rewards) % self.save_interval == 0 and len(self.all_rewards) > 0:
                    # Save losses and rewards.
                    with open(model_dir+'/losses.csv', 'w') as writeFile:
                        writer = csv.writer(writeFile)
                        writer.writerow(self.losses)
                    with open(model_dir+'/rewards.csv', 'w') as writeFile:
                        writer = csv.writer(writeFile)
                        writer.writerow(self.all_rewards)

                    # Save status
                    path = os.path.join(model_dir, "status.json")
                    with open(path, "w") as file:
                        json.dump(status, file)

                    # Saving model
                    if torch.cuda.is_available():
                        self.base_model.cpu()
                    torch.save(self.base_model, model_dir+"/model.pt")
                    print("Done saving model and logs...")
                    if torch.cuda.is_available():
                        self.base_model.cuda()

                    # TODO: Save replay buffer for training continuation

                    # Save q values if debug mode
                    if self.record_qvals:
                        with open(model_dir + '/q_vals.csv', 'w') as writeFile:
                            writer = csv.writer(writeFile)
                            writer.writerow(self.qvals)

    def compute_td_loss(self):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        obs = self.preprocess_obss(state, device=self.device)
        next_obs = self.preprocess_obss(next_state, device=self.device)
        action = torch.LongTensor(action).to(device=self.device)
        reward = torch.FloatTensor(reward).to(device=self.device)
        done = torch.FloatTensor(done).to(device=self.device)

        q_values = self.base_model(obs)
        next_q_values = self.base_model(next_obs)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.discount * next_q_value * (1 - done)

        loss = (q_value - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
