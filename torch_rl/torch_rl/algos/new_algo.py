import numpy as np
import torch
import torch.nn.functional as F
from torch_rl.algos.base import BaseAlgo
from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv
from abc import ABC
from collections import deque
import random
import math


class DQNAlgo_new(ABC):
    """The class for the DQN"""

    def __init__(self, env, base_model, num_frames, discount=0.99, lr=7e-4,
                 adam_eps=1e-5, batch_size=256, preprocess_obss=None, capacity=10000,
                 log_interval=1000, save_interval=10000):

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

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) \
                                             * math.exp(-1. * frame_idx / epsilon_decay)

    def update_parameters(self, logger, status, model_dir):
        num_frames = status['num_frames']
        episode_reward = 0
        self.obs = self.env.reset()
        for frame_idx in range(num_frames, self.num_frames):

            preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
            epsilon = self.epsilon_by_frame(frame_idx)

            action = self.base_model.act(preprocessed_obs, epsilon)
            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.push(self.obs, action, reward, next_state, done)
            self.obs = next_state

            episode_reward += reward

            if done:
                self.obs = self.env.reset()
                self.all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss()
                self.losses.append(loss.item())

            if frame_idx % self.log_interval == 0 and frame_idx > 0:
                print("Number of Frames", frame_idx, "Rewards:", self.all_rewards[-1], "Losses:", self.losses[-1])

            # TODO: Curriculum Implementation

            if frame_idx % self.save_interval == 0:
                # TODO: Save losses and rewards.
                # self.all_rewards
                # self.losses

                # Saving model
                if torch.cuda.is_available():
                    self.base_model.cpu()
                torch.save(self.base_model, model_dir+"/model.pt")
                print("Saving Model and Logs...")
                if torch.cuda.is_available():
                    self.base_model.cuda()

    def compute_td_loss(self):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        obs = self.preprocess_obss(state, device=self.device)
        next_obs = self.preprocess_obss(next_state, device=self.device)
        action = torch.LongTensor(action).cuda()
        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(done).cuda()

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

