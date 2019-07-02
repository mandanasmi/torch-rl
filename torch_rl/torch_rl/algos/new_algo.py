import numpy as np
from comet_ml import Experiment
import torch
from torch_rl.format import default_preprocess_obss
from abc import ABC
from collections import deque
import random
from matplotlib import pyplot as plt
import math
import json, os, csv
import torch.nn.functional as F

hyper_params = {
    "learning_rate": 0.01
}

experiment = Experiment("UcVgpp0wPaprHG4w8MFVMgq7j", project_name="navi-corl-2019")
experiment.log_parameters(hyper_params)


class DQNAlgo_new(ABC):
    """The class for the DQN"""

    def __init__(self, env, base_model, target_net, num_frames, discount=0.99, lr=0.005, adam_eps=1e-8,
                 batch_size=128, preprocess_obss=None, capacity=10000, log_interval=100,
                 save_interval=1000, train_interval=500, record_qvals=False, target_update=10):

        self.env = env
        self.base_model = base_model
        self.target_model = target_net
        self.base_model.train()
        self.discount = discount
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr) #, eps=adam_eps)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.batch_num = 0
        self.replay_buffer = ReplayBuffer(capacity)

        self.episode_success = []
        self.all_rewards = []
        self.losses = []
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.train_interval = train_interval
        self.target_update = target_update

        self.curriculum_threshold = 0.5

        self.qvals = []
        self.record_qvals = record_qvals

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 100000
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) \
                                             * math.exp(-1. * frame_idx / epsilon_decay)

    def update_parameters(self, status, model_dir):
        num_frames = status['num_frames']
        episode_reward = 0
        episode_length = 0
        episode_length_list = []
        self.obs = self.env.reset()

        if self.record_qvals:
            orig_obs = self.obs
            experiment.log_metric("good_action_for_qvals", self.env.shortest_path_length()[0])
            np.save(model_dir+"/orig_obs.npy", orig_obs)
            self.qvals.append(self.base_model(self.preprocess_obss([orig_obs], device=self.device)))


        for frame_idx in range(num_frames, self.num_frames):
            with experiment.train():

                preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
                epsilon = self.epsilon_by_frame(frame_idx)
                experiment.log_metric("epsilon", epsilon, step=frame_idx)

                action = self.base_model.act(preprocessed_obs, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.push(self.obs, action, reward, next_state, done)
                self.obs = next_state

                episode_reward += reward
                episode_length += 1

                if len(self.replay_buffer) > self.batch_size and frame_idx % self.train_interval == 0:
                    loss = self.compute_td_loss()
                    self.losses.append(loss.item())
                    experiment.log_metric("loss", loss.item(), step=frame_idx)

                    if self.record_qvals:
                        with torch.no_grad():
                            qvals = self.base_model(self.preprocess_obss([orig_obs], device=self.device)).cpu().numpy()
                            self.qvals.append(qvals)
                            qval_dict = {"BIG_LEFT": qvals[0][0], "SMALL_LEFT": qvals[0][1], "FORWARD": qvals[0][2], "SMALL_RIGHT": qvals[0][3], "BIG_RIGHT": qvals[0][4], }
                            experiment.log_metrics(qval_dict, step=frame_idx)
                if done:
                    success = 0.0
                    if reward == 2.0:
                        success = 1.0
                    self.episode_success.append(success)
                    experiment.log_metric("episode_success_rate", np.sum(self.episode_success)/len(self.episode_success))
                    experiment.log_metric("num_episodes_finished", len(self.episode_success))
                    experiment.log_metric("episode_length", episode_length, step=frame_idx)

                    episode_length_list.append(episode_length)
                    episode_length = 0

                    self.obs = self.env.reset()
                    self.all_rewards.append(episode_reward)
                    experiment.log_metric("episode_reward", episode_reward, step=frame_idx)
                    episode_reward = 0

                    if len(self.all_rewards) % self.target_update == 0:
                        self.target_model.load_state_dict(self.base_model.state_dict())

                    if len(self.all_rewards) % self.log_interval == 0 and len(self.all_rewards) > 0:
                        print("Number of Trajectories:", len(self.all_rewards),
                              "| Number of Frames:", frame_idx,
                              "| Success Rate:", np.mean(self.episode_success[-100:]),
                              "| Average Episode Reward:", np.mean(self.all_rewards[-100:]),
                              "| Losses:", np.mean(self.losses[-100:]),
                              "| Epsilon:", epsilon,
                              "| Length of Episode:", np.mean(episode_length_list[-100:]))
                        status["num_frames"] = frame_idx

                        # # Curriculum learning
                        # if np.mean(self.episode_success[-100:]) >= self.curriculum_threshold:
                        #     print("empirical_win_rate: " + str(np.mean(self.episode_success[-100:])))
                        #     print("Increasing Difficulty by 1!")
                        #     status["difficulty"] += 1
                        #     self.env.set_difficulty(status["difficulty"])
                        #     print("New Difficulty:", status["difficulty"])
                    if len(self.all_rewards) % self.save_interval == 0 and len(self.all_rewards) > 0:
                        # Save losses and rewards.
                        with open(model_dir+'/losses.csv', 'w') as writeFile:
                            writer = csv.writer(writeFile)
                            writer.writerow(self.losses)
                        with open(model_dir+'/rewards.csv', 'w') as writeFile:
                            writer = csv.writer(writeFile)
                            writer.writerow(self.all_rewards)
                        with open(model_dir+'/episode_success.csv', 'w') as writeFile:
                            writer = csv.writer(writeFile)
                            writer.writerow(self.episode_success)

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

                        # Save q values if debug mode
                        if self.record_qvals:
                            with open(model_dir + '/q_vals.csv', 'w') as writeFile:
                                writer = csv.writer(writeFile)
                                writer.writerow(self.qvals)

                    if len(self.all_rewards) % self.target_update == 0:
                        self.base_model.embed_imgs = []
                        self.base_model.embed_gps = []
                        with experiment.test():
                            obs = self.env.reset()
                            spl = self.env.shortest_path_length()
                            for action in spl:
                                obs = self.preprocess_obss([obs], device=self.device)
                                self.base_model.act(obs, 0)
                                obs, reward, done, _ = self.env.step(action)
                            self.process_embeddings()


    def process_embeddings(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if self.base_model.embed_imgs:
            img_means = [img.mean().item() for img in self.base_model.embed_imgs]
            img_medians = [img.median().item() for img in self.base_model.embed_imgs]
            ax.plot(img_means, label="img_means")
            ax.plot(img_medians, label="img_medians")

        gps_means = [gps.mean().item() for gps in self.base_model.embed_gps]
        gps_medians = [gps.median().item() for gps in self.base_model.embed_gps]
        ax.plot(gps_means, label="gps_means")
        ax.plot(gps_medians, label="gps_medians")
        plt.legend()
        plt.savefig("storage/figs/embedding_means_" + str(len(self.episode_success)))
        plt.close()

    def compute_td_loss(self):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        obs = self.preprocess_obss(state, device=self.device)
        next_obs = self.preprocess_obss(next_state, device=self.device)
        action = torch.LongTensor(action).to(device=self.device)
        reward = torch.FloatTensor(reward).to(device=self.device)
        done = torch.FloatTensor(done).to(device=self.device)

        q_values = self.base_model(obs)
        next_q_values = self.target_model(next_obs).detach()

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + (self.discount * next_q_value * (1 - done))

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_value, expected_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.base_model.parameters():
            param.grad.data.clamp_(-1, 1)
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
