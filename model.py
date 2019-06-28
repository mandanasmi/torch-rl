import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym
import random
import re

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        # self.image_conv = nn.Sequential(
        #     nn.Conv2d(3, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     # nn.Conv2d(16, 32, (2, 2)),
        #     # nn.ReLU(),
        #     nn.Conv2d(16, 64, (2, 2)),
        #     nn.ReLU()
        # )
        n = 7 #obs_space["image"][0]
        m = 7 #obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*75

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        #x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class DQNModel(nn.Module, torch_rl.RecurrentACModel):
    def __init__(self, action_space, use_goal=True, use_gps=False, use_visible_text=True, env='Minigrid'):
        super().__init__()

        self.num_actions = action_space.n
        # Decide which components are enabled
        self.use_goal = use_goal
        self.use_gps = use_gps
        self.use_visible_text = use_visible_text
        self.env = env

        if re.match("Hyrule-.*", self.env):
            self.image_embedding_size = 512  # Obtained by calculating output on below conv with input 84x84x3
        else:
            self.image_embedding_size = 75

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU()
        )

        self.embedding_size = self.image_embedding_size

        # Define goal usages
        if self.use_goal:
            goal_embedding = 64
            self.goal_net = nn.Sequential(
                nn.Linear(43, goal_embedding),
                nn.LeakyReLU(),
            )
            self.embedding_size += goal_embedding

        if self.use_gps:
            rel_gps_embedding = 8
            self.gps_net = nn.Sequential(
                nn.Linear(2, rel_gps_embedding),
                nn.LeakyReLU(),
            )
            self.embedding_size += rel_gps_embedding

        if self.use_visible_text:
            vistext_house_embedding = 32
            self.vistext_house_net = nn.Sequential(
                nn.Linear(120, vistext_house_embedding),
                nn.LeakyReLU(),
            )
            self.embedding_size += vistext_house_embedding

            vistext_street_embedding = 16
            self.vistext_street_net = nn.Sequential(
                nn.Linear(6, vistext_street_embedding),
                nn.LeakyReLU(),
            )
            self.embedding_size += vistext_street_embedding

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.net = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_actions)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        x = obs.image
        if re.match("Hyrule-.*", self.env):
            # x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)
        else:
            x = x.reshape(x.shape[0], -1)

        if self.use_goal:
            embed_goal = self.goal_net(obs.goal)
            x = torch.cat((x, embed_goal), dim=1)

        if self.use_gps:
            embed_gps = self.gps_net(obs.rel_gps)
            x = torch.cat((x, embed_gps), dim=1)

        if self.use_visible_text:
            embed_house = self.vistext_house_net(obs.visible_text["house_numbers"])
            x = torch.cat((x, embed_house), dim=1)
            embed_street = self.vistext_street_net(obs.visible_text["street_names"])
            x = torch.cat((x, embed_street), dim=1)

        return self.net(x)

    def act(self, obs, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                q_value = self.forward(obs)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)

        return action
