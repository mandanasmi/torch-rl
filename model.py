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
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # nn.Conv2d(16, 32, (2, 2)),
            # nn.ReLU(),
            nn.Conv2d(16, 64, (2, 2)),
            nn.ReLU()
        )
        n = 7 #obs_space["image"][0]
        m = 7 #obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

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

        x = self.image_conv(x)
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
    def __init__(self, action_space, use_goal=False, use_gps=True, use_visible_text=True, use_image=False, env='Minigrid'):
        super().__init__()

        self.num_actions = action_space.n
        self.embedding_size = 0
        # Decide which components are enabled
        self.use_goal = use_goal
        self.use_gps = use_gps
        self.use_image = use_image
        self.use_visible_text = use_visible_text
        self.env = env
        self.embed_imgs = []
        self.embed_gps = []

        if self.use_image:
            if re.match("Hyrule-.*", self.env):
                self.image_embedding_size = 128
            else:
                self.image_embedding_size = 75
            self.embedding_size = self.image_embedding_size

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )

            self.post_conv_net = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.image_embedding_size),
                nn.ReLU()
            )

        # Define goal usages
        if self.use_goal:
            house_embedding = 16
            self.house_net = nn.Sequential(
                nn.Linear(40, house_embedding),
                nn.LeakyReLU(),
            )
            self.embedding_size += house_embedding

            street_embedding = 8
            self.street_net = nn.Sequential(
                nn.Linear(3, street_embedding),
                nn.LeakyReLU(),
            )
            self.embedding_size += street_embedding

            if self.use_visible_text:
                self.embedding_size += house_embedding*3
                self.embedding_size += street_embedding*2

        if self.use_gps:
            rel_gps_embedding = 128
            self.gps_net = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, rel_gps_embedding),
                nn.ReLU(),
            )
            self.embedding_size += rel_gps_embedding

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.net = nn.Sequential(
                nn.Linear(self.embedding_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        x = torch.tensor([])
        if self.use_image:
            x = obs.image
            if re.match("Hyrule-.*", self.env):
                x = self.image_conv(x)
                x = x.reshape(x.shape[0], -1)
                x = self.post_conv_net(x)
            else:
                x = x.reshape(x.shape[0], -1)
            self.embed_imgs.append(x.cpu().detach())

        if self.use_goal:
            embed_house = self.house_net(obs.goal["house_numbers"])
            x = torch.cat((x, embed_house), dim=1)
            embed_street = self.street_net(obs.goal["street_names"])
            x = torch.cat((x, embed_street), dim=1)

            if self.use_visible_text:
                for i in range(3):
                    embed_house = self.house_net(obs.visible_text["house_numbers"][:, i*40:(i+1)*40])
                    x = torch.cat((x, embed_house), dim=1)
                for i in range(2):
                    embed_street = self.street_net(obs.visible_text["street_names"][:, i*3:(i+1)*3])
                    x = torch.cat((x, embed_street), dim=1)

        if self.use_gps:
            embed_gps = self.gps_net(obs.rel_gps)
            self.embed_gps.append(embed_gps.cpu().detach())
            if self.use_image:
                x = torch.cat((x, embed_gps), dim=1)
            else:
                x = embed_gps
        return self.net(x)

    def act(self, obs, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                q_value = self.forward(obs)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)

        return action
