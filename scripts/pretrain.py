import argparse

import gym
import sys
sys.path.append(".")
import utils
import torch
import networkx as nx

try:
    import gym_minigrid
except ImportError:
    pass

from model import ACModel, DQNModel
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--difficulty", required=False, default=1, type=int,
                    help="difficulty of the environment")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)\nIf > 1, a LSTM is added to the model to have memory")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--exp", type=int, default=0,
                    help="experiment id number")

args = parser.parse_args()

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(128, 128, bias=True)
        self.lin2 = torch.nn.Linear(128, 64, bias=True)
        self.lin3 = torch.nn.Linear(64, 1, bias=True)

    def forward(self, xb):
        x = xb.view(-1,128)
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.relu(self.lin2(x))
        return self.lin3(x)

model_dir = f"/home/martin/code/torch-rl/storage/MiniGrid-GridCity-4S30-v0_ppo_seed1_19-12-16-12-06-25/{args.env}/0/1/"
model = utils.load_model(model_dir)
env = gym.make(args.env)

obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, env.observation_space, model_dir)
optimizer = torch.optim.Adam(model.text_rnn.parameters(), lr=1e-2)
mlp = MLP()
loss_fn = torch.nn.L1Loss()
ob1 = env.reset()
ob2 = ob1.copy()
for epoch in range(100):
    grid, g = env.make_grid()
    doors = env.get_doors()
    for door1 in doors:
        id1 = door1.pos[0] + door1.pos[1] * env.grid_size
        mission1 = 'go to street %s door number %s' % (str(door1.street.street_name), str(door1.rel_address))
        for door2 in doors:
            id2 = door2.pos[0] + door2.pos[1] * env.grid_size
            mission2 = 'go to street %s door number %s' % (str(door2.street.street_name), str(door2.rel_address))
            try:
                y = torch.FloatTensor([len(nx.shortest_path(g, id1, id2)) - 1])
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
            ob1['mission'] = mission1
            ob2['mission'] = mission2

            obs1 = preprocess_obss([ob1])
            obs2 = preprocess_obss([ob2])
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            _, hidden1 = model.text_rnn(model.word_embedding(obs1.text))
            _, hidden2 = model.text_rnn(model.word_embedding(obs2.text))
            y_hat = mlp(hidden1)

            loss = loss_fn(y_hat, y)

            print(f"loss: {loss}")
            loss.backward()
            optimizer.step()

params = [x for x  in model.text_rnn.parameters()]
print(params[0])