import argparse
import pygame

import gym
import time
import torch

import utils

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--difficulty", required=False, default=1, type=int,
                    help="difficulty of the environment")
parser.add_argument("--model", required=True,
                    help="name of the model.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
args = parser.parse_args()

# Get model directory
model_dir = "storage/" + args.model + "_seed_"+str(args.seed)
utils.create_folders_if_necessary(model_dir)

# Set seed for all randomness sources
utils.seed(args.seed)

# Load training status
status = utils.load_status(model_dir)
print("Status: ", status)

# Init environment
env = gym.make(args.env)
if "Street" not in args.env:
    env.unwrapped.set_difficulty(status["difficulty"], weighted=False)
env.seed(args.seed)

# Get obs space and preprocess function
obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, env.observation_space, model_dir)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
base_model = utils.load_model(model_dir).to(device)
print("Model successfully loaded\n")

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

env.reset()
rendered = env.render(mode='rgb_array')
video_size = [rendered.shape[1], rendered.shape[0]]

running = True
env_done = True
transpose = True
fps = 10

pygame.font.init()
screen = pygame.display.set_mode(video_size)
clock = pygame.time.Clock()
pygame.display.set_caption('NAVI')

f = 0
start = time.time()
while running:
    if time.time() - start > 1:
        start = time.time()
        f = 0
    f += 1
    if env_done:
        env_done = False
        obs = env.reset()
    else:
        preprocessed_obs = preprocess_obss([obs], device=device)
        action = base_model.act(preprocessed_obs, epsilon=0.0)
        print("Action:", action)
        obs, rew, env_done, info = env.step(action)
        print("Reward:", rew)
        if env_done:
            print("Final Reward:", rew)

    if obs is not None:
        rendered = env.render(mode='rgb_array')
        display_arr(screen, rendered, transpose=transpose, video_size=video_size)

    # process pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(fps)
pygame.quit()
