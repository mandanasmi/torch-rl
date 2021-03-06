#!/usr/bin/env python3

import argparse
import gym
import time

try:
    import gym_minigrid
except ImportError:
    pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--difficulty", type=int, default=1,
                    help="difficulty level")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
if "Street" not in args.env:
    env.set_difficulty(args.difficulty, shaped_difficulty=False)
for _ in range(args.shift):
    env.reset()

# Define agent

model_dir = utils.get_model_dir(args.env, args.model, args.seed)
agent = utils.Agent(args.env, env.observation_space, model_dir, args.argmax)

# Run the agent

done = True

while True:
    if done:
        obs = env.reset()
    #env.target_door.color = "red"
    env.draw_address = True

    time.sleep(args.pause)
    renderer = env.render()

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    if done and reward:
        text = "WINNNNNNN!!!"
    elif done and not reward:
        text = "lose :("
    else:
        text = 'mission=%s, step=%s, reward=%.2f' % (env.mission, env.step_count, reward)
    renderer.window.setText(text)

    agent.analyze_feedback(reward, done)

    if renderer.window is None:
        break
