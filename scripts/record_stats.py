import jsonpickle
import argparse
import time
import sys
import gym
try:
    import gym_minigrid
except ImportError:
    pass
import utils
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording import playback
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


def keyDownCb(keyName):
    if keyName == 'BACKSPACE':
        print("restart this trajectory")
        return "restart"

    if keyName == 'ESCAPE':
        print("close the window")
        sys.exit(0)

    if keyName == 'RETURN':
        print("skip to next trajectory")
        return "return"

    else:
        print("unknown key %s" % keyName)
        return

def handle_ep(observations, actions, rewards, seed, difficulty):
    trajectory_reward = sum(rewards)
    print(trajectory_reward)

    if args.wins and trajectory_reward < 0.5:
        return
    if args.fails and trajectory_reward > 0.5:
        return
    print(difficulty)
    if args.hard and difficulty < 5:
        return
    env = gym.make("MiniGrid-GridCity-4S30-v0")
    env.seed(seed)
    utils.seed(seed)
    env.reset()

    trajectory = [int(x) for x in actions]
    actions = trajectory.copy()
    while actions:
        env.target_door.color = "red"
        action = actions.pop(0)
        renderer = env.render()
        obs, reward, done, info = env.step(action)
        if done and reward:
            text = "Win!"
        elif done and not reward:
            text = "Failure :("
        else:
            text = 'seed=%s, mission=%s, step=%s, reward=%.2f, trajectory=%s' % (str(seed), env.mission, env.step_count, reward, str(trajectory))

        renderer.window.setText(text)
        renderer.window.setKeyDownCb(keyDownCb)
        if renderer.window is None or renderer.window.key == "return":
            return
        if renderer.window.key == "restart":
            actions = trajectory.copy()
            env.grid_render.close()
            renderer.window.close()
            renderer.close()
            renderer = None
            env.seed(seed)
            utils.seed(seed)
            env.reset()
            window = renderer.window.copy()
            renderer = env.render()

parser = argparse.ArgumentParser()
parser.add_argument("--fails", action='store_true', required=False, help="shows only failing trajectories")
parser.add_argument("--wins", action='store_true', required=False, help="shows only succesful trajectories")
parser.add_argument("--hard", action='store_true', required=False, help="shows only difficult trajectories")
args = parser.parse_args()

playback.scan_recorded_traces("storage/recordings", handle_ep)
