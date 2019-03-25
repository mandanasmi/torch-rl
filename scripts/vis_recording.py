import jsonpickle
import argparse
import time
import sys
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

def handle_ep(observations, actions, rewards, encoded_env):
    trajectory_reward = sum(rewards)
    if args.wins and trajectory_reward < 0.5:
        return
    if args.fails and trajectory_reward > 0.5:
        return

    env = jsonpickle.decode(encoded_env)
    trajectory = [int(x) for x in actions]
    actions = trajectory.copy()
    while actions:
        action = actions.pop()
        renderer = env.render()
        obs, reward, done, info = env.step(action)
        text = 'mission=%s, step=%s, reward=%.2f' % (env.mission, env.step_count, reward)
        renderer.window.setText(text)
        renderer.window.setKeyDownCb(keyDownCb)
        if renderer.window is None:
            break
        if renderer.window.key == "return":
            return
        if renderer.window.key == "restart":
            actions = trajectory
            env.grid_render.close()
            renderer.window.close()
            renderer.close()
            renderer = None
            env = jsonpickle.decode(encoded_env)

parser = argparse.ArgumentParser()
parser.add_argument("--fails", action='store_true', required=False, help="shows only failing trajectories")
parser.add_argument("--wins", action='store_true', required=False, help="shows only failing trajectories")
args = parser.parse_args()
playback.scan_recorded_traces("storage/recordings", handle_ep)
