import jsonpickle
from gym_recording import playback

def handle_ep(observations, actions, rewards, env):
    env = jsonpickle.decode(env)
    actions = [int(x) for x in actions]
    for action in actions:
        renderer = env.render()
        print(action)
        env.step(action)
        print(observations)
        if renderer.window is None:
            break
playback.scan_recorded_traces("storage/recordings", handle_ep)
