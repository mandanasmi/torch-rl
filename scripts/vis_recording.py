from gym_recording import playback

def handle_ep(observations, actions, rewards, env, start_state):
    print(observations)
    import pdb; pdb.set_trace()

playback.scan_recorded_traces("storage/recordings", handle_ep)
