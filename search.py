import os
import json
import numpy as np
import subprocess
import argparse

def lognuniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--param-file", required=True, help="which param file to use (json)")
args = parser.parse_args()
params = json.load(open(args.param_file, "r"))
print(params)
np.random.seed(0)

for exp in range(params["num_experiments"]):

    # extract params to vary
    search_params = params["search_params"]
    rand_params = {}
    for key, val in search_params.items():
        if val.get('vals'):
            rand_params[key] = np.random.choice(val.get('vals'))
        else:
            rand_params[key] = eval(val['type'])(lognuniform(val['low'], val['high']))

    for trial in range(params["num_trials"]):
        seed = trial
        cmd = ['python', 'scripts/train.py']
        cmd.append('--exp=' + str(exp))
        cmd.append('--seed=' + str(seed))
        cmd.append('--env=' + params['env'])
        cmd.append('--algo=' + params['algo'])
        cmd.append('--model=' + params['model'])
        cmd.append('--frames=' + str(params['frames']))
        cmd.append('--save-interval=' + str(params['save-interval']))
        cmd.append('--text')
        for key, val in rand_params.items():
            cmd.append("--" + key + "=" + str(val))
        print(cmd)
        print(subprocess.check_output(cmd))
