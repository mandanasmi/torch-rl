import random
import os
import numpy
import torch
import collections

def get_storage_dir():
    if "TORCH_RL_STORAGE" in os.environ:
        return os.environ["TORCH_RL_STORAGE"]
    return "storage"

def get_model_dir(model_name, env_name, exp, seed):
    return os.path.join(get_storage_dir(), env_name, model_name, str(exp), str(seed))

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d
