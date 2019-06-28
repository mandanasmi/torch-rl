import os
import json
import numpy as np
import re
import torch
import torch_rl
import gym
from torchvision import transforms
from PIL import Image

import utils

def get_obss_preprocessor(env_id, obs_space, model_dir):
    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "text": 500} #100

        vocab = Vocabulary(model_dir, obs_space["text"])
        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_matrix([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })
        preprocess_obss.vocab = vocab

    # Part for Hyrule env
    elif re.match("Hyrule-.*", env_id):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_natural_images([obs["image"] for obs in obss], device=device),
                "goal": preprocess_matrix([obs["mission"] for obs in obss], device=device),
                "rel_gps": preprocess_matrix([obs["rel_gps"] for obs in obss], device=device),
                "visible_text": preprocess_visible_text([obs["visible_text"] for obs in obss], device=device)
            })

    # Check if the obs_space is of type Box([X, Y, 3])
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3 and obs_space.shape[2] == 3:
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_matrix(obss, device=device)
            })
    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss

def preprocess_matrix(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_visible_text(visible_text_dict, device=None):
    house_numbers = []
    street_names = []
    for idx in range(len(visible_text_dict)):
        house_numbers.append(visible_text_dict[idx]["house_numbers"])
        street_names.append(visible_text_dict[idx]["street_names"])
    house_numbers = np.array(house_numbers)
    street_names = np.array(street_names)
    return {"house_numbers": torch.tensor(house_numbers, device=device, dtype=torch.float),
            "street_names": torch.tensor(street_names, device=device, dtype=torch.float)}

def preprocess_natural_images(images, device=None):
    images = np.array(images, dtype=np.uint8)
    new_images = torch.tensor(images, device=device, dtype=torch.float)
    new_images = torch.transpose(torch.transpose(new_images, 1, 3), 2, 3)
    transform = transforms.Compose([transforms.Normalize(mean=[0.463, 0.488, 0.53], std=[0.5, 0.5, 0.5])])
    # std=[0.0290, 0.0226, 0.0323]
    # TODO: Improve effeciency of this function
    # TODO: add random crop to transforms
    for idx, image in enumerate(images):
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image.float().div(255)
        new_images[idx] = transform(image)
    return new_images.to(device)

def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("[+-]*[a-z0-9]+", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)

class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, model_dir, max_size):
        self.path = utils.get_vocab_path(model_dir)
        self.max_size = max_size
        self.vocab = {}
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self):
        utils.create_folders_if_necessary(self.path)
        json.dump(self.vocab, open(self.path, "w"))