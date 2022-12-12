import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
#from torch.nn.utils import spectral_norm
import os
from collections import deque
import random
import math
import pdb
import gym

import base64
import signal
import functools
import io
import json
import pathlib
import pickle
import warnings
import zipfile
from typing import Any, Dict, Optional, Tuple, Union
import cloudpickle

import dmc2gym


import yaml


def load_yaml(yaml_path):
    with open(yaml_path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

@functools.singledispatch
def open_path(path: Union[str, pathlib.Path, io.BufferedIOBase], mode: str, verbose: int = 0, suffix: Optional[str] = None):
    """
    Opens a path for reading or writing with a preferred suffix and raises debug information.
    If the provided path is a derivative of io.BufferedIOBase it ensures that the file
    matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
    If the mode is write ("w", "write") it checks that the file is writable.
    If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
    it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
    If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
    points to a folder, it changes the path to path_2. If the path already exists and verbose == 2,
    it raises a warning.
    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param mode: how to open the file. "w"|"write" for writing, "r"|"read" for reading.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    if not isinstance(path, io.BufferedIOBase):
        raise TypeError("Path parameter has invalid type.", io.BufferedIOBase)
    if path.closed:
        raise ValueError("File stream is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError:
        raise ValueError("Expected mode to be either 'w' or 'r'.")
    if ("w" == mode) and not path.writable() or ("r" == mode) and not path.readable():
        e1 = "writable" if "w" == mode else "readable"
        raise ValueError(f"Expected a {e1} file.")
    return path


def save_to_pkl(path: str, obj: Any, verbose: int = 0) -> None:
    """
    Save an object to path creating the necessary folders along the way.
    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.
    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param obj: The object to save.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open(path, 'wb') as file_handler:
        pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pkl(path: str, verbose: int = 0) -> Any:
    """
    Load an object from the path. If a suffix is provided in the path, it will use that suffix.
    If the path does not exist, it will attempt to load using the .pkl suffix.
    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open(path, 'rb') as file_handler:
        return pickle.load(file_handler)


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
            self.kill_now = False
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
            self.kill_now = True


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update_params(net, target_net):
    target_net.load_state_dict(net.state_dict())

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def he_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def xavier_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def leaky_mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def tanh_mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class MultiplyLayer(nn.Module):
    def __init__(self, factor):
        super(MultiplyLayer, self).__init__()
        self.factor = factor

    def forward(self, tensors):
        return tensors * self.factor


def spectral_mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, npower=1):
    if hidden_depth == 0:
        mods = [spectral_norm(nn.Linear(input_dim, output_dim), n_power_iterations=npower)]
    else:
        mods = [spectral_norm(nn.Linear(input_dim, hidden_dim), n_power_iterations=npower), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [spectral_norm(nn.Linear(hidden_dim, hidden_dim), n_power_iterations=npower), nn.ReLU(inplace=True)]
        mods.append(spectral_norm(nn.Linear(hidden_dim, output_dim), n_power_iterations=npower))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def spectral_multiply_mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, npower=1, factor=1):
    if hidden_depth == 0:
        mods = [spectral_norm(nn.Linear(input_dim, output_dim), n_power_iterations=npower), MultiplyLayer(factor)]
    else:
        mods = [spectral_norm(nn.Linear(input_dim, hidden_dim), n_power_iterations=npower), MultiplyLayer(factor), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [spectral_norm(nn.Linear(hidden_dim, hidden_dim), n_power_iterations=npower), MultiplyLayer(factor), nn.ReLU(inplace=True)]
        mods.append(spectral_norm(nn.Linear(hidden_dim, output_dim), n_power_iterations=npower))
        mods.append(MultiplyLayer(factor))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def last_sparse_spectral_mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, npower=1):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(spectral_norm(nn.Linear(hidden_dim, output_dim), n_power_iterations=npower))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def first_sparse_spectral_mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, npower=1):
    if hidden_depth == 0:
        mods = [spectral_norm(nn.Linear(input_dim, output_dim), n_power_iterations=npower)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk



