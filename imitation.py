#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import pdb

from video import VideoRecorder
from logger import Logger
import utils

from demonstrations import Demonstrations

import dmc2gym
import hydra


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

# Base Class for Imitation Learning Algorithms
class WorkspaceImitation(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        print(f"env: {cfg.env} | obs_dim: {cfg.agent.params.obs_dim}, acs_dim: {cfg.agent.params.action_dim}")
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            height=64,
            width=64)

        self.step = 0


        # NEW STUFF
        self.agent_params = self.agent.get_save_params()
        self.killer = utils.GracefulKiller()

        self.ckpt_root_dir = utils.make_dir(self.work_dir, 'checkpoints')
        self.demo_root_dir = utils.make_dir(self.work_dir, 'demonstrations')

        # Check that the load path for demos is being specified
        if cfg.load_demo_path:
            print("Loading Demonstrations...")
            self.load_demonstrations(cfg.load_demo_path)
        else:
            # Only when sampling demonstrations
            print("Sampling Demonstrations")
            self.demonstrations = Demonstrations(self.env.observation_space.shape,
                                                 self.env.action_space.shape,
                                                 int(cfg.num_transitions),
                                                 self.device)

        # For online IL
        if cfg.load_replay_path:
            print("Loading Replay Buffer...")
            self.load_replay_buffer(cfg.load_replay_path)
        else:
            print("Creating New Replay Buffer")
            self.replay_buffer = Demonstrations(self.env.observation_space.shape,
                                                self.env.action_space.shape,
                                                int(cfg.replay_buffer_capacity),
                                                self.device)

        # Load a pretrained agent
        if cfg.load_ckpt_path:
            self.load_checkpoint(cfg.load_ckpt_path)


    def invoke_killer(self):
        # Manually save checkpoint or kill training
        if self.killer.kill_now:
            sample_demo = input('Save Demonstrations (y/[n])?')
            if sample_demo == 'y':
                self.save_demonstrations()

            save_option = input('Save checkpoint (y/[n])?')
            if save_option == 'y':
                self.save_checkpoint()

            kill_option = input('Kill session (y/[n])?')
            if kill_option == 'y':
                exit(1)
            else:
                self.killer.kill_now = False

    def save_demonstrations(self):
        assert self.demonstrations is not None, "The demonstrations have not been defined"
        if self.cfg.noisy:
            utils.save_to_pkl(os.path.join(self.demo_root_dir, f'demo_{self.cfg.num_transitions}_noisy_seed_{self.cfg.seed}.pickle'),
                              self.demonstrations,
                              verbose=0)
        else:
            utils.save_to_pkl(os.path.join(self.demo_root_dir, 'demo_{}.pickle'.format(self.cfg.num_transitions)),
                              self.demonstrations,
                              verbose=0)

    def load_demonstrations(self, load_path):
        self.demonstrations = utils.load_from_pkl(load_path, verbose=0)
        if self.cfg.num_transitions:
            print(f"Sub-sampling {self.cfg.num_transitions} demos")
            self.demonstrations.get_subset(self.cfg.num_transitions)
        else:
            print("Using full demonstration set")
        assert isinstance(self.demonstrations, Demonstrations)

    def load_replay_buffer(self, load_path):
        self.replay_buffer = utils.load_from_pkl(load_path, verbose=0)
        if self.cfg.num_transitions:
            print(f"Sub-sampling {self.cfg.num_transitions} replay")
            self.replay_buffer.get_subset(self.cfg.num_transitions)
        else:
            print("Using full replay buffer")
        assert isinstance(self.replay_buffer, Demonstrations)

    def save_checkpoint(self):
        torch.save({'agent_params': self.agent_params,
                    'step': self.step},
                   os.path.join(self.ckpt_root_dir, 'step_{}.pt'.format(self.step)))

    def load_checkpoint(self, load_path):
        print(f"Loading checkpoint from {load_path}")
        checkpoint = torch.load(load_path)
        self.agent.load_params(checkpoint['agent_params'])
        self.step = checkpoint['step']

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step, ty='eval')

    def run(self):
        raise NotImplementedError



