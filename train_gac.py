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

from imitation import WorkspaceImitation, make_env

import dmc2gym
import hydra

import matplotlib.pyplot as plt
import time
from collections import defaultdict


class WorkspaceGAC(WorkspaceImitation):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.load_demo_path

        # Load the true expert just to evaluate various metrics (not used for learning!)
        self.load_expert(cfg.load_expert_path)

        # Get the irl_reward params to save
        self.reward_params = self.agent.get_reward_params()

        # Load reward for evaluation
        if cfg.load_reward_path:
            self.load_reward(cfg.load_reward_path)

        # Make directory for saving graphics
        self.figure_root_dir = utils.make_dir(self.work_dir, 'figures')

        # For time debugging
        self.time_readout = defaultdict(lambda: [])


    def load_expert(self, load_expert_path):
        assert load_expert_path

        self.cfg.expert.params.obs_dim = self.env.observation_space.shape[0]
        self.cfg.expert.params.action_dim = self.env.action_space.shape[0]
        self.cfg.expert.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        expert = hydra.utils.instantiate(self.cfg.expert)
        checkpoint = torch.load(load_expert_path)
        expert.load_params(checkpoint['agent_params'])

        self.agent.set_expert(expert)


    def save_reward(self):
        torch.save({'reward_params': self.reward_params,
                    'step': self.step},
                   os.path.join(self.ckpt_root_dir, 'reward_step_{}.pt'.format(self.step)))


    def load_reward(self, load_path):
        print(f"Loading reward from {load_path}")
        checkpoint = torch.load(load_path)
        self.agent.load_reward_params(checkpoint['reward_params'])


    def evaluate_irl(self, actor, actor_name):
        average_episode_reward = 0
        average_episode_irl_reward = 0
        bc_losses = []

        samp_freq = 10

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            actor.reset()
            self.video_recorder.init(enabled=True)
            done = False
            episode_reward = 0
            episode_irl_reward = 0
            episode_step = 0
            reward_list = []
            while not done:
                # Act and step in environment
                with utils.eval_mode(actor):
                    action = actor.act(obs, sample=False)

                # Track BC loss if evaluating learner
                if actor_name is 'expert':
                    lea_action = self.agent.act(obs, sample=False)
                    mse_loss = np.mean((lea_action - action)**2)
                    bc_losses.append(mse_loss)
                else:
                    exp_action = self.agent.expert.act(obs, sample=False)
                    mse_loss = np.mean((exp_action - action)**2)
                    bc_losses.append(mse_loss)


                # Step in environment
                obs, reward, done, _ = self.env.step(action)

                # Track env reward
                episode_reward += reward

                # Compute IRL reward
                t_obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                t_acs = torch.FloatTensor(action).to(self.device).unsqueeze(0)
                irl_reward = self.agent.irl_reward(t_obs, t_acs).item()
                episode_irl_reward += irl_reward

                # Save video/rewards at the same interval
                if episode_step % samp_freq == 0:
                    self.video_recorder.record(self.env)
                    reward_list.append(irl_reward)

                episode_step += 1

            # Track episode level average rewards
            average_episode_reward += episode_reward
            average_episode_irl_reward += episode_irl_reward

            # Save the video
            self.video_recorder.save(f'{actor_name}_step_{self.step}_eval_{episode}.mp4')

            # Plot the irl rewards for the episode
            plt.clf()
            plt.plot(reward_list)
            fig_name = os.path.join(self.figure_root_dir, f'{actor_name}_step_{self.step}_eval_{episode}.png')
            plt.savefig(fig_name)

        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_irl_reward /= self.cfg.num_eval_episodes
#        print("Expert Avg Reward: {}".format(average_episode_reward))

        # Save out the metrics
        self.logger.log(f'eval/{actor_name}_episode_reward',
                        average_episode_reward,
                        self.step)

        self.logger.log(f'eval/{actor_name}_irl_reward',
                        average_episode_irl_reward,
                        self.step)

        if actor_name is 'expert':
            assert bc_losses
            self.logger.log(f'eval/learner_bc_loss',
                            np.mean(bc_losses),
                            self.step)
        else:
            assert bc_losses
            self.logger.log(f'eval/expert_bc_loss',
                            np.mean(bc_losses),
                            self.step)


    def run_online(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        self.agent.reset()
        print("Running Online!")


        while self.step < self.cfg.num_train_steps:

            if done:
                # Evaluate agent periodically
                if self.step > self.cfg.num_seed_steps and self.step % self.cfg.eval_frequency == 0:
                    print("Evaluating...")
                    self.evaluate_irl(self.agent, 'learner')
                    self.evaluate_irl(self.agent.expert, 'expert')
                    self.logger.dump(self.step, ty='eval')

                    # save a checkpoint
                    if (self.cfg.save_reward and self.step < self.agent.stop_reward_update):
                        self.save_reward()

                self.logger.log('train/episode', episode, self.step)
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/duration', time.time() - start_time, self.step)
                start_time = time.time()

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                # sample action for data collection
                if self.step < self.cfg.num_seed_steps:
                    action = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=True)

            # Step in Environment
            next_obs, reward, done, _ = self.env.step(action)

            # Get next action a' in (s, a, s', a')
            if self.step < self.cfg.num_seed_steps:
                next_action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    next_action = self.agent.act(next_obs, sample=True)


            # Allow infinite bootstrap
            done = float(done)
            # true if episode terminated before max timelimit was reached (should always be false for dm_ctrl)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            assert (not done_no_max)

            # Add a transition to the demonstrations. There are self.step number of transitions added.
            self.replay_buffer.add(obs,
                                   action,
                                   reward,
                                   next_obs,
                                   next_action,
                                   done,
                                   done_no_max)

            # (s', a') becomes (s, a) for next time step
            obs = next_obs
            action = next_action

            # Increment step variables
            episode_step += 1
            self.step += 1

            # Update agent with GAC Loss
            if self.step > self.cfg.num_seed_steps:
                if self.cfg.full_rl:
                    self.agent.update_full_rl(self.demonstrations,
                                              self.replay_buffer,
                                              self.logger,
                                              self.step,
                                              self.cfg.eval_frequency,
                                              self.cfg.online)
                else:
                    self.agent.update_one_step_rl(self.demonstrations,
                                                  self.replay_buffer,
                                                  self.logger,
                                                  self.step,
                                                  self.cfg)

            # Interrupt training loop to perform various operations as needed
            self.invoke_killer()


@hydra.main(config_path='config/imitate.yaml', strict=True)
def main(cfg):
    workspace = WorkspaceGAC(cfg)
    workspace.run_online()

if __name__ == '__main__':
    main()

