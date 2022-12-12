import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchreparam import ReparamModule
import math
import time
from collections import defaultdict

from agent import Agent
import utils
import pdb
import copy

import hydra


class GACAgent(Agent):
    """Gradient Actor-Critic Algorithm."""
    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_range,
                 device,
                 critic_cfg,
                 actor_cfg,
                 irl_reward_cfg,
                 irl_grad_cfg,
                 discount,
                 init_temperature,
                 alpha_lr,
                 alpha_betas,
                 actor_lr,
                 actor_betas,
                 actor_update_frequency,
                 critic_lr,
                 critic_betas,
                 critic_tau,
                 critic_target_update_frequency,
                 critic_target_hard_update_frequency,
                 critic_reg_weight,
                 cql_alpha,
                 irl_reward_lr,
                 irl_reward_betas,
                 irl_reward_noise,
                 irl_reward_reg_weight,
                 irl_reward_update_frequency,
                 irl_reward_horizon,
                 irl_grad_lr,
                 irl_grad_betas,
                 irl_grad_tau,
                 irl_grad_target_update_frequency,
                 batch_size,
                 learnable_temperature):

        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount

        self.critic_tau = critic_tau
        self.irl_grad_tau = irl_grad_tau

        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.critic_target_hard_update_frequency = critic_target_hard_update_frequency
        self.irl_grad_target_update_frequency = irl_grad_target_update_frequency

        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_reg_weight = critic_reg_weight
        self.cql_alpha = cql_alpha

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # Learned reward
        self.irl_reward = hydra.utils.instantiate(irl_reward_cfg).to(self.device)
        self.irl_reward_noise = irl_reward_noise
        self.irl_reward_reg_weight = irl_reward_reg_weight
        self.irl_reward_update_frequency = irl_reward_update_frequency
        self.irl_reward_horizon = irl_reward_horizon
        self.discount_tensor = None
        self.irl_reward_cfg = irl_reward_cfg

        # Copy network for gradients
        self.update_reparam_mod(init=True)

        # Reward gradient, output dim is the number of param_dim of the reward func
        irl_grad_cfg.params.output_dim = self.irl_reward.get_param_dim()
        self.irl_grad = hydra.utils.instantiate(irl_grad_cfg).to(self.device)
        self.irl_grad_target = hydra.utils.instantiate(irl_grad_cfg).to(self.device)
        self.irl_grad_target.load_state_dict(self.irl_grad.state_dict())
        self.irl_grad_step = 0


        # idx arrays
        self.grad_idx_init = False
        self.reward_idx_init = False


        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.irl_reward_optimizer = torch.optim.Adam(self.irl_reward.parameters(),
                                                      lr=irl_reward_lr,
                                                      betas=irl_reward_betas)

        self.irl_grad_optimizer = torch.optim.Adam(self.irl_grad.parameters(),
                                                      lr=irl_grad_lr,
                                                      betas=irl_grad_betas)


        self.time_readout = defaultdict(lambda: [])

        self.train()
        self.critic_target.train()
        self.irl_grad_target.train()


    def get_save_params(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'log_alpha': self.log_alpha,
                'actor_op': self.actor_optimizer.state_dict(),
                'critic_op': self.critic_optimizer.state_dict(),
                'log_alpha_op': self.log_alpha_optimizer.state_dict()
                }

    def load_params(self, param_dict):
        self.actor.load_state_dict(param_dict['actor'])
        self.critic.load_state_dict(param_dict['critic'])
        self.critic_target.load_state_dict(param_dict['critic_target'])
        self.log_alpha = param_dict['log_alpha']
        self.actor_optimizer.load_state_dict(param_dict['actor_op'])
        self.critic_optimizer.load_state_dict(param_dict['critic_op'])
        self.log_alpha_optimizer.load_state_dict(param_dict['log_alpha_op'])


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.irl_reward.train(training)
        self.irl_grad.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def set_expert(self, expert):
        self.expert = expert
        self.expert.train(False)
        for param in self.expert.actor.parameters():
            param.requires_grad = False

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def clone_irl_reward(self, obs, acs, reward, logger, step, train=False):
        clone_loss = F.mse_loss(self.irl_reward(obs, acs), reward)
        logger.log('train_irl_reward/clone_loss', clone_loss, step)

        if train:
            self.irl_reward_optimizer.zero_grad()
            clone_loss.backward()
            self.irl_reward_optimizer.step()

            p_grad_abs = 0
            for p in self.irl_reward.parameters():
                p_grad_abs += torch.abs(p.grad).sum()

            mean_p_grad = p_grad_abs / self.irl_reward.get_param_dim()
            logger.log('train_irl_reward/grad_norm', mean_p_grad, step)


    def discounted_sum(self, tensor_to_sum, discount):
        # input has shape (batch, horizon)
        if self.discount_tensor is None:
            batch, horizon = tensor_to_sum.shape
            self.discount_tensor = discount * torch.ones_like(tensor_to_sum).to(self.device)
            self.time_exponents = torch.cat(batch * [torch.arange(horizon).unsqueeze(0).to(self.device)], dim=0)
            self.discount_tensor = self.discount_tensor ** self.time_exponents

        return (tensor_to_sum * self.discount_tensor).sum(dim=1, keepdims=True)


    def update_irl_reward_online_both(self, expert_obs, expert_acs, learner_obs, logger, step, use_double_q):
        batch_size = expert_obs.shape[0]

        if not self.reward_idx_init:
            self.learner_idx_reward = torch.nn.functional.one_hot(torch.zeros(batch_size, dtype=torch.int64),
                                                      num_classes=2).to(self.device)
            self.expert_idx_reward = torch.nn.functional.one_hot(torch.ones(batch_size, dtype=torch.int64),
                                                     num_classes=2).to(self.device)
            self.hybrid_idx_reward = torch.cat([self.learner_idx_reward, self.expert_idx_reward], dim=0)
            self.reward_idx_init = True

        learner_acs = self.actor(learner_obs).rsample()
        hybrid_obs = torch.cat([learner_obs, expert_obs], dim=0)
        hybrid_acs = torch.cat([learner_acs, expert_acs], dim=0)
        grad_Q1, grad_Q2 = self.irl_grad(hybrid_obs, hybrid_acs, self.hybrid_idx_reward, use_double_q)

        if use_double_q:
            grad_Q = torch.min(grad_Q1, grad_Q2)
        else:
            grad_Q = grad_Q1

        learner_grad = grad_Q[:batch_size].mean(dim=0).detach()
        expert_grad = grad_Q[batch_size:].mean(dim=0).detach()
        tot_grad = learner_grad - expert_grad

        self.irl_reward_optimizer.zero_grad()
        self.irl_reward.update_grad(tot_grad)
        self.irl_reward_optimizer.step()


    def update_irl_grad_hybrid(self, obs, action, irl_grad, next_obs, next_action, not_done, logger, step, use_double_q=False):
        batch = obs.shape[0]
        half_batch = int(batch/2)
#        assert batch == self.batch_size

        # get learner targets
        if not self.grad_idx_init:
            self.learner_idx_grad = torch.nn.functional.one_hot(torch.zeros(half_batch, dtype=torch.int64),
                                                           num_classes=2).to(self.device)
            self.expert_idx_grad = torch.nn.functional.one_hot(torch.ones(half_batch, dtype=torch.int64),
                                                     num_classes=2).to(self.device)
            self.hybrid_idx_grad = torch.cat([self.learner_idx_grad, self.expert_idx_grad], dim=0)

            self.grad_idx_init = True

        # get learner next actions
        learner_next_obs = next_obs[:half_batch]
        learner_dist = self.actor(learner_next_obs)
        learner_next_action = learner_dist.rsample()

        # get expert next actions
        expert_next_action = next_action[half_batch:]

        # aggregate learner, expert next actions
        next_action = torch.cat([learner_next_action, expert_next_action], dim=0)

        # get grad targets
        target_Q1, target_Q2 = self.irl_grad_target(next_obs, next_action, self.hybrid_idx_grad, use_double_q)

        # process double q targets
        if use_double_q:
            target_V = torch.min(target_Q1, target_Q2)
        else:
            target_V = target_Q1

        # get Q targets
        target_Q = irl_grad + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.irl_grad(obs, action, self.hybrid_idx_grad, use_double_q)

        # get Q-learning loss
        if use_double_q:
            irl_grad_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        else:
            irl_grad_loss = F.mse_loss(current_Q1, target_Q)

        logger.log('train_irl_grad/loss', irl_grad_loss, step)
        logger.log('train_irl_reward/true_grad_norm', torch.abs(target_Q).mean(), step)
        logger.log('train_irl_reward/pred_grad_norm', torch.abs(current_Q1).mean(), step)

        # Optimize the critic
        self.irl_grad_optimizer.zero_grad()
        irl_grad_loss.backward()
        self.irl_grad_optimizer.step()


    def update_critic_no_double_q(self, obs, action, reward, next_obs, next_action, not_done, logger, step, next_agent, no_double_q=True):
        dist = next_agent(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)

        if no_double_q:
            target_V = target_Q1 - self.alpha.detach() * log_prob
        else:
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob

        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        if no_double_q:
            current_Q1, _ = self.critic(obs, action)
            critic_loss = F.mse_loss(current_Q1, target_Q)
        else:
            # Double Q-learning loss
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

#        # Aggregates losses
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update_actor_no_double_q(self, obs, exp_action, logger, step, no_double_q=True):
        # Sample actions from actor to get (s, a)
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        actor_Q1, actor_Q2 = self.critic(obs, action)

        if no_double_q:
            actor_Q = actor_Q1
        else:
            actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()


    def update_reparam_mod(self, init=False):
#        self.irl_reward_copy = hydra.utils.instantiate(self.irl_reward_cfg).to(self.device)
#        self.irl_reward_copy.load_state_dict(self.irl_reward.state_dict())
        self.irl_reward_copy = copy.deepcopy(self.irl_reward)
        self.reparam_mod = ReparamModule(self.irl_reward_copy).to(self.device)


    def check_param_grad_correct(self, obs, action, irl_grad=None):
        if irl_grad is None:
            irl_grad = self.irl_reward.get_param_grad(obs, action, self.reparam_mod)

        self.irl_reward_optimizer.zero_grad()
        self.irl_reward(obs, action).mean().backward()
        irl_grad_mean = irl_grad.mean(dim=0)

        pointer = 0
        for param in self.irl_reward.parameters():
            # The length of the parameter
            num_param = param.numel()
            my_grad = irl_grad_mean[pointer:pointer+num_param].view_as(param).data

            assert torch.allclose(param.grad, my_grad, atol=1e-7), f"{param.grad.sum()}, {my_grad.sum()}"
            pointer += num_param

        self.irl_reward_optimizer.zero_grad()


    def update_one_step_rl(self, demonstrations, replay_buffer, logger, step, train_cfg):
        # Function to update the following modules
        # - Q-gradient: self.irl_grad
        # - Reward: self.irl_reward
        # - Actor: self.actor
        # - Critic: self.critic

        # Update Q-gradient
        half_batch = int(self.batch_size/2)
        for grad_idx in range(self.irl_reward_update_frequency):
            # Sample batch from learner replay buffer
            replay_obs, replay_action, replay_reward, replay_next_obs, replay_next_action, replay_not_done, replay_not_done_no_max = replay_buffer.sample(half_batch)

            # Sample a batch from expert demonstrations
            demo_obs, demo_action, demo_reward, demo_next_obs, demo_next_action, demo_not_done, demo_not_done_no_max = demonstrations.sample(half_batch)

            # Aggregate learner and expert data
            obs = torch.cat([replay_obs, demo_obs], dim=0)
            action = torch.cat([replay_action, demo_action], dim=0)
            reward = torch.cat([replay_reward, demo_reward], dim=0)
            next_obs = torch.cat([replay_next_obs, demo_next_obs], dim=0)
            next_action = torch.cat([replay_next_action, demo_next_action], dim=0)
            not_done = torch.cat([replay_not_done, demo_not_done], dim=0)
            not_done_no_max = torch.cat([replay_not_done_no_max, demo_not_done_no_max], dim=0)

            # Compute the reward parameter gradient at sampled state-actions
            irl_grad = self.irl_reward.get_param_grad(obs, action, self.reparam_mod, self.batch_size).detach()

            # Update the Q-gradient network
            self.update_irl_grad_hybrid(obs,
                                        action,
                                        irl_grad,
                                        next_obs,
                                        next_action,
                                        not_done_no_max,
                                        logger,
                                        step,
                                        use_double_q=False)


            # Update Q-gradient target network
            if self.irl_grad_step % self.irl_grad_target_update_frequency == 0:
                utils.soft_update_params(self.irl_grad, self.irl_grad_target, self.irl_grad_tau)

            # Update the Q-gradient step
            self.irl_grad_step += 1

        # Update Q-gradient for a couple extra steps before updating reward
        start_irl = train_cfg.num_seed_steps + 100

        # Update (Reward, Actor, Critic)
        if step >= start_irl:

            # Update irl_reward with gradient of log-likelihood of demonstrations
            assert self.irl_reward.training

            if train_cfg.learn_grad:
                # Update reward once every N steps
                if step % 20 == 0:
                    learner_obs, _, _, _, _, _, _ = replay_buffer.sample_initial(self.batch_size, init_tolerance=1)
                    expert_obs, expert_acs, _, _, _, _, _ = demonstrations.sample_initial(self.batch_size, init_tolerance=1)

                    self.update_irl_reward_online_both(expert_obs=expert_obs,
                                                       expert_acs=expert_acs,
                                                       learner_obs=learner_obs,
                                                       logger=logger,
                                                       step=step,
                                                       use_double_q=False)

                    self.update_reparam_mod()


                # Update critic via Policy Evaluation on the learned rewards
                obs, action, reward, next_obs, next_action, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
                irl_reward = self.irl_reward(obs, action).detach()
                self.update_critic_no_double_q(obs,
                                               action,
                                               irl_reward,
                                               next_obs,
                                               next_action,
                                               not_done_no_max,
                                               logger,
                                               step,
                                               self.actor,
                                               no_double_q=False)

                # Update critic target network
                if step % self.critic_target_update_frequency == 0:
                    utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

                # Update actor to be one-step improvement of critic
                self.update_actor_no_double_q(obs, action, logger, step, no_double_q=False)

        # Save aggregated evaluation metrics
        if step >= start_irl and step % 1000 == 0:
            logger.dump(step, save=True, ty='train')

        # Evaluation
        if step >= start_irl and step % train_cfg.eval_frequency == 0:
            # Sample expert demonstration data
            obs, action, reward, next_obs, next_action, not_done, not_done_no_max = demonstrations.sample(self.batch_size)

            # Sample data from random policy (collected during seeding of the replay buffer)
            rand_obs, rand_acs, _, _, _, _, _ = replay_buffer.sample_random_seed(self.batch_size, train_cfg.num_seed_steps)

            # Compute reward weight norms
            l2_norms = [torch.sum(w**2) for w in self.irl_reward.parameters()]
            logger.log('eval/weight_norm', sum(l2_norms), step)

            # Expert (s), Expert (a)
            exp_q, _ = self.critic(next_obs, next_action)
            exp_r = self.irl_reward(next_obs, next_action)
            print(exp_r[:200])

            # Expert (s), Random (a)
            r_q, _ = self.critic(next_obs, rand_acs)
            r_r = self.irl_reward(next_obs, rand_acs)

            # Random (s), Random (a)
            tr_q, _ = self.critic(rand_obs, rand_acs)
            tr_r = self.irl_reward(rand_obs, rand_acs)

            logger.log(f'eval/R_exp_sa', exp_r.mean(), step)
            logger.log(f'eval/Q_exp_sa', exp_q.mean(), step)
            logger.log(f'eval/R_exp_s_ran_a', r_r.mean(), step)
            logger.log(f'eval/Q_exp_s_ran_a', r_q.mean(), step)
            logger.log(f'eval/R_ran_s_ran_a', tr_r.mean(), step)
            logger.log(f'eval/Q_ran_s_ran_a', tr_q.mean(), step)

        return step



