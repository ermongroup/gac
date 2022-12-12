import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils
import copy
import future, sys, os, datetime, argparse
from typing import List, Tuple
import pdb
import time

from torchreparam import ReparamModule
from collections import defaultdict

#from torch.nn.utils.parametrizations import spectral_norm


class RewardGradientSeparateMLP(nn.Module):
    """Reward Gradient network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, output_dim):
        super().__init__()

        # reward_param_dim is the length of the "flattened" array
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, output_dim, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, output_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, agent_type):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)

        if agent_type == 'learner':
            q1 = self.Q1(obs_action)

        elif agent_type == 'expert':
            q1 = self.Q2(obs_action)

        q2 = False

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class RewardGradientHybridMLP(nn.Module):
    """Reward Gradient network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, output_dim):
        super().__init__()

        # reward_param_dim is the length of the "flattened" array
        self.Q1 = utils.mlp(obs_dim + action_dim + 2, hidden_dim, output_dim, hidden_depth)
#        self.Q2 = utils.mlp(obs_dim + action_dim + 2, hidden_dim, output_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, agent_idx, use_double_q):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action, agent_idx], dim=-1)
        q1 = self.Q1(obs_action)

        if use_double_q:
#            q2 = self.Q2(obs_action)
            q2 = False
        else:
            q2 = False

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class RewardGradientMLP(nn.Module):
    """Reward Gradient network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, output_dim):
        super().__init__()

        # reward_param_dim is the length of the "flattened" array
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, output_dim, hidden_depth)
#        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, output_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = False
#        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2



class RewardMLP(nn.Module):
    """Simple Reward network """
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, factor):
        super().__init__()

        print(f"MLP with sigmoid output")
        self.output = utils.mlp(obs_dim + action_dim,
                                hidden_dim,
                                1,
                                hidden_depth,
                                nn.Sigmoid())


        self.param_dim = len(nn.utils.parameters_to_vector(self.parameters()))
        self.device = 'cuda'

        # weight intialization
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        rewards = self.output(obs_action)

        return rewards

    def get_param_dim(self):
        return self.param_dim

    def get_param_grad(self, obs, acs, reparam_mod, minibatch_size=1024):
        assert obs.shape[0] == acs.shape[0]
        batch = obs.shape[0]

#        minibatch_size = 1024
        num_minibatch = batch // minibatch_size
        remainder = batch % minibatch_size

        t1 = time.time()
        device = self.device
#        device = next(self.parameters()).device
#        model_copy = copy.deepcopy(self)
        t2 = time.time()
#        reparam_mod = ReparamModule(model_copy).to(device)
#        reparam_mod = ReparamModule(self).to(device)
        cur_params = reparam_mod.flat_param.clone().detach().requires_grad_().to(device)
        t3 = time.time()


        grad_list = []
        should_vectorize = True
        for idx in range(num_minibatch):
            param_func = lambda p_vec: reparam_mod(obs[idx*minibatch_size:(idx + 1) * minibatch_size],
                                                   acs[idx*minibatch_size:(idx + 1) * minibatch_size],
                                                   flat_param=p_vec).flatten()

            mini_param_grad = torch.autograd.functional.jacobian(param_func,
                                                                 cur_params,
                                                                 strict=False,
                                                                 vectorize=should_vectorize)
            grad_list.append(mini_param_grad)


        if remainder != 0:
            param_func = lambda p_vec: reparam_mod(obs[num_minibatch * minibatch_size:],
                                                   acs[num_minibatch * minibatch_size:],
                                                   flat_param=p_vec).flatten()

            mini_param_grad = torch.autograd.functional.jacobian(param_func,
                                                                 cur_params,
                                                                 strict=False,
                                                                 vectorize=should_vectorize)
            grad_list.append(mini_param_grad)


        t4 = time.time()
        param_grad = torch.cat(grad_list, dim=0)
        t5 = time.time()


        assert param_grad.size() == (batch, self.param_dim)


#        time_out = {'copytime': t2 - t1,
#                    'reparamtime': t3 - t2,
#                    'jacobtime': t4 - t3,
#                    'cattime': t5 - t4
#                    }


#        assert torch.equal(test_grad, param_grad)
#        pdb.set_trace()

        return param_grad #, time_out


    def update_grad(self, flat_grad):
        # make sure gradient is of size (param_dim, )
        # check that gradient is on the same device as parameters
        assert flat_grad.size() == (self.param_dim, )
        assert flat_grad.device == next(self.parameters()).device

        # Ensure vec of type Tensor
        if not isinstance(flat_grad, torch.Tensor):
            raise TypeError('expected torch.Tensor, but got: {}'
                            .format(torch.typename(flat_grad)))
        # Flag for the device where the parameter is located
        param_device = None

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for _, param in self.named_parameters():

            # The length of the parameter
            num_param = param.numel()

            if param.grad == None:
                param.grad = torch.zeros_like(param)

            # Slice the vector, reshape it, and replace the old data of the parameter
            # Go through parameter list and update the grad attribute
            param.grad += flat_grad[pointer:pointer+num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param



class StateOnlyRewardMLP(RewardMLP):
    """Simple Reward network """
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, factor):
        super().__init__(obs_dim, action_dim, hidden_dim, hidden_depth, factor)

        # output
        self.output = utils.leaky_mlp(obs_dim,
                                      hidden_dim,
                                      1,
                                      hidden_depth,
                                      nn.Sigmoid())

#        self.output = nn.Linear(obs_dim+action_dim, 1)
        self.param_dim = len(nn.utils.parameters_to_vector(self.parameters()))

        # weight intialization
        self.apply(utils.weight_init)


    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        rewards = self.output(obs)
        return rewards



class RewardCutMLP(nn.Module):
    """Simple Reward network """
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        cut_dim = 4
        self.output = utils.mlp(cut_dim,
                                hidden_dim,
                                1,
                                hidden_depth,
                                nn.Sigmoid())

#        self.output = nn.Linear(obs_dim+action_dim, 1)
        self.param_dim = len(nn.utils.parameters_to_vector(self.parameters()))

        # weight intialization
        self.apply(utils.weight_init)


    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        # Cut obs but keep action
        obs_action = torch.cat([obs[:, 2:4], action], dim=-1)
        rewards = self.output(obs_action)

        return rewards

    def get_param_dim(self):
        return self.param_dim

    def get_param_grad(self, obs, acs):
        assert obs.shape[0] == acs.shape[0]
        batch = obs.shape[0]

        minibatch_size = 1000
        num_minibatch = batch // minibatch_size
        remainder = batch % minibatch_size

        device = next(self.parameters()).device
        model_copy = copy.deepcopy(self)

        reparam_mod = ReparamModule(model_copy).to(device)
        cur_params = reparam_mod.flat_param.clone().detach().requires_grad_().to(device)

        grad_list = []
        for idx in range(num_minibatch):
            param_func = lambda p_vec: reparam_mod(obs[idx*minibatch_size:(idx + 1) * minibatch_size],
                                                   acs[idx*minibatch_size:(idx + 1) * minibatch_size],
                                                   flat_param=p_vec).flatten()

            mini_param_grad = torch.autograd.functional.jacobian(param_func,
                                                                 cur_params,
                                                                 strict=False,
                                                                 vectorize=True)
            grad_list.append(mini_param_grad)

        if remainder != 0:
            param_func = lambda p_vec: reparam_mod(obs[num_minibatch * minibatch_size:],
                                                   acs[num_minibatch * minibatch_size:],
                                                   flat_param=p_vec).flatten()

            mini_param_grad = torch.autograd.functional.jacobian(param_func,
                                                                 cur_params,
                                                                 strict=False,
                                                                 vectorize=True)
            grad_list.append(mini_param_grad)

        param_grad = torch.cat(grad_list, dim=0)
        assert param_grad.size() == (batch, self.param_dim)

        return param_grad


    def update_grad(self, flat_grad):
        # make sure gradient is of size (param_dim, )
        # check that gradient is on the same device as parameters
        assert flat_grad.size() == (self.param_dim, )
        assert flat_grad.device == next(self.parameters()).device

        # Ensure vec of type Tensor
        if not isinstance(flat_grad, torch.Tensor):
            raise TypeError('expected torch.Tensor, but got: {}'
                            .format(torch.typename(flat_grad)))
        # Flag for the device where the parameter is located
        param_device = None

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for _, param in self.named_parameters():

            # The length of the parameter
            num_param = param.numel()

            if param.grad == None:
                param.grad = torch.zeros_like(param)

            # Slice the vector, reshape it, and replace the old data of the parameter
            # Go through parameter list and update the grad attribute
            param.grad += flat_grad[pointer:pointer+num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param














if __name__ == '__main__':
    batch = 100000
    obs_dim = 6
    acs_dim = 2
    hidden_dim = 64
    hidden_depth = 1
    lr = 0.1

#    device = torch.device('cuda')
    device = 'cuda'
    reward = RewardMLP(obs_dim, acs_dim, hidden_dim, hidden_depth).to(device)
    op = torch.optim.SGD(reward.parameters(), lr=lr)
    obs = torch.randn(batch, obs_dim).to(device)
    acs = torch.randn(batch, acs_dim).to(device)

    t0 = time.time()
    loop_num = 50
#    jac_list = compute_jacobian(reward, obs, acs)
    my_list = []
    target_list = []
    time_out_dict = defaultdict(lambda: 0)


    reward_copy = RewardMLP(obs_dim, acs_dim, hidden_dim, hidden_depth).to(device)
    reward_copy.load_state_dict(reward.state_dict())
    reparam_mod = ReparamModule(reward_copy).to(device)


    for i in range(loop_num):

        op.zero_grad()

        obs = torch.randn(batch, obs_dim).to(device)
        acs = torch.randn(batch, acs_dim).to(device)

        t1 = time.time()
        param_grad = reward.get_param_grad(obs, acs, reparam_mod).detach()
        t2 = time.time()

        my_list.append(t2 - t1)

#        for k, v in time_out.items():
#            time_out_dict[k] += v

#        print("Batch size: {} | Single Time: {}".format(batch, t2 - t1))

#        op.zero_grad()
#        avg_grad = param_grad.mean(0)
#        reward.update_grad(avg_grad)
#
#    #    print("avg_grad: {}".format(avg_grad))
#        old_p = []
#        for n, p in reward.named_parameters():
##            print("parameter | name: {}, value: {}".format(n, p.data[0]))
##            print("gradient | name: {}, value: {}".format(n, p.grad[0]))
#            old_p.append(p.clone())
#
#        op.step()
#
#        new_p = []
#        for n, p in reward.named_parameters():
##            print("parameter | name: {}, value: {}".format(n, p.data[0]))
##            print("gradient | name: {}, value: {}".format(n, p.grad[0]))
#            new_p.append(p.clone())
#
#        pointer = 0
#        for old, new in zip(old_p, new_p):
#            len_p = old.numel()
#            delta = avg_grad[pointer:pointer+len_p].view_as(old)
#            up_old = old - lr * delta
#            if not torch.all(torch.isclose(up_old, new)):
#                print("old parameter | value: {}".format(up_old.data[0]))
#                print("new parameter | value: {}".format(new.data[0]))
#
#            pointer += len_p

    for i in range(loop_num):

        op.zero_grad()

        obs = torch.randn(batch, obs_dim).to(device)
        acs = torch.randn(batch, acs_dim).to(device)

        t2 = time.time()
        jvp_grad = reward(obs, acs).mean().backward()
        t3 = time.time()
        target_list.append(t3 - t2)


    print(f"Batch size: {batch} | Avg Time: {np.mean(my_list)}, {np.mean(target_list)}")
#    for k, v in time_out_dict.items():
#        print(f"{k}: {v/loop_num}")







