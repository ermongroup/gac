import numpy as np
import torch
from replay_buffer import ReplayBuffer
import utils
import pdb


class Demonstrations(ReplayBuffer):
    """Buffer to store demonstrations in the format (s, a, s', a')"""
    def __init__(self, obs_shape, action_shape, capacity, device):

        super().__init__(obs_shape, action_shape, capacity, device)

        self.next_actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.recent_idx = None
        self.init_idx = False
        self.last_idx = 0


    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, next_action, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_actions[self.idx], next_action)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        self.recent_idx = idxs = np.random.randint(0, self.capacity if self.full else self.idx,
                                                   size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        next_actions = torch.as_tensor(self.next_actions[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, next_actions, not_dones, not_dones_no_max


    def sample_3d(self, batch_size, horizon):
        if self.full:
            assert self.capacity > horizon
        else:
            assert self.idx > horizon

        self.recent_idx = idxs = np.random.randint(0,
                                                   self.capacity - horizon if self.full else self.idx - horizon,
                                                   size=(batch_size, 1))

        idxs = np.concatenate([idxs + i for i in range(horizon)], axis=1)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        next_actions = torch.as_tensor(self.next_actions[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, next_actions, not_dones, not_dones_no_max


    def sample_initial(self, batch_size, init_tolerance=10, episode_len=1000):
        buffer_size = self.capacity if self.full else self.idx
        episode_len = 1000

        if not self.init_idx:
            self.init_idx = [init_idx for init_idx in range(buffer_size) if init_idx % episode_len < init_tolerance]

        elif (not self.full) and (self.idx > self.last_idx):
            for new_idx in range(self.last_idx, self.idx):
                if new_idx % episode_len < init_tolerance:
                    self.init_idx.append(new_idx)

        self.last_idx = self.idx

        idxs = np.random.choice(self.init_idx, size=batch_size, replace=True)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        next_actions = torch.as_tensor(self.next_actions[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, next_actions, not_dones, not_dones_no_max


    def sample_recent(self, batch_size, recent_horizon=100000):
        buffer_size = self.capacity if self.full else self.idx
        assert not self.full
#        assert buffer_size > recent_horizon
        recent_idx = list(range(np.max([0, buffer_size - recent_horizon]), buffer_size))

        idxs = np.random.choice(recent_idx, size=batch_size, replace=True)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        next_actions = torch.as_tensor(self.next_actions[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, next_actions, not_dones, not_dones_no_max


    def sample_random_seed(self, batch_size, num_seed=200000):
        buffer_size = self.capacity if self.full else self.idx
        assert not self.full
        assert buffer_size >= num_seed

        seed_idx = list(range(num_seed))

        idxs = np.random.choice(seed_idx, size=batch_size, replace=True)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        next_actions = torch.as_tensor(self.next_actions[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, next_actions, not_dones, not_dones_no_max


    def get_all_obs_acs(self):

        obses = torch.as_tensor(self.next_obses, device=self.device).float()
        actions = torch.as_tensor(self.next_actions, device=self.device)

        return obses, actions


    # TODO: check that this function is correct again
    def get_subset(self, num_trajectories, traj_len=1000):
        # check that this is a demonstration set, and thus the buffer is full
        demo_num_traj = int(self.capacity / traj_len)
        assert self.full
        assert demo_num_traj >= num_trajectories
        assert self.capacity % traj_len == 0

        traj_idxs = traj_len * np.random.choice(demo_num_traj,
                                                size=num_trajectories,
                                                replace=False)

        idxs = [idx + j for idx in traj_idxs for j in range(traj_len)]

        self.obses = self.obses[idxs]
        self.next_obses = self.next_obses[idxs]
        self.actions = self.actions[idxs]
        self.next_actions = self.next_actions[idxs]
        self.rewards = self.rewards[idxs]
        self.not_dones = self.not_dones[idxs]
        self.not_dones_no_max = self.not_dones_no_max[idxs]

        self.capacity = num_trajectories * traj_len
        self.idx = 0


    def update_irl_rewards(self, reward_func):

        assert isinstance(reward_func, torch.nn.Module)

        all_obs = torch.as_tensor(self.obses, device=self.device).float()
        all_acs = torch.as_tensor(self.actions, device=self.device)

        param_dim = reward_func.get_param_dim()

        # TODO: confirm that eval_mode is the right thing here
        with utils.eval_mode(reward_func):
            self.irl_rewards = utils.to_np(reward_func(all_obs, all_acs))
            self.irl_grads = utils.to_np(reward_func.get_param_grad(all_obs, all_acs))

        assert self.irl_rewards.shape == (self.capacity, 1)
        assert self.irl_grads.shape == (self.capacity, param_dim)

    def sample_irl_rewards(self, sample_idx=None):
        if sample_idx is not None:
            # Use user specified sampling idices
            idxs = sample_idx
        else:
            # Use the indices for the most recently sampled transitions
            assert self.recent_idx is not None
            assert hasattr(self, 'irl_rewards') and hasattr(self, 'irl_grads')

            idxs = self.recent_idx

        irl_rewards = torch.as_tensor(self.irl_rewards[idxs], device=self.device)
        irl_grads = torch.as_tensor(self.irl_grads[idxs], device=self.device)

        return irl_rewards, irl_grads






