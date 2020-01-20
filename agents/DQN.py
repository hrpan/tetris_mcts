import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import random, randint
from copy import deepcopy


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc1 = nn.Linear(3456, 128)
        self.fc_v = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)

        return v


class DistModel(nn.Module):
    def __init__(self, atoms=50):
        super(DistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.fc1 = nn.Linear(1728, 128)
        self.fc_v = nn.Linear(128, 7 * atoms)
        self.atoms = atoms

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc_v(x).view(x.shape[0], 7, self.atoms)
        p = F.softmax(x, dim=2)

        return p


def prob_to_exp(p, atoms, vmin, vmax):
    bin_center = (torch.arange(atoms) + 0.5) * (vmax - vmin) / atoms
    q = (p * bin_center).sum(dim=2)
    return q


class DQN:

    def __init__(self,
                 threads=2,
                 memory_size=1000000,
                 gamma=0.99,
                 eps_init=1,
                 eps_final=0.01,
                 batch_size=32,
                 target=True,
                 target_update_steps=100,
                 distributional=True,
                 distributional_atoms=50,
                 distributional_vmax=2500,
                 distributional_vmin=0,
                 **kwargs):

        torch.set_num_threads(threads)

        self.batch_size = batch_size

        self.eps_init = eps_init
        self.eps_final = eps_final

        self.eps = eps_init

        self.gamma = gamma

        self.memory_size = memory_size

        self.memory_index = 0

        self.memory = [
            np.empty((memory_size, 1, 22, 10), dtype=np.float32),
            np.empty((memory_size, ), dtype=np.int),
            np.empty((memory_size, ), dtype=np.float32),
            np.empty((memory_size, 1, 22, 10), dtype=np.float32),
            np.empty((memory_size, ), dtype=np.bool)
        ]

        self.distributional = distributional
        self.distributional_atoms = distributional_atoms
        self.distributional_vmax = distributional_vmax
        self.distributional_vmin = distributional_vmin

        if distributional:
            self.model = DistModel(atoms=distributional_atoms)
        else:
            self.model = Model()

        self.target = target
        self.target_update_steps = target_update_steps
        self.target_steps = 0

        if target:
            self.target_model = deepcopy(self.model)

        self.last_score = 0

        self.fresh_start = True

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-3)

    def play(self):

        state_tensor = torch.from_numpy(self.state)

        if random() < self.eps:
            action = randint(0, 6)
        else:
            with torch.no_grad():
                if self.distributional:
                    q_dist = self.model(state_tensor)
                    atoms = self.distributional_atoms
                    vmin = self.distributional_vmin
                    vmax = self.distributional_vmax
                    q_value = prob_to_exp(q_dist, atoms, vmin, vmax)[0]
                else:
                    q_value = self.model(state_tensor)[0]
            action = q_value.argmax().item()

        self.last_action = action

        return action

    def train_step(self):

        idx = np.random.randint(min(self.memory_index, self.memory_size), size=self.batch_size)

        si, a, r, sf, end = [x[idx] for x in self.memory]
        qi = self.model(torch.from_numpy(si))

        with torch.no_grad():
            if self.target:
                qf = self.target_model(torch.from_numpy(sf))
                self.target_steps = (self.target_steps + 1) % self.target_update_steps
                if self.target_steps == 0:
                    self.save_model()
                    self.target_model = deepcopy(self.model)

            else:
                qf = self.model(torch.from_numpy(sf))

        if self.distributional:
            gamma = self.gamma
            atoms = self.distributional_atoms
            vmin = self.distributional_vmin
            vmax = self.distributional_vmax
            delta = (vmax - vmin) / atoms
            qf_exp = prob_to_exp(qf, atoms, vmin, vmax)
            qf_max_action = qf_exp.argmax(dim=1)
            y = torch.zeros(qi.shape)

            r_bin = np.clip(np.floor((r - vmin) / delta).astype(np.int), 0, atoms - 1)
            lb = gamma * (np.tile(np.arange(atoms), (self.batch_size, 1)) * delta + vmin) + np.expand_dims(r, -1)
            lb_bin = (lb - vmin) / delta
            lb_bin_fl = np.clip(np.floor(lb_bin).astype(np.int), 0, atoms - 1)
            ub_bin_fl = np.clip(np.floor(lb_bin + gamma).astype(np.int), 0, atoms - 1)
            fraction = (ub_bin_fl - lb_bin) / gamma

            for i in range(self.batch_size):
                if end[i]:
                    y[i, a[i], r_bin[i]] = 1
                else:
                    _lb_bin_fl = lb_bin_fl[i]
                    _ub_bin_fl = ub_bin_fl[i]
                    _p = qf[i, qf_max_action[i]].numpy()
                    _pf = _p * fraction[i]
                    _pfc = _p - _pf
                    for atom in range(atoms):
                        y[i, a[i], _lb_bin_fl[atom]] += _pf[atom]
                        y[i, a[i], _ub_bin_fl[atom]] += _pfc[atom]
            loss = - (y * torch.log(qi)).sum() / self.batch_size
        else:
            qf = qf.max(1)[0]
            y = qi.clone().detach()

            for i in range(self.batch_size):
                if end[i]:
                    y[i, a[i]] = torch.tensor(r[i])
                else:
                    y[i, a[i]] = r[i] + self.gamma * qf[i]

            loss = F.smooth_l1_loss(qi, y)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.eps = self.eps_init + (self.eps_final - self.eps_init) * min(self.memory_index / self.memory_size, 1)

    def update_root(self, game, episode=None):

        si, a, r, sf, end = self.memory

        idx = self.memory_index % self.memory_size

        state = game.getState()

        score = game.getScore()

        if self.fresh_start:
            si[idx, 0] = state
            self.last_score = score
            self.fresh_start = False
        else:
            sf[idx, 0] = state
            r[idx] = score - self.last_score
            self.last_score = score
            a[idx] = self.last_action
            end[idx] = game.end
            self.memory_index += 1
            idx = (idx + 1) % self.memory_size
            si[idx] = state
            if game.end:
                self.fresh_start = True
            else:
                si[idx] = state

        self.state = state[None, None, :, :].astype(np.float32)

        if self.memory_index > 0:
            self.train_step()
        if self.memory_index % 1000 == 0:
            print('Memory index: {:10}    Epsilon: {:5.3f}'.format(self.memory_index, self.eps), flush=True)

    def load_model(self):
        state_dict = torch.load('./pytorch_model/dqn_ckpt')
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.model.load_state_dict(state_dict['model_state_dict'])

    def save_model(self):
        torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': self.model.state_dict(),
            }, './pytorch_model/dqn_ckpt')

    def close(self):
        pass
