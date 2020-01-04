import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import random, randint


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.fc1 = nn.Linear(1728, 128)
        self.fc_v = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)

        return v


class DQN:

    def __init__(self, threads=2, memory_size=1000000, gamma=0.99, batch_size=32, **kwargs):

        torch.set_num_threads(threads)

        self.batch_size = batch_size

        self.eps = 1

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

        self.model = Model()

        self.last_score = 0

        self.fresh_start = True

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-3)

        self.loss = nn.SmoothL1Loss()

    def play(self):

        state_tensor = torch.from_numpy(self.state)

        if random() < self.eps:
            action = randint(0, 6)
        else:
            with torch.no_grad():
                q_value = self.model(state_tensor)[0]
            action = q_value.argmax().item()

        self.last_action = action

        return action

    def train_step(self):

        idx = np.random.randint(min(self.memory_index, self.memory_size), size=self.batch_size)

        si, a, r, sf, end = [x[idx] for x in self.memory]
        qi = self.model(torch.from_numpy(si))

        qf = self.model(torch.from_numpy(sf)).max(1)[0]

        y = qi.clone().detach()
        for i in range(self.batch_size):
            if end[i]:
                y[i, a[i]] = torch.tensor(r[i])
            else:
                y[i, a[i]] = r[i] + self.gamma * qf[i]

        loss = self.loss(qi, y)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.eps = 1 - 0.99 * min(self.memory_index / self.memory_size, 1)

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

    def close(self):
        pass
