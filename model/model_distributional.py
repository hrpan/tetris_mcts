import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.utils.data as D
import os
import numpy as np
from collections import defaultdict, OrderedDict
from model.yogi import Yogi
from model.model import convOutShape, Model

EXP_PATH = './pytorch_model/'

n_actions = 7


class Net(nn.Module):

    def __init__(self, input_shape=(22, 10), atoms=50):
        super().__init__()

        kernel_size = 4
        stride = 1
        filters = 32

        _shape = convOutShape((22, 10), kernel_size, stride)
        _shape = convOutShape(_shape, kernel_size, stride)

        flat_in = _shape[0] * _shape[1] * filters

        n_fc1 = 128

        activation = nn.LeakyReLU(inplace=True)

        self.seq = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(1, filters, kernel_size, stride)),
                    ('act1', activation),
                    ('conv2', nn.Conv2d(filters, filters, kernel_size, stride)),
                    ('act2', activation),
                    ('flatten', nn.Flatten()),
                    ('fc1', nn.Linear(flat_in, n_fc1)),
                    ('act3', activation),
                    ('fc_v', nn.Linear(n_fc1, atoms)),
                ]))

    def forward(self, x):
        x = self.seq(x)
        value = F.softmax(x, 1)

        return value

    def log_prob(self, x):
        x = self.seq(x)
        value = F.log_softmax(x, 1)

        return value


class Model_Dist(Model):
    def __init__(self, atoms=50, **kwargs):

        super().__init__(**kwargs)

        self.model = Net(atoms=atoms)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-5, amsgrad=True)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
        self.scheduler = None
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: max(0.9 ** step, 1e-2))

    def _init_model(self):

        self.model = Net()
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = torch.jit.script(self.model)

        self.optimizer = Yogi(self.model.parameters(), lr=1e-3, eps=1e-3)

    def _loss(self, batch, weighted=False):

        batch_tensor = [torch.as_tensor(b, dtype=torch.float, device=self.device) for b in batch]
        state, value, weight = batch_tensor

        _v = self.model.log_prob(state)

        loss = -value * (_v - value.log())
        if weighted:
            loss = weight.squeeze() * loss
        std, mean = torch.std_mean(loss.sum(dim=1))

        return defaultdict(float, loss=mean, loss_std=std)

    def inference(self, batch):

        b = torch.as_tensor(batch, dtype=torch.float, device=self.device)

        with torch.no_grad():
            output = self.model(b).cpu()

        return [output.numpy()]
