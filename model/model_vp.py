import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_optimizer as toptim
from collections import defaultdict, OrderedDict
from model.model import convOutShape, Model, n_actions


class Net(nn.Module):

    def __init__(self, input_shape=(22, 10), eps=.1):
        super(Net, self).__init__()

        kernel_size = 3
        stride = 1
        filters = 32
        bias = False

        self.activation = nn.ReLU(inplace=True)

        _shape = convOutShape(input_shape, kernel_size, stride)
        _shape = convOutShape(_shape, kernel_size, stride)
        flat_in = _shape[0] * _shape[1] * filters

        n_fc = 256
        flat_out = n_fc // 2

        self.fc_v = nn.Linear(flat_out, 1)
        self.fc_p = nn.Linear(flat_out, n_actions)

        self.fc_v.bias.data[0] = 100.

        self.head = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, filters, kernel_size, stride, bias=bias)),
                ('norm1', nn.BatchNorm2d(filters)),
                ('act1', self.activation),
                ('conv2', nn.Conv2d(filters, filters, kernel_size, stride, bias=bias)),
                ('norm2', nn.BatchNorm2d(filters)),
                ('act2', self.activation),
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(flat_in, n_fc)),
                ('act3', self.activation),
            ]))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        value, logit = self.value_and_logit(x)

        policy = self.softmax(logit)

        return value, policy

    def value_and_logit(self, x):

        x = self.head(x)

        xv, xp = x.split(128, dim=1)

        value = self.fc_v(xv)
        logit = self.fc_p(xp)

        return value, logit


class Model_VP(Model):
    def __init__(self, label_smoothing=0.1, **kwarg):

        super().__init__(self, **kwarg)

        self.label_smoothing = label_smoothing

    def _init_model(self):

        self.model = Net()
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = torch.jit.script(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-3, amsgrad=True)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)

        self.scheduler = None

        self.vloss = nn.SmoothL1Loss(reduction='none')
        self.ploss = nn.KLDivLoss(reduction='none')

    def _loss(self, batch):

        batch_tensor = [torch.as_tensor(b, dtype=torch.float, device=self.device) for b in batch]

        state, value, policy, weight = batch_tensor

        policy.div_(policy.sum(dim=1, keepdim=True)).mul_(1 - self.label_smoothing).add_(self.label_smoothing / n_actions)

        _v, _logit = self.model.value_and_logit(state)

        loss = torch.std_mean(weight * (self.vloss(_v, value) + self.ploss(_logit.log_softmax(dim=1), policy)))
        return defaultdict(float, loss=loss[1], loss_std=loss[0])
