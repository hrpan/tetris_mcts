import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.utils.data as D
import os, subprocess
import numpy as np
import math
from collections import defaultdict
from caffe2.python import workspace
IMG_H, IMG_W, IMG_C = (22, 10, 1)

EXP_PATH = './pytorch_model/'

n_actions = 7

eps = 0.01


def convOutShape(shape_in, kernel_size, stride):

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    return ((shape_in[0] - kernel_size[0]) // stride[0] + 1,
            (shape_in[1] - kernel_size[1]) // stride[1] + 1)


class Net(nn.Module):

    def __init__(self, atoms=50):
        super(Net, self).__init__()

        kernel_size = 3
        stride = 1
        filters = 32

        self.conv1 = nn.Conv2d(1, filters, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        _shape = convOutShape((22, 10), kernel_size, stride)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        _shape = convOutShape(_shape, kernel_size, stride)

        flat_in = _shape[0] * _shape[1] * filters

        n_fc1 = 128
        self.fc1 = nn.Linear(flat_in, n_fc1)
        flat_out = n_fc1

        self.fc_v = nn.Linear(flat_out, atoms)

    def forward(self, x):
        act = F.relu
        x = self.bn1(act(self.conv1(x)))
        x = self.bn2(act(self.conv2(x)))
        x = x.view(x.shape[0], -1)

        x = act(self.fc1(x))

        value = F.softmax(self.fc_v(x), 1)

        return value

    def log_prob(self, x):
        act = F.relu
        x = self.bn1(act(self.conv1(x)))
        x = self.bn2(act(self.conv2(x)))
        x = x.view(x.shape[0], -1)

        x = act(self.fc1(x))

        value = F.log_softmax(self.fc_v(x), 1)

        return value



def convert(x):
    return torch.from_numpy(x.astype(np.float32))

class Model:
    def __init__(self, training=False, new=True, use_cuda=True, atoms=50):

        self.use_cuda = use_cuda

        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.training = training

        self.model = Net(atoms=atoms)
        if self.use_cuda:
            self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-4, amsgrad=True)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
        self.scheduler = None
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: max(0.9 ** step, 1e-2))


    def _loss(self, batch):

        state, value, weight = batch

        if type(state) is not torch.Tensor:
            state = torch.from_numpy(state)
            value = torch.from_numpy(value)
            weight = torch.from_numpy(weight)

        if self.use_cuda:
            state = state.cuda()
            value = value.cuda()
            weight = weight.cuda()

        _v = self.model.log_prob(state)
        loss = torch.std_mean(-weight * (value * _v).sum(dim=1))

        return defaultdict(float, loss=loss[1], loss_std=loss[0])

    def compute_loss(self, batch, chunksize=2048):

        self.model.eval()

        result = defaultdict(float)
        d_size = len(batch[0])
        ls, stds, bs = [], [], []
        with torch.no_grad():
            for c in range((d_size + chunksize - 1) // chunksize):
                b = [d[c * chunksize: (c + 1) * chunksize] for d in batch]
                loss = self._loss(b)
                ls.append(loss['loss'])
                stds.append(loss['loss_std'])
                bs.append(len(b[0]))

        b = np.array(bs)
        l = np.array(ls)
        l_avg = np.sum(l * b) / d_size
        result['loss'] = l_avg

        std = np.array(stds)
        l_sq = (b - 1) * std ** 2 / b + l ** 2
        l_sq_combined = np.sum(l_sq * b) / d_size
        l_std_combined = ((l_sq_combined - l_avg ** 2) * d_size / (d_size - 1)) ** 0.5

        result['loss_std'] = l_std_combined

        return result

    def train(self, batch):

        self.model.train()

        self.optimizer.zero_grad()

        loss = self._loss(batch)

        loss['loss'].backward()
        #norm = 0.
        #for p in self.model.parameters():
        #    if p.grad is None:
        #        continue
        #    norm += p.grad.data.norm(2).item() ** 2
        #norm = norm ** 0.5
        #print(' ', norm)
        #if norm > 100:
        #    input()
        #U.clip_grad_norm_(self.model.parameters(), 10)

        self.optimizer.step()

        return {k: v.item() for k, v in loss.items()}

    def inference(self, batch):

        self.model.eval()

        b = torch.from_numpy(batch).float()
        if self.use_cuda:
            b = b.cuda()

        with torch.no_grad():
            output = self.model(b)

        result = [o.cpu().numpy() for o in output]

        return result

    def inference_stochastic(self, batch):

        self.model.eval()

        #state = convert(batch)
        b = torch.from_numpy(batch).float()
        if self.use_cuda:
            b = b.cuda()

        with torch.no_grad():
            output = self.model(b)

        result = [o.cpu().numpy() for o in output]
        v = np.random.normal(result[0], np.sqrt(result[1]))
        result[0] = v
        return result

    def reset_optimizer(self):

        self.optimizer.state = defaultdict(dict)

    def update_scheduler(self, **kwarg):

        if self.scheduler:
            self.scheduler.step(**kwarg)

    def save(self, verbose=True):

        if verbose:
            print('Saving model...', flush=True)

        if not os.path.isdir(EXP_PATH):
            if verbose:
                print('Export path does not exist, creating a new one...', flush=True)
            os.mkdir(EXP_PATH)

        filename = EXP_PATH + 'dist_model_checkpoint'

        full_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

        if self.scheduler:
            full_state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(full_state, filename)


    def load(self, filename=EXP_PATH + 'dist_model_checkpoint'):

        #filename = EXP_PATH + 'model_checkpoint'

        if os.path.isfile(filename):
            print('Loading model...', flush=True)
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.p0 = [p.clone() for p in self.model.parameters()]
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print('Checkpoint not found, using default model', flush=True)
