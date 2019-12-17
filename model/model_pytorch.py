import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.onnx
import torch.utils.data as D
#import onnx
import os, subprocess
import numpy as np
from collections import defaultdict
from caffe2.python import workspace
IMG_H, IMG_W, IMG_C = (22, 10, 1)

EXP_PATH = './pytorch_model/'

n_actions = 7

eps = 1e-1


def convOutShape(shape_in, kernel_size, stride):

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    return ((shape_in[0] - kernel_size[0]) // stride[0] + 1,
            (shape_in[1] - kernel_size[1]) // stride[1] + 1)


class DenseBlock(nn.Module):
    def __init__(self, channels_in, growth_rate, depth):
        super(DenseBlock, self).__init__()

        self.depth = depth
        self.conv = nn.ModuleList([nn.Conv2d(channels_in + i * growth_rate, growth_rate, 1, 1) for i in range(depth)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(growth_rate) for i in range(depth)])

    def forward(self, x):

        _x = [x]

        for d in range(self.depth):
            tmp = self.bn[d](F.relu(self.conv[d](torch.cat(_x, 1))))
            _x.append(tmp)
        return tmp


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        kernel_size = 3
        stride = 1
        filters = 32

        self.conv1 = nn.Conv2d(1, filters, kernel_size, stride)
        #self.bn1 = nn.BatchNorm2d(filters)
        _shape = convOutShape((22, 10), kernel_size, stride)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, stride)
        #self.bn2 = nn.BatchNorm2d(filters)
        _shape = convOutShape(_shape, kernel_size, stride)

        self.conv3 = nn.Conv2d(filters, filters, kernel_size, stride)
        #self.bn3 = nn.BatchNorm2d(filters)
        _shape = convOutShape(_shape, kernel_size, stride)

        self.conv4 = nn.Conv2d(filters, filters, kernel_size, stride)
        #self.bn4 = nn.BatchNorm2d(filters)
        _shape = convOutShape(_shape, kernel_size, stride)

        flat_in = _shape[0] * _shape[1] * filters

        n_fc1 = 128
        self.fc1 = nn.Linear(flat_in, n_fc1)
        flat_out = n_fc1

        #self.fc_p = nn.Linear(flat_out, n_actions)
        self.fc_v = nn.Linear(flat_out, 1)
        self.fc_var = nn.Linear(flat_out, 1)
        torch.nn.init.normal_(self.fc_v.bias, mean=1e3, std=.1)
        torch.nn.init.normal_(self.fc_var.bias, mean=10, std=.1)

    def forward(self, x):
        #x = self.bn1(F.relu(self.conv1(x)))
        #x = self.bn2(F.relu(self.conv2(x)))
        #x = self.bn3(F.relu(self.conv3(x)))
        #x = self.bn4(F.relu(self.conv4(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))

        #policy = F.softmax(self.fc_p(x), dim=1)
        policy = torch.ones((x.shape[0], 7)) / 7
        value = torch.abs(self.fc_v(x))
        #value = torch.exp(self.fc_v(x))
        #value = self.fc_v(x)
        var = torch.exp(self.fc_var(x))
        #var = torch.abs(self.fc_var(x)) + eps
        #var = F.elu(self.fc_var(x)) + 2
        return value, var, policy


def convert(x):
    return torch.from_numpy(x.astype(np.float32))


class Dataset(D.Dataset):
    def __init__(self, data):
        #states, values, variance, policy, weights
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return [d[index] for d in self.data]


class Model:
    def __init__(self, training=False, new=True, weighted=False, ewc=False, ewc_lambda=1, use_variance=True, use_policy=False, loss_type='mle', use_onnx=False, use_cuda=True):

        self.use_cuda = use_cuda

        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.training = training

        self.model = Net()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-3, amsgrad=True)
        #self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        #self.optimizer = optim.Adamax(self.model.parameters(), lr=1e-3)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        self.scheduler = None
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: max(0.9 ** step, 1e-2))

        self.v_mean = 0
        self.v_std = 1

        self.var_mean = 0
        self.var_std = 1

        self.weighted = weighted

        self.ewc = ewc
        self.ewc_lambda = ewc_lambda

        self.fisher = None

        self.use_policy = use_policy
        self.use_variance = use_variance

        self.loss_type = loss_type
        if loss_type == 'mae':
            self.l_func = lambda x, y: torch.abs(x-y)
        elif loss_type == 'mse':
            self.l_func = lambda x, y: (x - y) ** 2
        elif loss_type == 'mle' or loss_type == 'kldiv':
            self.l_func = lambda v_p, mu_p, v, mu: torch.log((v_p + eps) / (v + eps)) + (v + eps + (mu - mu_p) ** 2) / (v_p + eps) - 1

        self.use_onnx = use_onnx

    def _loss(self, batch):

        state, value, variance, policy, weight = batch

        if type(state) is not torch.Tensor:
            state = torch.from_numpy(batch[0])
            value = torch.from_numpy(batch[1])
            variance = torch.from_numpy(batch[2])
            policy = torch.from_numpy(batch[3])
            weight = torch.from_numpy(batch[4])

        if self.use_cuda:
            state = state.cuda()
            value = value.cuda()
            variance = variance.cuda()
            policy = policy.cuda()
            weight = weight.cuda()

        _v, _var, _p = self.model(state)
        if self.loss_type == 'mle':
            loss = (weight * self.l_func(_var, _v, variance, value)).mean()
            return loss, torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0])
        elif self.loss_type == 'kldiv':
            loss = self.l_func(_var, _v, variance, value).mean()
            return loss, torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0])
        else:
            loss_v = self.l_func(_v, value)

            if self.use_variance:
                loss_var = self.l_func(_var, variance)
            else:
                loss_var = torch.FloatTensor([0])

            if self.weighted:
                loss_v = weight * loss_v
                loss_var = weight * loss_var

            loss_v = loss_v.mean()
            loss_var = loss_var.mean()

            if self.use_policy:
                loss_p = F.kl_div(torch.log(_p), policy)
            else:
                loss_p = torch.FloatTensor([0])

            loss = loss_v + loss_var + loss_p

            return loss, loss_v, loss_var, loss_p

    def compute_ewc_loss(self):

        if self.fisher:
            ewc_loss = torch.tensor([0.], dtype=torch.float32, requires_grad=True, device=self.device)
            for i, p in enumerate(self.model.parameters()):
                ewc_loss = ewc_loss + 0.5 * self.ewc_lambda * torch.sum(self.fisher[i] * (p - self.p0[i]) ** 2)
            return ewc_loss
        else:
            return torch.tensor([0.], dtype=torch.float32, requires_grad=True, device=self.device)

    def compute_loss(self, batch, chunksize=2048):

        self.model.eval()

        result = []
        with torch.no_grad():
            for c in range(len(batch[0]) // chunksize + 1):
                b = [d[c * chunksize: (c + 1) * chunksize] for d in batch]
                losses = self._loss(b)
                r = [l.item() * len(b[0]) for l in losses]
                result.append(r)

        result = (np.sum(result, axis=0) / len(batch[0])).tolist()

        if self.ewc:
            result.append(self.compute_ewc_loss().item())
        else:
            result.append(0)

        return result

    def train(self, batch):

        self.model.train()

        self.optimizer.zero_grad()

        losses = self._loss(batch)

        if self.ewc:
            ewc_loss = self.compute_ewc_loss()
            l = losses[0] + ewc_loss
        else:
            l = losses[0]

        l.backward()
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

        result = [l.item() for l in losses]

        if self.ewc:
            result.append(ewc_loss.item())
            result[0] += ewc_loss.item()
        else:
            result.append(0)

        return result

    def train_dataset(self,
                      dataset=None,
                      batch_size=32,
                      iters_per_validation=1000,
                      early_stopping=True,
                      early_stopping_patience=10,
                      validation_fraction=0.1,
                      num_workers=2):

        validation_size = int(len(dataset[0]) * validation_fraction) + 1
        training_set = Dataset([d[:-validation_size] for d in dataset])
        training_loader = D.DataLoader(
                training_set,
                batch_size=batch_size,
                sampler=D.RandomSampler(training_set, replacement=True),
                pin_memory=True,
                num_workers=2)
        validation_set = Dataset([d[-validation_size:] for d in dataset])
        validation_loader = D.DataLoader(
                validation_set,
                batch_size=1024,
                num_workers=2)

        fail = 0
        #states, values, variance, policy, weights
        loss_avg = 0
        idx = 0
        while True:
            for batch in training_loader:
                idx += 1
                l = self.train(batch)
                loss_avg += l[0]
                if (idx + 1) % iters_per_validation == 0:

                    l_val = 0
                    """
                    for b in validation_loader:
                        l = self.compute_loss(b)
                        l_val += l[0] * len(b[0])
                    l_val /= validation_size
                    """
                    print(loss_avg/iters_per_validation, l_val)

    def get_fisher_from_adam(self):

        self.fisher = []
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                if 'exp_avg_sq' in self.optimizer.state[p]:
                    self.fisher.append(self.optimizer.state[p]['exp_avg_sq'])

    def compute_fisher(self, batch):

        self.model.train()

        fisher = [torch.zeros(p.data.shape) for p in self.model.parameters()]

        for i in range(len(batch[0])):

            self.optimizer.zero_grad()

            loss = self._loss([b[i:i+1] for b in batch])

            loss = loss[0] + self.compute_ewc_loss()

            loss.backward()

            for j, p in enumerate(self.model.parameters()):
                if p.grad is not None:
                    fisher[j] += torch.pow(p.grad.data, 2) / len(batch[0])

        self.fisher = fisher

    def inference(self, batch):

        self.model.eval()

        b = torch.from_numpy(batch).float()
        if self.use_cuda:
            b = b.cuda()

        with torch.no_grad():
            output = self.model(b)

        result = [o.cpu().numpy() for o in output]
        #result = self.model_caffe.run([batch.astype(np.float32)])

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

        filename = EXP_PATH + 'model_checkpoint'

        full_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'v_mean': self.v_mean,
                    'v_std': self.v_std,
                    'var_mean': self.var_mean,
                    'var_std': self.var_std,
                    'fisher': self.fisher,
                }

        if self.scheduler:
            full_state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(full_state, filename)

        if self.use_onnx:
            self.save_onnx()

    def save_onnx(self, verbose=False):

        dummy = torch.ones((1, 1, 22, 10))

        if not os.path.isdir(EXP_PATH):
            if verbose:
                print('Export path does not exist, creating a new one...', flush=True)
            os.mkdir(EXP_PATH)

        torch.onnx.export(self.model, dummy, EXP_PATH + 'model.onnx')

        output_arg = '-o ' + EXP_PATH + 'pred.pb'
        init_arg = '--init-net-output ' + EXP_PATH + 'init.pb'
        file_arg = EXP_PATH + 'model.onnx'

        subprocess.run('convert-onnx-to-caffe2' + ' ' + output_arg + ' ' + init_arg + ' ' + file_arg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def load(self, filename=EXP_PATH + 'model_checkpoint'):

        #filename = EXP_PATH + 'model_checkpoint'

        if os.path.isfile(filename):
            print('Loading model...', flush=True)
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.v_mean = checkpoint['v_mean']
            self.v_std = checkpoint['v_std']
            self.var_mean = checkpoint['var_mean']
            self.var_std = checkpoint['var_std']
            self.fisher = checkpoint['fisher']
            self.p0 = [p.clone() for p in self.model.parameters()]
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print('Checkpoint not found, using default model', flush=True)

        if not self.training and self.use_onnx:
            self.load_onnx()

    def load_onnx(self):

        init_filename = EXP_PATH + 'init.pb'
        pred_filename = EXP_PATH + 'pred.pb'

        print('Loading ONNX model...', flush=True)

        if not (os.path.isfile(init_filename) or os.path.isfile(pred_filename)):
            self.save_onnx()

        with open(init_filename, mode='r+b') as f:
            init_net = f.read()

        with open(pred_filename, mode='r+b') as f:
            pred_net = f.read()

        self.model_caffe = workspace.Predictor(init_net, pred_net)
