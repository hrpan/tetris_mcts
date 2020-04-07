import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch_optimizer as toptim
from collections import defaultdict, OrderedDict
from model.model import convOutShape, Model


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

        self.head = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, filters, kernel_size, stride, bias=bias)),
                ('norm1', nn.BatchNorm2d(filters)),
                ('act1', self.activation),
                ('conv2', nn.Conv2d(filters, filters, kernel_size, stride, bias=bias)),
                ('norm2', nn.BatchNorm2d(filters)),
                ('act2', self.activation),
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(flat_in, 128)),
                ('act4', self.activation),
                ('fc_out', nn.Linear(128, 2)),
                #('act_out', nn.Softplus()),
            ]))

        self.head.fc_out.bias.data[0] = 1e2
        self.head.fc_out.bias.data[1] = 1e3

        self.eps = nn.Parameter(torch.tensor([eps]), requires_grad=False)

    def forward(self, x):

        x = self.head(x)

        value, variance = x.split(1, dim=1)

        return value, F.softplus(variance).add(self.eps)


class Ensemble(nn.Module):

    def __init__(self, n_models=5):

        super(Ensemble, self).__init__()

        self.n_models = n_models

        self.nets = nn.ModuleList([Net() for i in range(n_models)])

    def forward(self, x):

        if self.training:
            m = torch.randint(0, self.n_models-1, (1,))
            return self.nets[m](x)
        else:
            results = [net(x) for net in self.nets]
            value = torch.mean(torch.stack([r[0] for r in results]), dim=0)
            var = torch.mean(torch.stack([r[1] for r in results]), dim=0)
            return value, var


class Model_VV(Model):
    def __init__(self, weighted=False, ewc=False, ewc_lambda=1, loss_type='mle', **kwarg):
        super().__init__(**kwarg)

        self.weighted = weighted

        self.ewc = ewc
        self.ewc_lambda = ewc_lambda

        self.fisher = None

        self.loss_type = loss_type
        if loss_type == 'mae':
            self.l_func = lambda x, y: torch.abs(x-y)
        elif loss_type == 'mse':
            self.l_func = lambda x, y: (x - y) ** 2
        elif loss_type == 'kldiv' or loss_type == 'mle':
            self.l_func = lambda v_p, mu_p, v, mu: torch.log(v_p) + (v + (mu - mu_p) ** 2) / v_p - torch.log(v) - 1

    def _init_model(self):

        self.model = Net()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model = torch.jit.script(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-3, weight_decay=1e-4)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.95, nesterov=True)

        self.scheduler = None

    def _loss(self, batch, variance_clip=1.0):

        batch_tensor = [torch.as_tensor(b, dtype=torch.float, device=self.device) for b in batch]

        state, value, variance, weight = batch_tensor

        variance.clamp_(min=variance_clip)

        _v, _var = self.model(state)
        if self.loss_type == 'mle':
            loss = torch.std_mean(weight * self.l_func(_var, _v, variance, value))
            return defaultdict(float, loss=loss[1], loss_std=loss[0])
        elif self.loss_type == 'kldiv':
            loss = torch.std_mean(self.l_func(_var, _v, variance, value))
            return defaultdict(float, loss=loss[1], loss_std=loss[0])
        else:
            loss_v = self.l_func(_v, value)

            loss_var = self.l_func(_var, variance)

            if self.weighted:
                loss_v = weight * loss_v
                loss_var = weight * loss_var

            loss_v = loss_v.mean()
            loss_var = loss_var.mean()

            loss = loss_v + loss_var

            return defaultdict(float, loss=loss, loss_v=loss_v, loss_var=loss_var)

    def compute_ewc_loss(self):

        if self.fisher:
            ewc_loss = torch.tensor([0.], dtype=torch.float32, requires_grad=True, device=self.device)
            for i, p in enumerate(self.model.parameters()):
                ewc_loss = ewc_loss + 0.5 * self.ewc_lambda * torch.sum(self.fisher[i] * (p - self.p0[i]) ** 2)
            return ewc_loss
        else:
            return torch.tensor([0.], dtype=torch.float32, requires_grad=True, device=self.device)


#    def train_dataset(self,
#                      dataset=None,
#                      batch_size=128,
#                      iters_per_validation=100,
#                      early_stopping=True,
#                      early_stopping_patience=10,
#                      validation_fraction=0.1,
#                      num_workers=2):
#
#        validation_size = int(len(dataset[0]) * validation_fraction) + 1
#        training_set = Dataset([d[:-validation_size] for d in dataset])
#        training_loader = D.DataLoader(
#                training_set,
#                batch_size=batch_size,
#                sampler=D.RandomSampler(training_set, replacement=True),
#                pin_memory=True,
#                num_workers=2)
#        validation_set = Dataset([d[-validation_size:] for d in dataset])
#        validation_loader = D.DataLoader(
#                validation_set,
#                batch_size=1024,
#                num_workers=2)
#
#        fail = 0
#        #states, values, variance, policy, weights
#        loss_avg = 0
#        idx = 0
#        while True:
#            for batch in training_loader:
#                idx += 1
#                l = self.train(batch)
#                loss_avg += l[0]
#                if (idx + 1) % iters_per_validation == 0:
#
#                    l_val = 0
#                    """
#                    for b in validation_loader:
#                        l = self.compute_loss(b)
#                        l_val += l[0] * len(b[0])
#                    l_val /= validation_size
#                    """
#                    print(loss_avg/iters_per_validation, l_val)

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

    def inference_stochastic(self, batch):

        result = self.inference(batch)

        result[0] = np.random.normal(result[0], np.sqrt(result[1]))

        return result

#    def save_onnx(self, verbose=False):
#
#        dummy = torch.ones((1, 1, 22, 10))
#
#        if not os.path.isdir(EXP_PATH):
#            if verbose:
#                print('Export path does not exist, creating a new one...', flush=True)
#            os.mkdir(EXP_PATH)
#
#        torch.onnx.export(self.model, dummy, EXP_PATH + 'model.onnx')
#
#        output_arg = '-o ' + EXP_PATH + 'pred.pb'
#        init_arg = '--init-net-output ' + EXP_PATH + 'init.pb'
#        file_arg = EXP_PATH + 'model.onnx'
#
#        subprocess.run('convert-onnx-to-caffe2' + ' ' + output_arg + ' ' + init_arg + ' ' + file_arg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
#    def load_onnx(self):
#
#        init_filename = EXP_PATH + 'init.pb'
#        pred_filename = EXP_PATH + 'pred.pb'
#
#        print('Loading ONNX model...', flush=True)
#
#        if not (os.path.isfile(init_filename) or os.path.isfile(pred_filename)):
#            self.save_onnx()
#
#        with open(init_filename, mode='r+b') as f:
#            init_net = f.read()
#
#        with open(pred_filename, mode='r+b') as f:
#            pred_net = f.read()
#
#        self.model_caffe = workspace.Predictor(init_net, pred_net)
