import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, OrderedDict
from model.model import convOutShape, Model
from model.yogi import Yogi

variance_bound = 1.


class Net(nn.Module):

    def __init__(self, input_shape=(20, 10), eps=variance_bound):
        super().__init__()

        kernel_size = 3
        stride = 1
        filters = 32
        bias = True

        self.activation = nn.LeakyReLU(inplace=True)
        #self.activation = nn.ReLU(inplace=True)

        _shape = convOutShape(input_shape, kernel_size, stride)
        _shape = convOutShape(_shape, kernel_size, stride)
        flat_in = _shape[0] * _shape[1] * filters

        n_fc = 128
        self.head = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, filters, kernel_size, stride, bias=bias)),
                ('act1', self.activation),
                ('conv2', nn.Conv2d(filters, filters, kernel_size, stride, bias=bias)),
                ('act2', self.activation),
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(flat_in, n_fc)),
                ('fc_act1', self.activation),
                ('fc_out', nn.Linear(n_fc, 2)),
                ('act_out', nn.Sigmoid()),
            ]))

        #self.head.fc_out.bias.data[0].fill_(1e2)
        #self.head.fc_out.bias.data[1].fill_(1e4)

        self.out_ubound = nn.Parameter(torch.tensor([1e2, 1e3]), requires_grad=False)
        self.out_lbound = nn.Parameter(torch.tensor([0, eps]), requires_grad=False)

    def forward(self, x):

        x = self.head(x)
        x = x * self.out_ubound + self.out_lbound
        #variance = variance.exp().add(self.eps)
        #variance = variance.add(self.eps)
        #variance = variance.pow(2).add(self.eps)
        #variance = F.softplus(variance).add(self.eps)
        return x


class Ensemble(nn.Module):

    def __init__(self, n_models=5):

        super().__init__()

        self.n_models = n_models

        self.nets = nn.ModuleList([Net() for i in range(n_models)])

    def forward(self, x):

        if self.training:
            m = torch.randint(0, self.n_models-1, (1,))
            return self.nets[n](x)
        else:
            r = [n(x) for n in self.nets]
            return torch.mean(torch.stack(r))


class WeakGaussianLL(nn.Module):

    def __init__(self, sigma=3.):
        super().__init__()

        self.sigma = sigma

    def forward(self, var_pred, mean_pred, var, mean):
        diff = (mean - mean_pred).abs()

        threshold = self.sigma * var_pred.sqrt()

        vloss = torch.where(diff < threshold, diff ** 2, 2 * threshold * diff - threshold ** 2)

        logl = var_pred.log() + (var + vloss) / var_pred - var.log() - 1

        return logl


class GaussianLL(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, var_pred, mean_pred, var, mean):
        logl = var_pred.log().add_((mean - mean_pred).pow_(2).add_(var).div_(var_pred)).add_(-1, var.log()).add(-1)
        return logl


class Model_VV(Model):
    def __init__(self, weighted=False, ewc=False, ewc_lambda=1, loss_type='kldiv', **kwarg):
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
            self.l_func = GaussianLL()
        elif loss_type == 'mle_approx':
            self.l_func = lambda v_p, mu_p, v, mu: (1 - v_p / v) ** 2 + 2 * (mu - mu_p) ** 2 / v

    def _init_model(self):

        self.model = Net()
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = torch.jit.script(self.model)

        #self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, eps=1e-3)
        self.optimizer = Yogi(self.model.parameters(), lr=1e-3, eps=1e-3)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.95, nesterov=True)

        self.scheduler = None

    def _loss(self, batch, variance_clip=variance_bound, weighted=False):

        batch_tensor = [torch.as_tensor(b, dtype=torch.float, device=self.device) for b in batch]

        state, value, variance, weight = batch_tensor

        variance.clamp_(min=variance_clip)

        _v, _var = self.model(state).split(1, dim=1)
        if self.loss_type == 'mle' or self.loss_type == 'kldiv':
            if weighted:
                std, mean = torch.std_mean(weight * self.l_func(_var, _v, variance, value))
            else:
                std, mean = torch.std_mean(self.l_func(_var, _v, variance, value))
            return defaultdict(float, loss=mean, loss_std=std)
        elif self.loss_type == 'mle_approx':
            std, mean = torch.std_mean(weight * self.l_func(_var, _v, variance, value))
            return defaultdict(float, loss=mean, loss_std=std)
        else:
            loss_v = self.l_func(_v, value)

            loss_var = self.l_func(_var, variance)

            if weighted:
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

        b = torch.as_tensor(batch, dtype=torch.float, device=self.device)

        with torch.no_grad():
            output = self.model(b).cpu().split(1, dim=1)

        return [o.numpy() for o in output]

    def inference_stochastic(self, batch):

        result = self.inference(batch)

        result[0] = np.random.normal(result[0], np.sqrt(result[1]))

        return result

    def train_data(self, data, **kwargs):
        d_bound = torch.tensor([data[1].max(), data[2].max()], device=self.device)
        self.model.out_ubound = torch.max(self.model.out_ubound, d_bound)

        super().train_data(data, **kwargs)

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
