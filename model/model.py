import torch
import torch.utils.data as D
import torch.nn.utils as U
import numpy as np
import os
from sys import stderr
from collections import defaultdict

n_actions = 7

EXP_PATH = './pytorch_model/'

perr = dict(file=stderr, flush=True)


def convOutShape(shape_in, kernel_size, stride):

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    return ((shape_in[0] - kernel_size[0]) // stride[0] + 1,
            (shape_in[1] - kernel_size[1]) // stride[1] + 1)


class Dataset(D.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return [d[index] for d in self.data]


class Model:
    def __init__(self, training=False, new=True, use_cuda=True, **kwarg):

        self.use_cuda = use_cuda

        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._init_model()

        self.training(training)

    def compute_loss(self, batch, chunksize=512):

        _tmp = defaultdict(list)
        d_size = len(batch[0])
        with torch.no_grad():
            for c in range(0, d_size, chunksize):
                b = [d[c:c+chunksize] for d in batch]
                loss = self._loss(b)
                _tmp['bsize'].append(len(b[0]))
                for k, v in loss.items():
                    _tmp[k].append(v.item())

        result = {}
        bsize = np.array(_tmp['bsize'])
        for k, v in _tmp.items():
            if 'std' in k:
                continue
            loss = np.array(_tmp[k])
            loss_combined = np.sum(loss * bsize) / d_size
            result[k] = loss_combined

            k_std = k + '_std'
            if k_std in _tmp:
                loss_std = np.array(_tmp[k_std])
                loss_sq = (bsize - 1) * loss_std ** 2 / bsize + loss ** 2
                loss_sq_combined = np.sum(loss_sq * bsize) / d_size
                loss_std_combined = ((loss_sq_combined - loss_combined ** 2) * d_size / (d_size - 1)) ** 0.5
                result[k_std] = loss_std_combined

        return result

    def compute_gradient_norm(self, p=2):

        norm = 0.
        for _p in self.model.parameters():
            if _p.grad is None:
                continue
            norm += _p.grad.data.norm(p).item() ** p

        return norm ** (1/p)

    def train(self, batch, grad_clip=0):

        self.optimizer.zero_grad()

        loss = self._loss(batch)

        loss['loss'].backward()

        if grad_clip > 0:
            U.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()

        result = {k: v.item() for k, v in loss.items()}

        return result

    def training(self, mode=True):

        if mode:
            self.model.train()
        else:
            self.model.eval()

    def inference(self, batch):

        b = torch.as_tensor(batch, dtype=torch.float, device=self.device)

        with torch.no_grad():
            output = self.model(b)

        return [o.cpu().numpy() for o in output]

    def reset_optimizer(self):
        self.optimizer.state = defaultdict(dict)

    def update_scheduler(self, **kwarg):
        if self.scheduler:
            self.scheduler.step(**kwarg)

    def save(self, filename=EXP_PATH + 'model_checkpoint', verbose=True):
        if verbose:
            print('Saving model...', flush=True)

        if not os.path.isdir(EXP_PATH):
            if verbose:
                print('Export path does not exist, creating a new one...', flush=True)
            os.mkdir(EXP_PATH)

        full_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

        if self.scheduler:
            full_state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(full_state, filename)

    def load(self, filename=EXP_PATH + 'model_checkpoint'):

        if os.path.isfile(filename):
            print('Loading model...', flush=True)
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #self.fisher = checkpoint['fisher']
            self.p0 = [p.clone() for p in self.model.parameters()]
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print('Checkpoint not found, using default model', flush=True)

    def train_data(self, data, batch_size=128, iters_per_val=500, validation_fraction=0.1,
                   early_stopping=True, early_stopping_patience=10,
                   early_stopping_threshold=1., max_iters=1000000):

        data_size = len(data[0])
        validation_size = int(data_size * validation_fraction)

        weights = data[-1] / data[-1].mean()

        data[-1] = weights

        batch_training = [d[:-validation_size] for d in data]
        batch_validation = [d[-validation_size:] for d in data]

        print('Training data size: {}    Validation data size: {}'.format(data_size-validation_size, validation_size), **perr)

        if early_stopping:
            fails = 0
            loss_val_min = float('inf')

        self.training(True)
        for iters in range(max_iters):
            b_idx = np.random.choice(data_size-validation_size, size=batch_size)
            batch = [b[b_idx] for b in batch_training]
            loss = self.train(batch)

            if iters % iters_per_val == 0:
                self.training(False)
                loss_val = self.compute_loss(batch_validation)
                self.training(True)
                loss_val_mean, loss_val_std = loss_val['loss'], loss_val['loss_std']
                loss_val_std /= validation_size ** 0.5

                if early_stopping:
                    if loss_val_mean - loss_val_min < loss_val_std * early_stopping_threshold:
                        fails = 0
                        if loss_val_mean < loss_val_min:
                            self.save(verbose=False)
                            loss_val_min = loss_val_mean
                    else:
                        fails += 1
                print('Iteration:{:7d}  training loss:{:.3f}  validation loss:{:.3f}±{:.3f}'
                      .format(iters, loss['loss'], loss_val_mean, loss_val_std), **perr)

        if early_stopping:
            self.load()
        else:
            self.save()

        self.training(False)

    def _init_model(self):
        raise NotImplementedError('_init_model not implemented')

    def _loss(self, batch):
        raise NotImplementedError('_loss not implemented')
