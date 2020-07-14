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

    def compute_loss(self, batch, weighted, chunksize=1024):

        _tmp = defaultdict(list)
        d_size = 0.
        with torch.no_grad():
            for c in range(0, len(batch[0]), chunksize):
                b = [d[c:c+chunksize] for d in batch]
                loss = self._loss(b, weighted=weighted)
                if weighted:
                    _tmp['bsize'].append(np.sum(b[-1]))
                else:
                    _tmp['bsize'].append(len(b[0]))
                d_size += _tmp['bsize'][-1]
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
                np.nan_to_num(loss_std, copy=False)
                loss_std_combined = np.sqrt(np.sum(bsize * loss_std ** 2 + bsize * (loss ** 2 - loss_combined ** 2)) / d_size)
                result[k_std] = loss_std_combined

        return result

    def compute_gradient_norm(self, p=2):

        norm = 0.
        for _p in self.model.parameters():
            if _p.grad is None:
                continue
            norm += _p.grad.data.norm(p).item() ** p

        return norm ** (1/p)

    def train(self, batch, grad_clip=0., g_norm_warn=1e3, weighted=False):

        self.optimizer.zero_grad()

        loss = self._loss(batch, weighted=weighted)

        loss['loss'].backward()

        g_norm = self.compute_gradient_norm()
        if g_norm > g_norm_warn:
            print('Large gradient ({}) detected'.format(g_norm), **perr)
            _tmp = self.inference(batch[0])
            np.savez('data/dump_grad', *batch, *_tmp)

        if grad_clip > 0:
            U.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()

        result = {k: v.item() for k, v in loss.items()}
        result['grad_norm'] = g_norm
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
                   sample_replacement=True, oversampling=False, weighted=True,
                   early_stopping=True, early_stopping_patience=10, early_stopping_threshold=1.,
                   shuffle=False, max_iters=100000):

        data_size = len(data[0])
        validation_size = int(data_size * validation_fraction)

        weights = data[-1] / data[-1].mean()
        data[-1] = weights

        if shuffle:
            idx = np.random.permutation(data_size)
            data = [d[idx] for d in data]

        batch_training = [d[:-validation_size] for d in data]
        batch_validation = [d[-validation_size:] for d in data]

        if oversampling:
            p = np.squeeze(batch_training[-1] / batch_training[-1].sum())
        else:
            p = None

        print('Training data size: {}    Validation data size: {}'.format(data_size-validation_size, validation_size), **perr)

        if early_stopping:
            fails = 0
            loss_val_min = float('inf')

        loss_avg = g_norm_avg = 0
        self.training(True)
        for iters in range(max_iters):
            b_idx = np.random.choice(data_size-validation_size, size=batch_size, replace=sample_replacement, p=p)
            batch = [b[b_idx] for b in batch_training]
            loss = self.train(batch, weighted=weighted)

            loss_avg += loss['loss']
            g_norm_avg += loss['grad_norm']
            if (iters + 1) % iters_per_val == 0:
                self.training(False)
                loss_val = self.compute_loss(batch_validation, weighted=weighted)
                self.training(True)
                loss_val_mean, loss_val_std = loss_val['loss'], loss_val['loss_std']
                loss_val_std /= validation_size ** 0.5

                suffix = ''
                if early_stopping:
                    if loss_val_mean - loss_val_min < loss_val_std * early_stopping_threshold:
                        fails = 0
                        if loss_val_mean < loss_val_min:
                            suffix = '*'
                            self.save(verbose=False)
                            loss_val_min = loss_val_mean
                    else:
                        fails += 1

                        if fails >= early_stopping_patience:
                            break

                print('Iteration:{:7d}  training loss:{:6.4f}  validation loss:{:6.4f}±{:6.4f}  gradient norm:{:6.3f}    {}'
                      .format(iters+1, loss_avg / iters_per_val, loss_val_mean, loss_val_std, g_norm_avg / iters_per_val, suffix), **perr)

                loss_avg = g_norm_avg = 0

        if early_stopping:
            self.load()
            if hasattr(self.optimizer, 'finalize'):
                self.optimizer.finalize()
        else:
            if hasattr(self.optimizer, 'finalize'):
                self.optimizer.finalize()
            self.save()

        self.training(False)

    def _init_model(self):
        raise NotImplementedError('_init_model not implemented')

    def _loss(self, batch):
        raise NotImplementedError('_loss not implemented')
