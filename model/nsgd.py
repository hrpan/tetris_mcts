import torch
from torch.optim import Optimizer


class NSGD(Optimizer):
    def __init__(self, params, lr=0.01, weight_decay=0, norm_decay=0.999):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, norm_decay=norm_decay)
        super(NSGD, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            norm_decay = group['norm_decay']
            norm = group.get('norm', 0) 
            norm_tmp = 0.
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                norm_tmp = max(norm_tmp, torch.max(d_p))
            norm = max(norm_decay * norm, norm_tmp)

            group['norm'] = norm

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(-group['lr'] / group['norm'], p.grad.data)

        return loss
