import torch
from torch.optim.optimizer import Optimizer

class SNGD(Optimizer):
    """
    Implements Stochastic Normalized Gradient Descent (SNGD)
    from https://arxiv.org/abs/1507.02030
    """
    def __init__(self, params, lr=0.01, eps=1e-8):

        super(SNGD, self).__init__(params, defaults=dict(lr=lr, eps=eps))


    def step(self, closure=None):

        g_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                g_norm += (p.grad.data ** 2).sum()

        g_norm = g_norm ** 0.5
        
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                p.data.add_(-lr / (g_norm + eps), p.grad.data)
