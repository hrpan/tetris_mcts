import math
import torch
from torch.optim.optimizer import Optimizer


class Yogi(Optimizer):
    """
    Yogi Optimizer Algorithm.
        Paper: https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
        Reference code: https://github.com/jettify/pytorch-optimizer
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-3, weight_decay=0):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(Yogi, self).__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Yogi does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = grad.mul(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                grad_squared = grad.mul(grad)

                exp_avg_sq.addcmul_(
                    -(1 - beta2),
                    torch.sign(exp_avg_sq - grad_squared),
                    grad_squared,
                )

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps']
                )
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
