import torch
from torch.optim.optimizer import Optimizer
from torch.nn.functional import softplus

class BBB(Optimizer):
    """Implements Bayes-by-backprop.
    Code mostly taken from https://github.com/igolan/bgd
    Usage:
    for samples, labels in batches:
        for mc_iter in range(mc_iters):
            optimizer.set_weights()
            output = model.forward(samples)
            loss = cirterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.aggregate_grads()
        optimizer.step()
    """
    def __init__(self, params, lr=0.01, beta=0.99, rho_init=-3., _eps=1e-3):
        """
        Initialization of BGD optimizer
        group["mean_param"] is the learned mean.
        group["std_param"] is the learned STD.
        :param params: List of model parameters
        """
        
        super(BBB, self).__init__(params, defaults=dict(lr=lr, beta=beta, _eps=_eps))
        # Initialize mu (mean_param) and sigma (std_param)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['rho_prior'] = torch.ones_like(p.data) 
                state['mu_prior'] = p.data.clone()
                state['rho'] = torch.ones_like(p.data) * rho_init
                state['mu'] = p.data.clone()
                state['eps'] = torch.empty_like(p.data)
                state['mu_g_sq'] = torch.ones_like(p.data)
                state['rho_g_sq'] = torch.ones_like(p.data)
        self._init_accumulators()
        
    def _init_accumulators(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['eps'] = None
                state['grad_mu'] = torch.zeros_like(p.data)
                state['grad_rho'] = torch.zeros_like(p.data)
                state['mc_iters'] = 0

    def set_weights(self, maxap=False):
        """
        Randomize the weights according to N(mean, std).
        :param force_std: If force_std>=0 then force_std is used for STD instead of the learned STD.
        :return: None
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if maxap:
                    p.data.copy_(state['mu'])
                else:
                    state['eps'] = torch.normal(torch.zeros_like(p.data), torch.ones_like(p.data))
                    std = softplus(state['rho'])
                
                    w = state['mu'] + state['eps'].mul(std)
                    p.data.copy_(w)

    def aggregate_grads(self, batch_size):
        """
        Aggregates a single Monte Carlo iteration gradients. Used in step() for the expectations calculations.
        optimizer.zero_grad() should be used before calling .backward() once again.
        :param batch_size: BGD is using non-normalized gradients, but PyTorch gives normalized gradients.
                            Therefore, we multiply the gradients by the batch size.
        :return: None
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                assert state['eps'] is not None, "Must randomize weights before using aggregate_grads"
                state['mc_iters'] += 1
                w = p.data
                mu = state['mu'].data
                mu_p = state['mu_prior'].data
                rho = state['rho'].data
                rho_p = state['rho_prior'].data
                std = softplus(rho)
                std_p = softplus(rho_p)
                state['grad_mu'] += p.grad.data * batch_size + (w - mu_p) / (std_p ** 2) 
                state['grad_rho'] += ((-1.0 / std) + std / (std_p ** 2) + p.grad.data * batch_size * state['eps'].data) / (1 + torch.exp(-rho))
                state['eps'] = None

    def update_prior(self):
        for group in self.param_groups:
            for p in group['params']:
                state['mu_prior'].copy_(state['mu'])
                state['rho_prior'].copy_(state['rho'])

    def step(self, closure=None):
        """
        Updates the learned mean and STD.
        :return:
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['mc_iters'] <= 0:
                    continue
                
                lr = group['lr']
                beta = group['beta']
                _eps = group['_eps']

                state['mu_g_sq'] = state['mu_g_sq'] * beta + (state['grad_mu'] ** 2) * (1 - beta)
                dmu = -lr * state['grad_mu'] / state['mc_iters'] / (state['mu_g_sq'] + _eps)
                state['mu'].data += dmu

                state['rho_g_sq'] = state['rho_g_sq'] * beta + (state['grad_rho'] ** 2) * (1 - beta)
                drho = -lr * state['grad_rho'] / state['mc_iters'] / (state['rho_g_sq'] + _eps)
                state['rho'].data += drho
                #print(state['grad_mu'].max() * group['lr'] / state['mc_iters'])
                #state['mu'].data.add_(-group['lr'] / state['mc_iters'], state['grad_mu'])
                #print(state['grad_rho'].max() * group['lr'] / state['mc_iters'])
                #state['rho'].data.add_(-group['lr'] / state['mc_iters'], state['grad_rho'])
        self._init_accumulators()

