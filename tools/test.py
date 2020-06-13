#from model.model_vp import Model_VP as M
from model.model_vv import Model_VV as M
import torch
import numpy as np

np.set_printoptions(suppress=True, threshold=100000, precision=3)

m = M(use_cuda=False)
m.load()

pars = 0
for p in m.model.parameters():
    pars += np.prod(p.size())

print('Parameters: {}'.format(pars))

print(m.model.out_ubound)
#print(m.model.head.norm_out.bias)
#print(m.model.head.norm_out.weight)
print(m.model.head.fc_out.bias)
#print('value mean/std: {:.3f}/{:.3f}'.format(m.model.value_mean.item(), m.model.value_std.item()))
#print('variance mean/std: {:.3f}/{:.3f}'.format(m.model.variance_mean.item(), m.model.variance_std.item()))
b = np.random.randint(0, 2, (18, 1, 20, 10))

for i in range(18):
    b[i,:,0:i+2,:] = 0
b[:,:,1,5:7] = -1
b[:,:,2,5:7] = -1

y = m.inference(b)

print(y[0])
print(y[1])

b = np.random.randint(0, 2, (1000, 1, 20, 10))
b[:,0,0:5,:] = 0
b_flip = np.flip(b, axis=3).copy()
y = m.inference(b)
y_flip = m.inference(b_flip)
print(np.sqrt(np.average((y[0] - y_flip[0]) ** 2)))
print(np.sqrt(np.average((y[1] - y_flip[1]) ** 2)))

print('lr', m.optimizer.param_groups[0]['lr'])
keys = ['avg', 'avg_sq', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', 'exp_avg_sq_last', 'momentum_buffer', 'exp_inf', 'square_avg']
for g in m.optimizer.param_groups:
    for p in g['params']:
        
        state = m.optimizer.state[p]
        if len(state) == 0:
            continue
        for k in keys:
            if k in state:
                print('{} {:10.3e} {:10.3e} {:10.3e} {:10.3e}'.format(k, state[k].mean().item(), state[k].std().item(), state[k].min().item(), state[k].max().item()), end='  ')

        print()
