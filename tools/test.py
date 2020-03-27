import model.model_pytorch as M
import torch
import numpy as np

np.set_printoptions(suppress=True, threshold=100000, precision=20)

m = M.Model(use_cuda=False)
m.load()

b = np.random.randint(0, 2, (18, 1, 22, 10))

for i in range(18):
    b[i,:,0:i+4,:] = 0
b[:,:,1,5:7] = -1
b[:,:,2,5:7] = -1

print(m.model.fc_v.bias)
print(m.model.fc_var.bias)

y = m.inference(b)

print(y[0])
print(y[1])

b = np.random.randint(0, 2, (1000, 1, 22, 10))
b[:,0,0:5,:] = 0
b_flip = np.flip(b, axis=3).copy()
y = m.inference(b)
y_flip = m.inference(b_flip)
print(np.average((y[0] - y_flip[0]) ** 2))
print(np.average((y[1] - y_flip[1]) ** 2))

print('lr', m.optimizer.param_groups[0]['lr'])
keys = ['exp_avg_sq', 'max_exp_avg_sq', 'exp_avg_sq_last', 'momentum_buffer', 'exp_inf']
for g in m.optimizer.param_groups:
    for p in g['params']:
        
        state = m.optimizer.state[p]
        if len(state) == 0:
            continue
        for k in keys:
            if k in state:
                print('{} {:10.3e} {:10.3e}'.format(k, state[k].mean().item(), state[k].std().item()), end='    ')
        print()
