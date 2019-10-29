import model.model_pytorch as M
import torch
import numpy as np
np.set_printoptions(suppress=True, threshold=100000, precision=20)
m = M.Model(use_cuda=False)
m.load()
b = np.random.randint(0, 2, (20, 1, 22, 10))
for i in range(20):
    b[i,:,0:i+2,:] = 0
b[:,:,1,5:7] = -1
b[:,:,2,5:7] = -1
#print(b[:,0,:,:])
print(m.model.fc_v.bias)
print(m.model.fc_var.bias)
y = m.inference(b)
print(y[0])
print(y[1])
print('lr', m.optimizer.param_groups[0]['lr'])
for g in m.optimizer.param_groups:
    for p in g['params']:
        
        state = m.optimizer.state[p]
        if len(state) == 0:
            continue
        if 'exp_avg_sq' in state:
            exp_avg = state['exp_avg_sq']
            if 'max_exp_avg_sq' in state:
                max_exp_avg = state['max_exp_avg_sq']
                print('exp_avg',exp_avg.mean(), exp_avg.std(), 'max', max_exp_avg.mean(), max_exp_avg.std())
            else:
                print('exp_avg',exp_avg.mean(), exp_avg.std())
