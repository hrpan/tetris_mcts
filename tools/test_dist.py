import model.model_distributional as M
import torch
import numpy as np
atoms=50
#l = torch.load('pytorch_model/model_checkpoint', map_location=torch.device('cpu'))
#atoms = l['model_state_dict']['seq.fc_v.bias'].shape[0]
np.set_printoptions(suppress=True, threshold=100000, precision=20)
m = M.Model(use_cuda=False, atoms=atoms)
m.load()
b = np.random.randint(0, 2, (18, 1, 22, 10))
for i in range(18):
    b[i,:,0:i+4,:] = 0
b[:,:,1,5:7] = -1
b[:,:,2,5:7] = -1
#print(b[:,0,:,:])
print(b[0])
y = np.array(m.inference(b))
y1 = (y * np.arange(atoms)).sum(axis=1)
y2 = (y * np.arange(atoms)**2).sum(axis=1)
print(y[0])
print(y[-1])
print(y1)
print((y2 - y1 ** 2) ** 0.5)
print((y * np.log(y)).sum(1))
#print(y_flip[0])
#print(y_flip[1])
b = np.random.randint(0, 2, (1000, 1, 22, 10))
b[:,0,0:5,:] = 0
b_flip = np.flip(b, axis=3).copy()
y = np.array(m.inference(b))
y1 = (y * np.arange(atoms)).sum(axis=1)
y_flip = np.array(m.inference(b_flip))
y_flip1 = (y_flip * np.arange(atoms)).sum(axis=1)
print(np.average(y1 - y_flip1))

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
        if 'sum' in state:
            print('sum', state['sum'].mean(), state['sum'].std())
        if 'exp_inf' in state:
            exp_avg = state['exp_inf']
            print(exp_avg.mean(), exp_avg.std())
        if 'momentum_buffer' in state:
            tmp = state['momentum_buffer']
            print(tmp.mean(), tmp.std())
