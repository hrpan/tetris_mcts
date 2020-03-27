import numpy as np
import model.model_pytorch as M


data = np.load('data/dump.npz')

m = M.Model()

batch_size = 128

states = data['states']
values = data['values']
variance = data['variance']
weights = data['weights']
print(states.shape)
weights = weights / weights.mean()

v_size = int(len(states) * 0.1)
dummy = np.empty(1)
batch_val = [states[-v_size:], values[-v_size:], variance[-v_size:], dummy, weights[-v_size:]]

iters = 100000
min_vloss = float('inf')
for i in range(iters):
    idx = np.random.choice(int(len(states) - v_size), size=batch_size)
    batch = [states[idx], values[idx], variance[idx], dummy, weights[idx]]
    
    loss = m.train(batch)
    if i % 100 == 0:
        loss_v = m.compute_loss(batch_val)
        print('{:5d} {:.5f}, {:.5f}'.format(i, loss['loss'], loss_v['loss']))
        if loss_v['loss'] < min_vloss:
            min_vloss = loss_v['loss']
            m.save()
