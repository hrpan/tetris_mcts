import numpy as np
from model.model_vv import Model_VV as Model


data = np.load('data/dump.npz')

m = Model(training=False)

states = data['states']
values = data['values']
variance = data['variance']
weights = data['weights']
print(states.shape)

m.train_data([states, values, variance, weights])
