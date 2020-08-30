import numpy as np
from model.model_vv import Model_VV as Model


data = np.load('data/dump.npz')

m = Model(training=True)

states = data['states']
values = data['values']
variance = data['variance']
weights = data['weights']
print(states.shape)

_data = [states, values, variance, weights]

m.train_data(_data, batch_size=1024, max_iters=50000, iters_per_val=100, early_stopping_patience=100, early_stopping=True, validation_fraction=0.1, oversampling=False, weighted=True)
