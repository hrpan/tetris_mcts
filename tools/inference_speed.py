from model.model_vv import Model_VV as Model
import numpy as np
from tqdm import tqdm 

batch_size = 5
m = Model(training=False)
b = np.ones((batch_size, 1, 20, 10))
for i in tqdm(range(10000)):
    m.inference(b)
