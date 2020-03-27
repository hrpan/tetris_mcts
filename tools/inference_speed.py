import model.model_pytorch as M
import numpy as np
from tqdm import tqdm 

batch_size = 1
m = M.Model()
b = np.ones((batch_size, 1, 22, 10))
for i in tqdm(range(10000)):
    m.inference(b)
