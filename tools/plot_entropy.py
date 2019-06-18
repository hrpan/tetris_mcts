import sys
import os
import numpy as np
import argparse
import glob
sys.path.append('.')
from util.Data import DataLoader, keyFile
import matplotlib.pyplot as plt

n_actions = 7

parser = argparse.ArgumentParser()

parser.add_argument('--data_paths', default=[], nargs='*', help='Data paths')
args = parser.parse_args()
data_paths = args.data_paths

list_of_data = []
for path in data_paths:
    list_of_data += glob.glob(path)

list_of_data = sorted(list_of_data, key=keyFile)

x = []
y = []

for f in list_of_data:
    print('Processing: ', f)
    try:
        _tmp = DataLoader([f,])
    except:
        pass
    _cycle = _tmp.cycle[0]
    n = _tmp.child_stats[:,0]
    n_sum = np.sum(n, axis=1)
    p = np.array([n[i]/n_sum[i] for i in range(len(n))])
    entropy = np.mean(np.sum(-p * np.log(p), axis=1))
    x.append(_cycle)
    y.append(entropy)

plt.plot(x, y)
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.axhline(np.log(n_actions), color='red')
plt.show()
