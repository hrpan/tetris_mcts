import sys
import os
import numpy as np
import argparse
import glob
sys.path.append('.')
from util.Data import DataLoader, keyFile
import matplotlib.pyplot as plt

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
y_err = []

for f in list_of_data:
    print('Processing: ', f)
    try:
        _tmp = DataLoader([f,])
    except:
        pass
    _idx = np.ediff1d(_tmp.episode, to_end=-1) == -1
    _scores = _tmp.score[_idx]
    _cycle = _tmp.cycle[0]
    _mean = _scores.mean()
    _stddev = _scores.std() / np.sqrt(np.sum(np.absolute(_idx)))

    x.append(_cycle)
    y.append(_mean)
    y_err.append(_stddev)
"""
_sorted = sorted(zip(x,y ,y_err))
x = [x for x, _, _ in _sorted]
y = [y for _, y, _ in _sorted]
y_err = [y_err for _, _, y_err in _sorted]
"""
plt.errorbar(x, y, y_err)
plt.show()
