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
y_s = []
y_s_err = []
y_l = []
y_l_err = []

for f in list_of_data:
    print('Processing: ', f)
    try:
        _tmp = DataLoader([f,])
    except:
        pass
    _idx = np.ediff1d(_tmp.episode, to_end=-1) == -1
    _cycle = _tmp.cycle[0]
    _scores = _tmp.score[_idx]
    s_mean = _scores.mean()
    s_stddev = _scores.std() / np.sqrt(np.sum(np.absolute(_idx)))
    _lines = _tmp.lines[_idx]
    l_mean = _lines.mean()
    l_stddev = _lines.std() / np.sqrt(np.sum(np.absolute(_idx)))

    x.append(_cycle)
    y_s.append(s_mean)
    y_s_err.append(s_stddev)
    y_l.append(l_mean)
    y_l_err.append(l_stddev)

c1 = 'tab:red'
plt.errorbar(x, y_s, y_s_err, color=c1)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.gca().tick_params(axis='y', labelcolor=c1)

c2 = 'tab:blue'
ax2 = plt.gca().twinx()
ax2.errorbar(x, y_l, y_l_err, color=c2)
ax2.set_ylabel('Lines cleared')
ax2.tick_params(axis='y', labelcolor=c2)

plt.show()
