import matplotlib.pyplot as plt
import tables
import sys

file_path = sys.argv[1]
_file = tables.open_file(file_path, mode='r')

table = _file.root.Loss
for k in table.coldescrs.keys():
    if 'loss' in k:
        a, = plt.plot(table.col(k))
        a.set_label(k)
plt.legend()
plt.show()
