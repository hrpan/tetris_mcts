import numpy as np

l = np.load('data/dump_grad.npz')
s = l['arr_0']
v = l['arr_1']
var = l['arr_2']
w = l['arr_3']
v_p = l['arr_4']
var_p = l['arr_5']

diff = (v - v_p) ** 2 / var_p
w_diff = w * diff
idx = w_diff.argmax()
print(s[idx])
print('w_diff:{}'.format(w_diff[idx]))
print('diff:{}'.format(diff[idx]))
print('var:{}/{}'.format(var[idx], var_p[idx]))
print('v:{}/{}'.format(v[idx], v_p[idx]))
print('w:{}'.format(w[idx]))

idx = diff.argmax()
print(s[idx])
print('w_diff:{}'.format(w_diff[idx]))
print('diff:{}'.format(diff[idx]))
print('var:{}/{}'.format(var[idx], var_p[idx]))
print('v:{}/{}'.format(v[idx], v_p[idx]))
print('w:{}'.format(w[idx]))
