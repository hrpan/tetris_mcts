import numpy as np
import random
import sys

from numba import jit
from numba import int32, float32

n_actions = 6

@jit(nopython=True,cache=True)
def select_index(index,child,node_stats):

    trace = []

    while True:

        trace.append(index)

        is_leaf_node = True

        _child_nodes = []
        for i in range(n_actions):
            if child[index][i] != 0:
                _child_nodes.append(child[index][i])

        len_c = len(_child_nodes)

        if len_c == 0:
            break

        has_unvisited_node = False

        _stats = np.zeros((2,len_c))

        for i in range(len_c):
            _idx = _child_nodes[i]
            _stats[0][i] = node_stats[_idx][0]
            _stats[1][i] = node_stats[_idx][1]
            if node_stats[_idx][0] == 0:
                index = _idx
                has_unvisited_node = True
                break

        if has_unvisited_node:
            continue

        _t = np.sum(_stats[0]) 

        _c = 1.0 * np.sqrt( np.log(_t) / _stats[0] )

        _q = _stats[1] / _stats[0]

        _v = _q + _c 
        
        _a = np.argmax(_v)

        index = _child_nodes[_a]

    return trace

@jit(nopython=True,cache=True)
def backup_trace(trace,node_stats,value):

    for idx in trace:
        v = value - node_stats[idx][2] 
        node_stats[idx][0] += 1
        node_stats[idx][1] += v

@jit(nopython=True,cache=True)
def get_all_childs(index,child):
    
    to_traverse = [index,]

    traversed = []

    i = 0
    while i < len(to_traverse):
        idx = to_traverse[i]
        if idx not in traversed:
            to_traverse += list(child[idx])   
            traversed.append(idx)
        i += 1
    return set(to_traverse)

@jit(nopython=True,cache=True)
def choose_action(p):
    _cdf = p.cumsum()

    rnd = np.random.rand()

    _a = np.searchsorted(_cdf,rnd)

    return _a


