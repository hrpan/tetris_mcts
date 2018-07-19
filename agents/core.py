import numpy as np
import random

from numba import jit
from numba import int32, float32

n_actions = 6

eps = 1e-7

@jit(nopython=True, cache=True)
def findZero(arr):
    for i in range(n_actions):
        if arr[i] == 0:
            return i
    return False

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

        _max = 1.0

        for i in range(len_c):
            _idx = _child_nodes[i]
            _stats[0][i] = node_stats[_idx][0]
            _stats[1][i] = node_stats[_idx][1]
            _max = max(_max,node_stats[_idx][4])
            if node_stats[_idx][0] == 0:
                index = _idx
                has_unvisited_node = True
                break

        if has_unvisited_node:
            continue

        _t = np.sum(_stats[0]) 

        _c = _max * np.sqrt( np.log(_t) / _stats[0] )

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
        node_stats[idx][3] += v * v
        node_stats[idx][4] = max(v,node_stats[idx][4])

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

@jit(nopython=True,cache=True)
def computeQ(child_set, node_stats):
    q = 0
    q2 = 0
    q_max = 0
    n_tot = 0
    for c in child_set:
        n_tot += node_stats[c][0]
        q += node_stats[c][1]
        q2 += node_stats[c][3]
        q_max = max(q_max, node_stats[c][4])

    return q, q/(n_tot+1e-7), q2, q_max

@jit(nopython=True,cache=True)
def atomicSelect(stats):

    _n_sum = np.sum(stats[0])
    
    _max = max(1, np.amax(stats[5]))

    _q = stats[3]

    _c = _max * np.sqrt( np.log(_n_sum) / stats[0] )

    return np.argmax( _q + _c )

@jit(nopython=True,cache=True)
def atomicFill(i, c_set, _stats, node_stats):
    q_acc, q_mean, q2_acc, q_max = computeQ(c_set, node_stats)
    _stats[1][i] = q_acc
    _stats[2][i] = 0
    _stats[3][i] = q_mean
    _stats[4][i] = q2_acc
    _stats[5][i] = q_max

@jit(nopython=True,cache=True)
def update_child_info(trace, action, child_info):
    for i in range(len(action)):
        s = trace[i]
        _s = trace[i+1]
        a = action[i]
        for j in range(child_info.shape[2]):
            if child_info[s][a][j][0] == _s:
                child_info[s][a][j][1] += 1
                break
            elif child_info[s][a][j][0] == 0:
                child_info[s][a][j][0] = _s
                child_info[s][a][j][1] = 1
                break
            if j == child_info.shape[2] - 1:
                return False
    return True

@jit(nopython=True,cache=True)
def fill_child_stats(idx, node_stats, child_info):
    _stats = np.zeros((6, n_actions),dtype=np.float32)

    for i in range(n_actions):
        q_avg = 0
        counts = 0
        _val_2 = 0
        _max = 0
        for j in range(child_info.shape[2]):
            _node = child_info[idx][i][j][0]
            if _node == 0:
                break
            _count = child_info[idx][i][j][1]
            _val = node_stats[_node][1] / node_stats[_node][0]
            q_avg += _count * _val
            counts += _count
            _val_2 += node_stats[_node][3]
            _max = max(_max, node_stats[_node][4]) 

        _stats[0][i] = counts
        _stats[1][i] = 0
        _stats[2][i] = 0
        _stats[3][i] = q_avg / (counts+eps)
        _stats[4][i] = _val_2
        _stats[5][i] = _max
    
    return _stats 

@jit(nopython=True,cache=True)
def get_all_child_2(index, child_info):
    to_traverse = [index, ]
    traversed = set([0])

    i = 0
    while i < len(to_traverse):
        idx = to_traverse[i]
        if idx not in traversed:
            traversed.add(idx)
            for j in range(n_actions):
                to_traverse += list(child_info[idx, j, :, 0]) 
        i += 1

    return traversed

def select_index_2(game, node_dict, node_stats, child_info):

    trace = []
    action = []
    idx = node_dict.get(game)

    while idx and not game.end:

        trace.append(idx)

        action_counts = np.sum(child_info[idx, :, :, 1], axis=1)        
 
        _a = findZero(action_counts)
       
        if not _a:
            _stats = fill_child_stats(idx, node_stats, child_info)
                
            _a = atomicSelect(_stats)

        action.append(_a)

        game.play(_a)

        idx = node_dict.get(game)

    return trace, action


