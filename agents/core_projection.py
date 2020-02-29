import numpy as np

from agents.special import std_quantile2, norm_quantile
from agents.core import *

from numba import jit

jit_args = {'nopython': True, 'cache': False, 'fastmath': True}


@jit(**jit_args)
def get_occupied_obs(index, child, n_to_o):

    to_traverse = [index, ]

    traversed = []

    traversed_obs = []

    i = 0
    while i < len(to_traverse):
        idx = to_traverse[i]
        if idx not in traversed:
            for c in child[idx]:
                if c not in traversed:
                    to_traverse.append(c)
            traversed.append(idx)
            traversed_obs.append(n_to_o[idx])
        i += 1
    return set(traversed_obs)


@jit(**jit_args)
def select_trace_obs(index, child, node_stats, obs_stats, n_to_o, low=1):

    trace = []

    while True:

        trace.append(index)

        _child_nodes = []
        _child_obs = []
        for i in range(n_actions):
            c = child[index][i]
            if c == 0:
                continue
            o = n_to_o[c]
            if o not in _child_obs:
                _child_nodes.append(c)
                _child_obs.append(o)
            else:
                _idx = _child_obs.index(o)
                if node_stats[c][2] > node_stats[_child_nodes[_idx]][2]:
                    _child_nodes[_idx] = c

        len_c = len(_child_nodes)

        if len_c == 0:
            break

        r = node_stats[index][2]
        index = check_low(_child_obs, obs_stats, n=low)

        if not index:
            stats = np.zeros((2, len_c), dtype=np.float32)
            n = 0
            for i in range(len_c):
                c, o = _child_nodes[i], _child_obs[i]
                n += obs_stats[o][0]
                stats[0][i] = obs_stats[o][1] + node_stats[c][2] - r
                stats[1][i] = obs_stats[o][2] / (obs_stats[o][0] + eps)

            _q = stats[0] + norm_quantile(n) * np.sqrt(stats[1])

            index = _child_nodes[np.argmax(_q)]
        else:
            index = _child_nodes[_child_obs.index(index)]
    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def backup_trace_obs(trace, node_stats, value, obs_stats, n_to_o):
    """
    numerical stable sample variance calculation based on welford's online algorithm
    [0]:count
    [1]:mean
    [3]:M2
    """
    for idx in trace:
        v = value - node_stats[idx][2]
        obs = n_to_o[idx]
        o_stats = obs_stats[obs]
        if o_stats[0] == 0:
            o_stats[1] = v
        else:
            delta = v - o_stats[1]
            o_stats[1] += delta / (o_stats[0] + 1)
            delta2 = v - o_stats[1]
            o_stats[2] = (o_stats[2] * o_stats[0] + delta * delta2) / (o_stats[0] + 1)
        o_stats[0] += 1
        node_stats[idx][0] += 1
        node_stats[idx][4] = max(v, node_stats[idx][4])
