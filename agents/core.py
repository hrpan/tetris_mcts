import numpy as np
import random
from agents.policy import *
from agents.special import std_quantile2, norm_quantile

from numba import jit

jit_args = {'nopython': True, 'cache': False, 'fastmath': True}

n_actions = 7

eps = 1e-7

__stats = np.zeros((6, n_actions), dtype=np.float32)


@jit(**jit_args)
def findZero(arr):
    for i in range(n_actions):
        if arr[i] == 0:
            return i
    return False


@jit(**jit_args)
def sample_from(weights):
    x = np.random.rand()
    cum = 0
    p = weights / weights.sum()
    for i in range(len(p)):
        cum += p[i]
        if cum >= x:
            return i

    return -1


@jit(**jit_args)
def backup_trace(trace, node_stats, value):

    for idx in trace:
        v = value - node_stats[idx][2]
        n, q = node_stats[idx][0], node_stats[idx][1]
        node_stats[idx][0] += 1
        node_stats[idx][1] = (n * q + v) / (n + 1)
        node_stats[idx][3] += v * v
        node_stats[idx][4] = max(v, node_stats[idx][4])


@jit(**jit_args)
def get_all_childs(index, child):

    to_traverse = [index, ]

    traversed = []

    i = 0
    while i < len(to_traverse):
        idx = to_traverse[i]
        if idx not in traversed:
            for c in child[idx]:
                if c not in traversed:
                    to_traverse.append(c)
            traversed.append(idx)
        i += 1
    return set(to_traverse)


@jit(**jit_args)
def choose_action(p):
    _cdf = p.cumsum()

    rnd = np.random.rand()

    _a = np.searchsorted(_cdf, rnd)

    return _a


@jit(**jit_args)
def atomicSelect(stats):

    _n_sum = np.sum(stats[0])

    _max = max(1, np.amax(stats[5]))

    _q = stats[3]

    _c = _max * np.sqrt(np.log(_n_sum) / stats[0])

    return np.argmax(_q + _c)


def update_child_info(trace, action, child_info):
    for i in range(len(action)):
        s = trace[i]
        _s = trace[i+1]
        a = action[i]
        found = False
        for pair in child_info[s][a]:
            if pair[0] == _s:
                found = True
                pair[1] += 1
                break
        if not found:
            child_info[s][a] = np.concatenate((child_info[s][a], [[_s, 1]]))


@jit(**jit_args)
def atomicFill(act, stats, node_stats, childs):

    val = 0
    counts = 0
    val_2 = 0
    m = 0

    for j in range(len(childs)):
        n = childs[j][0]
        c = childs[j][1]
        _v = node_stats[n][1] / node_stats[n][0]
        val += c * _v
        counts += c
        val_2 += node_stats[n][3]
        m = max(m, node_stats[n][4])

    stats[0][act] = counts
    stats[1][act] = 0
    stats[2][act] = 0
    stats[3][act] = val / counts
    stats[4][act] = val_2
    stats[5][act] = m


def fill_child_stats(idx, node_stats, child_info):

    global __stats
    __stats.fill(0)

    for i in range(n_actions):
        if len(child_info[idx][i]) > 0:
            atomicFill(i, __stats, node_stats, child_info[idx][i])

    return __stats


def get_all_child_2(index, child_info):
    to_traverse = [index, ]
    traversed = set([0])

    i = 0
    while i < len(to_traverse):
        idx = to_traverse[i]
        if idx not in traversed:
            traversed.add(idx)
            _list = [p[0] for a in range(n_actions) for p in child_info[idx][a]]
            to_traverse += _list
        i += 1

    return traversed


def findZero_2(index, child_info):
    for i in range(n_actions):
        if len(child_info[index][i]) == 0:
            return i
    return False


@jit(**jit_args)
def _tmp_func(stats, act, node_stats, childs):
    q_max = 0
    for i in range(len(childs)):
        idx = childs[i][0]
        node = node_stats[idx]
        stats[0][act] += childs[i][1]
        stats[1][act] += childs[i][1] * node[1] / node[0]
        stats[2][act] += childs[i][1] * node[4] * np.sqrt(1 / node[0])
        q_max = max(q_max, node[4])
    stats[1][act] /= (stats[0][act]+eps)
    stats[2][act] /= (stats[0][act]+eps)
    stats[3][act] = len(childs)

    return q_max


@jit(**jit_args)
def _tmp_select(stats, v_max):
    v_max = v_max + eps
    _p = (stats[3] + 0.5) / (stats[0] + 1)
    _n = np.sqrt(np.log(np.sum(stats[0])))
    _u = _n * (_p * v_max * np.sqrt(1 / stats[0]) + (1 - _p) * stats[2])
    return np.argmax(stats[1] + _u)


def select_index_2(game, node_dict, node_stats, child_info):

    trace = []
    action = []
    idx = node_dict.get(game)

    while idx and not game.end:

        trace.append(idx)

        _a = findZero_2(idx, child_info)

        if _a is False:

            _stats_tmp = np.zeros((4, n_actions), dtype=np.float32)

            _max = max([_tmp_func(_stats_tmp, i, node_stats, child_info[idx][i]) for i in range(n_actions)])
            _a = _tmp_select(_stats_tmp, _max)
        action.append(_a)

        game.play(_a)

        idx = node_dict.get(game)

    return trace, action


@jit(**jit_args)
def select_index_3(index, child, node_stats):

    trace = []

    while True:

        trace.append(index)

        _child_nodes = []
        for i in range(n_actions):
            if child[index][i] != 0:
                _child_nodes.append(child[index][i])

        _child_nodes = list(set(_child_nodes))

        len_c = len(_child_nodes)

        if len_c == 0:
            break

        has_unvisited_node = False

        _stats = np.zeros((2, len_c), dtype=np.float32)

        _n = 0

        for i in range(len_c):
            _idx = _child_nodes[i]
            if node_stats[_idx][0] == 0:
                index = _idx
                has_unvisited_node = True
                break
            _n += node_stats[_idx][0]
            _stats[0][i] = node_stats[_idx][1] + node_stats[_idx][2] - node_stats[index][2]
            _stats[1][i] = node_stats[_idx][3] / node_stats[_idx][0]

        if has_unvisited_node:
            continue

        _c = np.sqrt(_stats[1] * np.log(_n))

        _q = _stats[0]

        _v = _q + _c

        _a = np.argmax(_v)

        index = _child_nodes[_a]

    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def backup_trace_3(trace, node_stats, value, alpha=0.01):
    for idx in trace:
        v = value - node_stats[idx][2]
        if node_stats[idx][0] == 0:
            node_stats[idx][1] = v
        else:
            _v = v - node_stats[idx][1]
            node_stats[idx][1] += alpha * _v
            node_stats[idx][3] = (1-alpha) * (node_stats[idx][3] + alpha * _v * _v)
        node_stats[idx][0] += 1
        node_stats[idx][4] = max(v, node_stats[idx][4])


@jit(**jit_args)
def select_index_bayes(index, child, node_stats, min_n=10):
    """
    based on bayes UCB
    http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf
    """
    trace = []

    while True:

        trace.append(index)

        _child_nodes = []
        for i in range(n_actions):
            if child[index][i] != 0:
                _child_nodes.append(child[index][i])

        _child_nodes = list(set(_child_nodes))

        len_c = len(_child_nodes)

        if len_c == 0:
            break

        has_low_node = False

        low_nodes = []

        _stats = np.zeros((3, len_c), dtype=np.float32)

        _n = 0

        for i in range(len_c):
            _idx = _child_nodes[i]
            if node_stats[_idx][0] < min_n:
                low_nodes.append(_idx)
                has_low_node = True

            if has_low_node: continue

            _n += node_stats[_idx][0]

            _stats[0][i] = node_stats[_idx][0]
            _stats[1][i] = node_stats[_idx][1] + node_stats[_idx][2] - node_stats[index][2]
            _stats[2][i] = node_stats[_idx][3] / (node_stats[_idx][0] - 1)

        if low_nodes:
            index = np.random.choice(np.array(low_nodes))
            continue

        quantiles = std_quantile2(_stats[0] - 1, _n)

        _c = np.sqrt(_stats[2] / _stats[0]) * quantiles

        _q = _stats[1]

        _v = _q + _c

        _a = np.argmax(_v)

        index = _child_nodes[_a]

    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def select_index_clt(index, child, node_stats):

    trace = []

    while True:

        trace.append(index)

        _child_nodes = [child[index][i] for i in range(n_actions) if child[index][i] != 0]
        _child_nodes = list(set(_child_nodes))

        len_c = len(_child_nodes)

        if len_c == 0:
            break

        has_unvisited_node = False

        _stats = np.zeros((2, len_c), dtype=np.float32)

        _n = 0

        for i in range(len_c):
            _idx = _child_nodes[i]
            if node_stats[_idx][0] == 0:
                index = _idx
                has_unvisited_node = True
                break
            _n += node_stats[_idx][0]
            _stats[0][i] = node_stats[_idx][1] + node_stats[_idx][2] - node_stats[index][2]
            _stats[1][i] = node_stats[_idx][3] / node_stats[_idx][0]

        if has_unvisited_node:
            continue

        _c = np.sqrt(_stats[1]) * norm_quantile(_n)

        _q = _stats[0]

        _v = _q + _c

        _a = np.argmax(_v)

        index = _child_nodes[_a]

    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def backup_trace_welford(trace, node_stats, value):
    """
    numerical stable sample variance calculation based on welford's online algorithm
    [0]:count
    [1]:mean
    [3]:M2
    """
    for idx in trace:
        v = value - node_stats[idx][2]
        node_stats[idx][0] += 1

        delta = v - node_stats[idx][1]

        node_stats[idx][1] += delta / node_stats[idx][0]

        delta2 = v - node_stats[idx][1]

        node_stats[idx][3] += delta * delta2
        node_stats[idx][4] = max(v, node_stats[idx][4])


@jit(**jit_args)
def backup_trace_welford_v2(trace, visit, value, variance, score, _value, _variance):
    """
    numerical stable sample variance calculation based on welford's online algorithm
    [0]:count
    [1]:mean
    [3]:M2
    """
    for idx in trace:
        v = _value - score[idx]
        if visit[idx] == 0:
            value[idx] = v
            variance[idx] = _variance
        else:
            delta = v - value[idx]
            value[idx] += delta / (visit[idx] + 1)
            delta2 = v - value[idx]
            variance[idx] = (variance[idx] * visit[idx] + delta * delta2) / (visit[idx] + 1)
        visit[idx] += 1


@jit(**jit_args)
def backup_trace_with_variance(trace, node_stats, value, variance):
    for idx in trace:
        v = value - node_stats[idx][2]
        n = node_stats[idx][0]
        if n == 0:
            node_stats[idx][1] = v
        else:
            node_stats[idx][1] += (v - node_stats[idx][1]) / (n + 1)
            #assuming fully correlated
            std_diff = (variance ** 0.5 - node_stats[idx][3] ** 0.5) / (n + 1)
            node_stats[idx][3] = node_stats[idx][3] + 2 * node_stats[idx][3] ** 0.5 * std_diff + std_diff ** 2
        node_stats[idx][0] += 1
        node_stats[idx][4] = max(v, node_stats[idx][4])


@jit(**jit_args)
def check_low(indices, count, n=1):
    low = [i for i in indices if count[i] < n]
    if low:
        return low[np.random.randint(len(low))]
    else:
        return 0


@jit(**jit_args)
def select_trace(index, child, visit, value, variance, score, policy=policy_clt, low=1):

    trace = []

    while True:

        trace.append(index)

        child_nodes = [child[index][i] for i in range(n_actions) if child[index][i] != 0]
        child_nodes = list(set(child_nodes))

        if not child_nodes:
            break

        n_index = check_low(child_nodes, visit, n=low)

        if not n_index:
            _tmp = np.empty((3, len(child_nodes)))
            for i, c in enumerate(child_nodes):
                _tmp[0][i] = visit[c]
                _tmp[1][i] = value[c] + score[c] - score[index]
                _tmp[2][i] = variance[c]
            index = policy(child_nodes, _tmp[0], _tmp[1], _tmp[2])
        else:
            index = n_index
    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def select_trace_with_policy(index, child, policy):

    trace = []

    while True:

        trace.append(index)

        index = child[index][sample_from(policy[index])]

        if index == 0:
            break
    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def backup_trace_by_policy(trace, node_stats, child, policy=policy_greedy):

    for idx in trace:
        node_stats[idx][0] += 1
        child_nodes = [child[idx][i] for i in range(n_actions) if (child[idx][i] != 0 and node_stats[child[idx][i]][0] > 0)]
        child_nodes = list(set(child_nodes))

        if not child_nodes:
            continue

        index = policy(child_nodes, node_stats, node_stats[idx][2])

        node_stats[idx][1] = node_stats[index][2] - node_stats[idx][2] + node_stats[index][1]
        node_stats[idx][3] = node_stats[index][3]
        node_stats[idx][4] = max(node_stats[idx][1], node_stats[idx][4])


@jit(**jit_args)
def backup_trace_value_policy(trace, child, visit, value, action, score, _value):
    for idx in trace[::-1]:
        v = _value - score[idx]
        visit[idx] += 1
        value[idx] += (v - value[idx]) / visit[idx]
        vmax, amax = -9999999, 0
        for i, c in enumerate(child[idx]):
            _v = value[c] + score[c] - score[idx]
            if _v > vmax:
                vmax, amax = _v, i
        policy[idx][amax] += 1
