import numpy as np
from agents.policy import *
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
def get_unique_child_obs(index, child, score, n_to_o):
    child_nodes = []
    child_obs = []
    for i in range(n_actions):
        c = child[index][i]
        if c == 0:
            continue
        o = n_to_o[c]
        if o not in child_obs:
            child_nodes.append(c)
            child_obs.append(o)
        else:
            _idx = child_obs.index(o)
            if score[c] > score[child_nodes[_idx]]:
                child_nodes[_idx] = c
    return child_nodes, child_obs


@jit(**jit_args)
def _policy_obs_mc(child_nodes, child_obs, node_stats, obs_stats, root, sims=100):
    stats = np.zeros((2, len(child_nodes)), dtype=np.float32)
    for i, (c, o) in enumerate(zip(child_nodes, child_obs)):
        stats[0][i] = obs_stats[o][1] + node_stats[c][2] - node_stats[root][2]
        stats[1][i] = obs_stats[o][2]

    rnd = np.random.randn(sims, len(child_nodes))

    q = rnd * np.sqrt(stats[1]) + stats[0]

    action_count = np.zeros((len(child_nodes),))
    for i in range(sims):
        action_count[np.argmax(q[i])] += 1

    return action_count / action_count.sum()


@jit(**jit_args)
def policy_obs_mc(child_nodes, child_obs, node_stats, obs_stats, root, sims=100, stochastic=True):
    weights = _policy_obs_mc(child_nodes, child_obs, node_stats, obs_stats, root, sims)

    if stochastic:
        return child_nodes[sample_from(weights)]
    else:
        return child_nodes[np.argmax(weights)]


@jit(**jit_args)
def select_trace_obs(index, child, visit, value, variance, score, n_to_o, policy=policy_clt, low=1):

    trace = []

    while True:

        trace.append(index)

        child_nodes, child_obs = get_unique_child_obs(index, child, score, n_to_o)

        if not child_nodes:
            break

        o = check_low(child_obs, visit, n=low)
        if not o:
            _tmp = np.empty((3, len(child_nodes)))
            for i, _o in enumerate(child_obs):
                _tmp[0][i] = visit[_o]
                _tmp[1][i] = value[_o] + score[child_nodes[i]] - score[index]
                _tmp[2][i] = variance[_o]
            index = policy(child_nodes, _tmp[0], _tmp[1], _tmp[2])
        else:
            index = child_nodes[child_obs.index(o)]
    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def backup_trace_obs_by_policy(trace, child, visit, value, variance, score, n_to_o, policy=policy_clt, low=1):
    for idx in trace[::-1]:
        obs = n_to_o[idx]
        visit[obs] += 1
        child_nodes, child_obs = get_unique_child_obs(idx, child, score, n_to_o)
        if not child_nodes:
            continue
        empty = False
        for o in child_obs:
            if visit[o] == 0:
                empty = True
                break
        if empty:
            continue
        value[obs] = 0
        variance[obs] = 0
        p = policy(child_nodes, child_obs, visit, value, variance, score, idx)
        for _p, _c, _o in zip(p, child_nodes, child_obs):
            mu = value[_o] + score[_c] - score[idx]
            value[obs] += _p * mu
            variance[obs] += _p * mu * mu
        variance[obs] -= value[obs] ** 2


@jit(**jit_args)
def backup_trace_obs(trace, visit, value, variance, n_to_o, score, _value, _variance, gamma=0.999):
    for idx in trace[::-1]:
        _value -= score[idx]
        obs = n_to_o[idx]
        if visit[obs] == 0:
            value[obs] = _value
            variance[obs] = _variance
        else:
            delta = _value - value[obs]
            value[obs] += delta / (visit[obs] + 1)
            delta2 = _value - value[obs]
            variance[obs] += (delta * delta2 - variance[obs]) / (visit[obs] + 1)
        visit[obs] += 1
        _value = gamma * _value + score[idx]


@jit(**jit_args)
def backup_trace_obs_exp_moving(trace, visit, value, variance, n_to_o, score, _value, _variance, alpha=0.1, gamma=0.999):
    for idx in trace[::-1]:
        _value -= score[idx]
        obs = n_to_o[idx]
        if visit[obs] == 0:
            value[obs] = _value
            variance[obs] = _variance
        else:
            _v = _value - value[obs]
            value[obs] += alpha * _v
            variance[obs] = (1 - alpha) * (variance[obs] + alpha * _v * _v)
        visit[obs] += 1
        _value = gamma * _value + score[idx]


@jit(**jit_args)
def backup_trace_value_policy_obs(trace, child, visit, value, policy, n_to_o, score, _value):
    for idx in trace[::-1]:
        v = _value - score[idx]
        obs = n_to_o[idx]
        visit[obs] += 1
        value[obs] += (v - value[obs]) / visit[obs]
        vmax, amax = -9999999, 0
        for i, c in enumerate(child[idx]):
            _o = n_to_o[c]
            _v = value[_o] + score[c] - score[idx]
            if _v > vmax:
                vmax, amax = _v, i
        policy[obs][amax] += 1


@jit(**jit_args)
def backup_trace_mixture_obs(trace, visit, value, variance, n_to_o, score, _value, _variance, gamma=0.999):
    for idx in trace[::-1]:
        _value -= score[idx]
        obs = n_to_o[idx]
        visit[obs] += 1

        v_diff = _value - value[obs]
        v_sq_diff = _value ** 2 - value[obs] ** 2

        v_tmp = value[obs]

        value[obs] += v_diff / visit[obs]

        var_diff = _variance - variance[obs]

        variance[obs] += (var_diff + v_sq_diff) / visit[obs] - (v_diff / visit[obs]) * (v_tmp + value[obs])

        _value = gamma * _value + score[idx]
        _variance *= gamma
