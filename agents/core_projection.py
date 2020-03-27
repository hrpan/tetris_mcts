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
def get_unique_child_obs(index, child, node_stats, n_to_o):
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
            if node_stats[c][2] > node_stats[child_nodes[_idx]][2]:
                child_nodes[_idx] = c
    return child_nodes, child_obs


@jit(**jit_args)
def policy_obs_clt(child_nodes, child_obs, node_stats, obs_stats, root):
    stats = np.zeros((2, len(child_nodes)), dtype=np.float32)
    n = 0
    for i in range(len(child_nodes)):
        c, o = child_nodes[i], child_obs[i]
        n += obs_stats[o][0]
        stats[0][i] = obs_stats[o][1] + node_stats[c][2] - node_stats[root][2]
        stats[1][i] = obs_stats[o][2] / (obs_stats[o][0] + eps)
    q = stats[0] + norm_quantile(n) * np.sqrt(stats[1])
    return child_nodes[np.argmax(q)]


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
def select_trace_obs(index, child, node_stats, obs_stats, n_to_o, policy=policy_obs_mc, low=1):

    trace = []

    while True:

        trace.append(index)

        child_nodes, child_obs = get_unique_child_obs(index, child, node_stats, n_to_o)

        if not child_nodes:
            break

        o = check_low(child_obs, obs_stats, n=low)
        if not o:
            index = policy(child_nodes, child_obs, node_stats, obs_stats, index)
        else:
            index = child_nodes[child_obs.index(o)]
    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def backup_trace_obs_by_policy(trace, child, node_stats, value, obs_stats, n_to_o, policy=_policy_obs_mc):
    for idx in trace[::-1]:
        node_stats[idx][0] += 1
        obs = n_to_o[idx]
        o_stats = obs_stats[obs]
        o_stats[0] += 1
        child_nodes, child_obs = get_unique_child_obs(idx, child, node_stats, n_to_o)
        if not child_nodes:
            continue
        empty = False
        for o in child_obs:
            if obs_stats[o][0] == 0:
                empty = True
                break
        if empty:
            continue
        o_stats[1:] = 0
        p = policy(child_nodes, child_obs, node_stats, obs_stats, idx)
        for _p, _c, _o in zip(p, child_nodes, child_obs):
            mu = obs_stats[_o][1] + node_stats[_c][2] - node_stats[idx][2]
            o_stats[1] += _p * mu
            o_stats[2] += _p * mu * mu
        o_stats[2] -= o_stats[1] ** 2


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
