import numpy as np
from numba import jit
from agents.special import std_quantile2, norm_quantile

jit_args = {'nopython': True, 'cache': True, 'fastmath': True}

@jit(**jit_args)
def policy_clt(nodes, visit, value, variance):

    n = visit.sum()

    _q = value + norm_quantile(n) * np.sqrt(variance / visit)

    return nodes[np.argmax(_q)]


@jit(**jit_args)
def policy_gauss(nodes, visit, value, variance):

    n = visit.sum()

    _q = value + norm_quantile(n) * np.sqrt(variance)

    return nodes[np.argmax(_q)]


@jit(**jit_args)
def policy_max(nodes, visit, value, v_max):
    _max = np.max(v_max)

    _q = value + _max * np.sqrt(np.log(visit.sum()) / visit)

    return nodes[np.argmax(_q)]


@jit(**jit_args)
def policy_mc(nodes, value, variance):
    _q = value + np.random.randn(variance) * np.sqrt(variance)
    return nodes[np.argmax(_q)]


@jit(**jit_args)
def policy_random(nodes):
    return nodes[np.random.randint(len(nodes))]


@jit(**jit_args)
def policy_greedy(nodes, value):
    v_max = 0
    n_max = 0
    for c, n in enumerate(nodes):
        if value[c] >= v_max:
            v_max = value[c]
            n_max = n
    return n_max
