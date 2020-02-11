import numpy as np
from agents.core import check_low
from agents.special import std_quantile2, norm_quantile
from numba import jit

jit_args = {'nopython': True, 'cache': False, 'fastmath': True}

eps = 1e-3
n_actions = 7


@jit(**jit_args)
def shift_distribution(dist, x, vmin, vmax):
    bins = len(dist)
    delta = (vmax - vmin) / bins

    result = np.zeros(bins, dtype=np.float32)

    bin_shift = x / delta

    fraction = bin_shift - np.floor(bin_shift)

    for b in range(bins):
        b_lb = int(b + bin_shift)
        if b_lb >= bins:
            b_lb = bins - 1

        if b_lb + 1 >= bins:
            b_ub = bins - 1
        else:
            b_ub = b_lb + 1

        result[b_lb] += dist[b] * (1 - fraction)
        result[b_ub] += dist[b] * fraction

    return result


@jit(**jit_args)
def mean_dist(dist, vmin, vmax):
    bins = len(dist)
    delta = (vmax - vmin) / bins
    center = (np.arange(bins) + 0.5) * delta

    return (dist * center).sum()


@jit(**jit_args)
def mean_variance(dist, vmin, vmax):
    bins = len(dist)
    delta = (vmax - vmin) / bins

    mean = 0
    m2 = 0
    for b in range(bins):
        center = (b + 0.5) * delta
        tmp = center * dist[b]
        mean += tmp
        m2 += center * tmp

    var = m2 - mean ** 2

    return mean, var


@jit(**jit_args)
def policy_dist(child_nodes, node_stats, node_dist, curr_reward, vmin, vmax):
    stats = np.zeros((2, len(child_nodes)), dtype=np.float32)
    n = 0
    for i, c in enumerate(child_nodes):
        n += node_stats[c][0]
        mean, var = mean_variance(node_dist[c], vmin, vmax)
        stats[0][i] = node_stats[c][1] + node_stats[c][2] - curr_reward
        #stats[1][i] = var / (node_stats[c][0] + eps)
        stats[1][i] = node_stats[c][3] / (node_stats[c][0] + eps)

    q = stats[0] + norm_quantile(n) * np.sqrt(stats[1])

    return child_nodes[np.argmax(q)]


@jit(**jit_args)
def select_trace_distributional(index, child, node_stats, node_dist, vmin, vmax, low=5):

    trace = []

    while True:

        trace.append(index)

        _child_nodes = [child[index][i] for i in range(n_actions) if child[index][i] != 0]
        _child_nodes = list(set(_child_nodes))

        len_c = len(_child_nodes)

        if len_c == 0:
            break

        r = node_stats[index][2]

        index = check_low(_child_nodes, node_stats, n=low)

        if not index:
            index = policy_dist(_child_nodes, node_stats, node_dist, r, vmin, vmax)

    return np.array(trace, dtype=np.int32)


@jit(**jit_args)
def backup_trace_distributional(trace, node_stats, node_dist, r, dist, vmin, vmax):
    mean = mean_dist(dist, vmin, vmax)
    for idx in trace:
        ns = node_stats[idx]
        _r = r - ns[2]
        new_dist = shift_distribution(dist, _r, vmin, vmax)
        node_dist[idx] = (node_dist[idx] * ns[0] + new_dist) / (ns[0] + 1)
        x = mean + _r
        ns[0] += 1
        delta = x - ns[1]
        ns[1] += delta / ns[0]
        delta2 = x - ns[1]
        ns[4] += delta * delta2
        if ns[0] > 1:
            ns[3] = ns[4] / (ns[0] - 1)
