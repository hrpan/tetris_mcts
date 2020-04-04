import numpy as np
from collections import Counter, deque
from agents.agent import TreeAgent
from agents.core import *
from agents.core_projection import *
from model.model_vp import Model_VP as Model
from sys import stderr

eps = 1e-7
perr = dict(file=stderr, flush=True)


class ApproxPolicyIter(TreeAgent):

    def __init__(self, online=True, memory_size=250000, min_visits_to_store=10, **kwargs):

        super().__init__(max_nodes=100000, **kwargs)

        self.g_tmp = self.env(*self.env_args)

        self.online = online

        self.arrays['policy'] = np.zeros((self.max_nodes, n_actions), dtype=np.float32)
        self.arrays['policy_new'] = np.zeros((self.max_nodes, n_actions), dtype=np.float32)
        if self.projection:
            self.obs_arrays['policy'] = np.zeros((self.max_nodes, n_actions), dtype=np.float32)
            self.obs_arrays['policy_new'] = np.zeros((self.max_nodes, n_actions), dtype=np.float32)

        if online:

            self.memory_size = memory_size

            self.memory = [
                    np.zeros((memory_size, 1, 22, 10), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32),
                    np.zeros((memory_size, n_actions), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32)
                    ]

            self.memory_index = 0

            self.n_trains = 0

        self.min_visits_to_store = min_visits_to_store

        self.model = Model()
        self.model.load()

        self.inference = lambda state: self.model.inference(state[None, None, :, :])

    def evaluate_state(self, state):

        v, policy = self.inference(state)

        return v[0][0], policy[0]

    def get_action(self):
        if self.projection:
            _value = self.obs_arrays['value']
        else:
            _value = self.arrays['value']
        _score = self.arrays['score']
        vmax, amax = -float('inf'), 0
        for i, c in enumerate(self.arrays['child'][self.root]):
            _c = self.node_to_obs[c] if self.projection else c
            _v = _value[_c] + _score[c] - _score[self.root]
            if _v > vmax:
                vmax, amax = _v, i
        return amax

    def get_prob(self):
        return self.arrays['policy'][self.root]

    def get_stats(self):

        _tmp = np.zeros((3, n_actions), dtype=np.float32)
        if self.projection:
            _visit = self.obs_arrays['visit']
            _value = self.obs_arrays['value']
        else:
            _visit = self.arrays['visit']
            _value = self.arrays['value']
        _score = self.arrays['score']
        for i, c in enumerate(self.arrays['child'][self.root]):
            _c = self.node_to_obs[c] if self.projection else c
            _tmp[0][i] = _visit[_c]
            _tmp[1][i] = _value[_c] + _score[c] - _score[self.root]
        return _tmp

    def mcts(self, root_index):

        trace = select_trace_with_policy(root_index, self.arrays['child'], self.arrays['policy'])

        leaf_index = trace[-1]

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.score
        if not leaf_game.end:

            v, p = self.evaluate_state(leaf_game.getState())

            self.arrays['policy'][leaf_index] = p
            self.arrays['policy_new'][leaf_index] = p
            if self.projection:
                self.obs_arrays['policy_new'][self.node_to_obs[leaf_index]] = p

            value += v
            self.expand(leaf_game)

        if self.projection:
            backup_trace_value_policy_obs(
                trace,
                self.arrays['child'],
                self.obs_arrays['visit'],
                self.obs_arrays['value'],
                self.obs_arrays['policy_new'],
                self.node_to_obs,
                self.arrays['score'],
                value)
        else:
            backup_trace_value_policy(
                trace,
                self.arrays['child'],
                self.arrays['visit'],
                self.arrays['value'],
                self.arrays['policy_new'],
                self.arrays['score'],
                value)

    def remove_nodes(self):

        print('\nWARNING: REMOVING UNUSED NODES...', **perr)

        _c = get_all_childs(self.root, self.arrays['child'])

        self.update_available(_c)

        if not self.benchmark and self.online:
            if self.projection:
                nodes_to_store = self.obs_available
            else:
                nodes_to_store = self.available
            self.store_nodes(nodes_to_store)
            self.train_nodes()

        if self.node_saver:
            self.save_nodes(self.available)

        self.reset_arrays()

    def store_nodes(self, nodes):

        print('Storing unused nodes...', **perr)

        states, values, policy, weights = self.memory

        m_idx = self.memory_index

        if self.projection:
            _visit = self.obs_arrays['visit']
            _value = self.obs_arrays['value']
            _policy = self.obs_arrays['policy_new']
            _end = self.obs_arrays['end']
        else:
            _visit = self.arrays['visit']
            _value = self.arrays['value']
            _policy = self.arrays['policy_new']
            _end = self.arrays['end']
            g_arr = self.game_arr

        for idx in nodes:
            if _visit[idx] < self.min_visits_to_store or _end[idx]:
                continue
            if self.projection:
                states[m_idx, 0] = self.obs_arrays['state'][idx]
            else:
                states[m_idx, 0] = g_arr[idx].getState()
            values[m_idx] = _value[idx]
            policy[m_idx] = _policy[idx]
            weights[m_idx] = _visit[idx]

            m_idx += 1
            if m_idx >= self.memory_size:
                break

        print('{} nodes stored.'.format(m_idx - self.memory_index), **perr)

        self.memory_index = m_idx

    def train_nodes(self, batch_size=128, iters_per_val=500, loss_threshold=1, val_fraction=0.1, patience=10, growth_rate=5000, max_iters=100000, dump_data=True):

        print('Training...', **perr)

        states, values, policy, weights = self.memory
        weights = weights / weights[:self.memory_index].mean()

        d_size = self.memory_index
        m_size = min((self.n_trains + 1) * growth_rate, self.memory_size)
        if d_size >= m_size:
            print('Enough training data ({} >= {}), proceed to training.'.format(d_size, m_size), **perr)
        else:
            print('Not enough training data ({} < {}), collecting more data.'.format(d_size, m_size), **perr)
            return None

        if dump_data:
            np.savez('./data/dump', states=states[:d_size], values=values[:d_size], policy=policy[:d_size], weights=weights[:d_size])

        self.n_trains += 1

        self.model.training = True
        val_size = int(d_size * val_fraction)
        batch_val = [
                states[d_size-val_size:d_size],
                values[d_size-val_size:d_size],
                policy[d_size-val_size:d_size],
                weights[d_size-val_size:d_size]]

        print('Training data size: {}    Validation data size: {}'.format(d_size-val_size, val_size), **perr)

        iters = fails = 0
        l_val_min = float('inf')
        while fails < patience and iters < max_iters:
            l_avg = 0
            for it in range(iters_per_val):
                b_idx = np.random.choice(d_size - val_size, size=batch_size, replace=False)
                batch = [states[b_idx], values[b_idx], policy[b_idx], weights[b_idx]]
                loss = self.model.train(batch)
                l_avg += loss['loss']
            iters += iters_per_val
            l_avg /= iters_per_val
            l_val = self.model.compute_loss(batch_val)
            l_val_mean, l_val_std = l_val['loss'], l_val['loss_std'] / val_size ** 0.5
            if l_val_mean - l_val_min < l_val_std * loss_threshold:
                if l_val_mean < l_val_min:
                    self.model.save(verbose=False)
                    l_val_min = l_val_mean
                fails = 0
            else:
                fails += 1
            print('Iteration:{:6d}  training loss:{:.3f} validation loss:{:.3f}±{:.3f}'.format(iters, l_avg, l_val_mean, l_val_std), **perr)

        self.model.load()
        self.model.update_scheduler()
        self.model.training = False

        self.memory_index = 0

        print('Training complete.', **perr)
