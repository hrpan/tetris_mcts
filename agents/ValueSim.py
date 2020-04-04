import numpy as np
from agents.agent import TreeAgent
from agents.core import *
from agents.core_projection import *
from model.model_vv import Model_VV as Model
from sys import stderr

eps = 1e-7
perr = dict(file=stderr, flush=True)


class ValueSim(TreeAgent):

    def __init__(self, online=True, memory_size=250000, min_visits_to_store=10, **kwargs):

        super().__init__(max_nodes=100000, **kwargs)

        self.g_tmp = self.env(*self.env_args)

        self.online = online
        if online:

            self.memory_size = memory_size

            self.memory = [
                    np.zeros((memory_size, 1, 22, 10), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32)
                    ]

            self.memory_index = 0

            self.n_trains = 0

        self.min_visits_to_store = min_visits_to_store

        self.model = Model()
        self.model.load()

        self.inference = lambda state: self.model.inference(state[None, None, :, :])

    def evaluate_state(self, state):

        v, var = self.inference(state)

        return v[0][0], var[0][0]

    def mcts(self, root_index, sims):

        child = self.arrays['child']
        score = self.arrays['score']

        if self.projection:
            visit = self.obs_arrays['visit']
            value = self.obs_arrays['value']
            variance = self.obs_arrays['variance']
            n_to_o = self.node_to_obs
            selection = lambda: select_trace_obs(
                    root_index, child,
                    visit, value, variance,
                    score, n_to_o)
            backup = lambda trace, _value, _variance: backup_trace_obs(
                    trace, visit, value, variance,
                    n_to_o, score, _value, _variance)
        else:
            visit = self.arrays['visit']
            value = self.arrays['value']
            variance = self.arrays['variance']
            selection = lambda: select_trace(
                    root_index, child, visit, value, variance, score)
            backup = lambda trace, _value, _variance: backup_trace_obs(
                    trace, visit, value, variance, score, _value, _variance)

        for i in range(sims):
            trace = selection()

            leaf_index = trace[-1]

            leaf_game = self.game_arr[leaf_index]

            _value, _variance = leaf_game.score, 0
            if not leaf_game.end:

                v, _variance = self.evaluate_state(leaf_game.getState())

                _value += v

                self.expand(leaf_game)

            backup(trace, _value, _variance)

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

        states, values, variance, weights = self.memory

        m_idx = self.memory_index

        if self.projection:
            _visit = self.obs_arrays['visit']
            _value = self.obs_arrays['value']
            _variance = self.obs_arrays['variance']
            _end = self.obs_arrays['end']
        else:
            _visit = self.arrays['visit']
            _value = self.arrays['value']
            _variance = self.arrays['variance']
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
            variance[m_idx] = _variance[idx]
            weights[m_idx] = _visit[idx]

            m_idx += 1
            if m_idx >= self.memory_size:
                break

        print('{} nodes stored.'.format(m_idx - self.memory_index), **perr)

        self.memory_index = m_idx

    def train_nodes(self, batch_size=128, iters_per_val=500, loss_threshold=1, val_fraction=0.1, patience=10, growth_rate=5000, max_iters=100000, dump_data=True):

        print('Training...', **perr)

        states, values, variance, weights = self.memory
        weights = weights / weights[:self.memory_index].mean()

        d_size = self.memory_index
        m_size = min((self.n_trains + 1) * growth_rate, self.memory_size)
        if d_size >= m_size:
            print('Enough training data ({} >= {}), proceed to training.'.format(d_size, m_size), **perr)
        else:
            print('Not enough training data ({} < {}), collecting more data.'.format(d_size, m_size), **perr)
            return None

        if dump_data:
            np.savez('./data/dump', states=states[:d_size], values=values[:d_size], variance=variance[:d_size], weights=weights[:d_size])

        self.n_trains += 1

        self.model.training = True
        val_size = int(d_size * val_fraction)
        batch_val = [
                states[d_size-val_size:d_size],
                values[d_size-val_size:d_size],
                variance[d_size-val_size:d_size],
                weights[d_size-val_size:d_size]]

        print('Training data size: {}    Validation data size: {}'.format(d_size-val_size, val_size), **perr)

        iters = fails = 0
        l_val_min = float('inf')
        while fails < patience and iters < max_iters:
            l_avg = 0
            for it in range(iters_per_val):
                b_idx = np.random.choice(d_size - val_size, size=batch_size, replace=False)
                batch = [states[b_idx], values[b_idx], variance[b_idx], weights[b_idx]]
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
