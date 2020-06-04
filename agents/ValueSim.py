import numpy as np
from agents.agent import TreeAgent
from agents.core import select_trace, backup_trace
from agents.cppmodule.core import get_all_childs, select_trace_obs, backup_trace_obs
from model.model_vv import Model_VV as Model
from sys import stderr

eps = 1e-7
perr = dict(file=stderr, flush=True)


class ValueSim(TreeAgent):

    def __init__(self, online=True, memory_size=500000, min_visits_to_store=10, gamma=0.999, memory_growth_rate=5000, **kwargs):

        super().__init__(max_nodes=100000, **kwargs)

        self.g_tmp = self.env(*self.env_args)

        self.online = online
        if online:

            self.memory_size = memory_size

            self.memory = [
                    np.zeros((memory_size, 1, *self.env_args[0]), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32)
                    ]

            self.memory_index = 0

            self.n_trains = 0
            self.memory_growth_rate = memory_growth_rate
            self.last_episode = 0

        self.min_visits_to_store = min_visits_to_store

        self.gamma = gamma

        self.model = Model()
        self.model.load()
        self.model.training(False)

    def evaluate_state(self, state):

        v, var = self.model.inference(state[None, None, :, :])

        return v[0][0], var[0][0]

    def mcts(self, root_index, sims):

        child = self.arrays['child']
        score = self.arrays['score']

        if self.projection:
            visit = self.obs_arrays['visit']
            value = self.obs_arrays['value']
            variance = self.obs_arrays['variance']
            n_to_o = self.node_to_obs
            s_args = [root_index, child, visit, value, variance, score, n_to_o, 1]
            selection = select_trace_obs
            b_args = [None, visit, value, variance, n_to_o, score, None, None, self.gamma]
            #backup = backup_trace_mixture_obs
            backup = backup_trace_obs
        else:
            visit = self.arrays['visit']
            value = self.arrays['value']
            variance = self.arrays['variance']
            s_args = [root_index, child, visit, value, variance, score]
            selection = select_trace
            b_args = [None, visit, value, variance, score, None, None, self.gamma]
            backup = backup_trace

        for i in range(sims):
            trace = selection(*s_args)

            leaf_index = trace[-1]

            leaf_game = self.game_arr[leaf_index]

            _value, _variance = leaf_game.score, 0
            if not leaf_game.end:

                v, _variance = self.evaluate_state(leaf_game.getState())

                _value += v

                self.expand(leaf_game)
            b_args[0] = trace
            b_args[-3] = _value
            b_args[-2] = _variance
            backup(*b_args)
            #print('{:3d} {:3d} {} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            #    i, visit[n_to_o[self.root]], self.root,value[n_to_o[self.root]],
            #    variance[n_to_o[self.root]], _value - score[self.root], _variance))
        #print(self.compute_stats().astype(int))
        #input()

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

    def train_nodes(self, dump_data=True):

        print('Training...', **perr)

        states, values, variance, weights = self.memory

        d_size = self.memory_index

        m_size = min(self.n_trains * self.memory_growth_rate, self.memory_size)
        if d_size >= m_size:
            print('Enough training data ({} >= {}), proceed to training.'.format(d_size, m_size), **perr)
        else:
            print('Not enough training data ({} < {}), collecting more data.'.format(d_size, m_size), **perr)
            return None

        if dump_data:
            np.savez('./data/dump', states=states[:d_size], values=values[:d_size], variance=variance[:d_size], weights=weights[:d_size])

        self.n_trains += 1
        self.model.train_data([d[:d_size] for d in self.memory], iters_per_val=100, batch_size=1024, max_iters=50000)
        self.model.training(False)

        self.memory_index = 0

        print('Training complete.', **perr)
