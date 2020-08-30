import collections
import numpy as np
from agents.agent import Agent
from agents.core import *
from agents.core_distributional import *
from model.ValueSim import perr, ValueSim

eps = 1e-7


class DistValueSim(ValueSim):

    def __init__(self, atoms=50, vmin=0, vmax=5000, **kwargs):

        self.atoms = atoms

        super().__init__(stochastic_inference=False, **kwargs)

        self.arrays['value'] = np.zeros((len(), atoms), dtype=np.float32)

        if self.projection:
            self.obs_arrays['value'] = np.zeros((self.memory_size, atoms), dtype=np.float32)

        self.vrange = (vmin, vmax)

        self.v_dummy = np.zeros((atoms, ), dtype=np.float32)
        self.v_dummy[0] = 1

        from model.model_distributional import Model
        self.model = Model(atoms=self.atoms)
        self.model.load()
        self.model.training(False)

        self.inference = lambda state: self.model.inference(state[None, None, :, :])

    def mcts(self, root_index, sims):

        vmin, vmax = self.vrange

        child = self.arrays['child']
        score = self.arrays['score']

        if self.projection:
            visit = self.obs_arrays['visit']
            value = self.obs_arrays['value']
            n_to_o = self.node_to_obs
            s_args = [root_index, child, visit, value, score, n_to_o]
            selection = select_trace_obs_dist
            b_args = []
            backup = backup_trace_obs_dist
        else:
            raise NotImplementedError
        _child = self.arrs['child']
        _node_stats = self.arrs['node_stats']
        _node_dist = self.arrs['node_dist']

        for i in range(sims):
            trace = selection(*s_args)

            leaf_index = trace[-1]

            leaf_game = self.game_arr[leaf_index]

            value = leaf_game.getScore()

            if not leaf_game.end:

                v = self.model.inference(state[None, None, :, :])

                self.expand(leaf_game)
            else:
                v_dist = self.v_dummy

            backup(*b_args)

    def compute_stats(self, node=None):

        if node is None:
            node = self.root

        _stats = np.zeros((6, self.n_actions), dtype=np.float32)

        _childs = self.arrs['child'][node]
        _ns = self.arrs['node_stats']
        _nd = self.arrs['node_dist']

        vmin, vmax = self.vrange

        counter = collections.Counter(_childs)

        root_score = _ns[node][2]

        for i in range(self.n_actions):
            _idx = _childs[i]
            if _ns[_idx][0] < 1:
                return False
            _stats[0][i] = _ns[_idx][0] / counter[_idx]
            _stats[1][i] = _ns[_idx][1] + _ns[_idx][2] - root_score
            _stats[3][i] = _stats[1][i]
            _stats[4][i] = _ns[_idx][3]
        #print(_stats[[0, 1, 4]].astype(np.int))
        #print(_nd[self.root])
        #print(mean_variance(_nd[self.root], *self.vrange))
        return _stats

    def get_value(self, node=None):

        if node is None:
            node = self.root

        _nd = self.arrays['value'][node]

        mean, var = mean_variance(_nd, *self.vrange)

        return mean, var

#    def store_nodes(self, nodes, min_visits=50):
#
#        print('Storing unused nodes...', **perr)
#
#        states, values, weights = self.memory
#
#        g_arr = self.game_arr
#        n_stats = self.arrs['node_stats']
#        n_dist = self.arrs['node_dist']
#        child = self.arrs['child']
#
#        actions = self.n_actions
#
#        m_idx = self.memory_index
#        for idx in nodes:
#            ns = n_stats[idx]
#            if ns[0] < min_visits or any(n_stats[child[idx][i]][0] < 1 for i in range(actions)):
#                continue
#            states[m_idx, 0] = g_arr[idx].getState()
#            values[m_idx] = n_dist[idx]
#            weights[m_idx] = ns[0]
#
#            m_idx += 1
#            if m_idx >= self.memory_size:
#                break
#        print('{} nodes stored.'.format(m_idx - self.memory_index), **perr)
#
#        self.memory_index = m_idx

    def train_nodes(self, dump_data=True):

        print('Training...', **perr)

        states, values, _, weights = self.memory

        d_size = self.memory_index
        m_size = min(self.n_trains * self.memory_growth_rate, self.memory_size)
        if d_size >= m_size:
            print('Enough training data ({} >= {}), proceed to training.'.format(d_size, m_size), **perr)
        else:
            print('Not enough training data ({} < {}), collecting more data.'.format(d_size, m_size), **perr)
            return None

        if dump_data:
            np.savez('./data/memory_dump', states=states[:d_size], values=values[:d_size], weights=weights[:d_size])

        self.n_trains += 1

        data = [states, values, weights]
        self.model.train_data([d[:d_size] for d in data], iters_per_val=100, batch_size=1024, max_iters=50000)
        self.model.training(False)

        self.memory_index = 0
        print('Training complete.', **perr)
