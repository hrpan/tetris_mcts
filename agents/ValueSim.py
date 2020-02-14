import numpy as np
from collections import Counter, deque
from agents.agent import Agent
from agents.core import *
from agents.core_projection import *
from sys import stderr

eps = 1e-7
perr = dict(file=stderr, flush=True)


class ValueSim(Agent):

    def __init__(self, online=False, memory_size=250000, projection=True, **kwargs):

        super().__init__(**kwargs)

        self.g_tmp = self.env(*self.env_args)

        if online:
            self.online = online

            self.memory_size = memory_size

            self.memory = [
                    np.zeros((memory_size, 1, 22, 10), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32),
                    np.zeros((memory_size, 1), dtype=np.float32)
                    ]

            self.memory_index = 0

            self.n_trains = 0

        self.projection = projection
        if projection:
            self.node_to_obs = np.zeros(self.init_nodes, dtype=np.int32)
            self.obs_stats = np.zeros((self.init_nodes, 3), dtype=np.float32)
            self.obs_arr = np.zeros((self.init_nodes, 22, 10), dtype=np.int8)
            self.obs_index_dict = dict()
            self.obs_available = deque(range(1, self.init_nodes), maxlen=self.init_nodes)
            self.obs_occupied = deque([0], maxlen=self.init_nodes)

    def mcts(self, root_index):

        _child = self.arrs['child']
        _node_stats = self.arrs['node_stats']

        proj = self.projection
        if proj:
            o_stats = self.obs_stats
            n_to_o = self.node_to_obs
            trace = select_trace_obs(root_index, _child, _node_stats, o_stats, n_to_o)
        else:
            trace = select_trace(root_index, _child, _node_stats)

        leaf_index = trace[-1]

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore()
        if not leaf_game.end:

            v, var, p = self.evaluate_state(leaf_game.getState())

            _node_stats[leaf_index][3] = var
            if proj:
                o_stats[n_to_o[leaf_index]][2] = var

            value += v

            _g = self.g_tmp
            for i in range(self.n_actions):
                _g.copy_from(leaf_game)
                _g.play(i)
                _n = self.new_node(_g)
                _child[leaf_index][i] = _n
                _node_stats[_n][2] = _g.getScore()
                if proj:
                    self.new_obs(_g.getState(), _n)

        #backup_trace_3(trace, _node_stats, value, alpha=self.backup_alpha)
        if proj:
            backup_trace_obs(trace, _node_stats, value, o_stats, n_to_o)
        else:
            backup_trace_welford_v2(trace, _node_stats, value)

    def compute_stats(self, node=None):

        if node is None:
            node = self.root

        _stats = np.zeros((6, self.n_actions), dtype=np.float32)

        _childs = self.arrs['child'][node]
        _ns = self.arrs['node_stats']

        counter = Counter(_childs)

        root_score = _ns[node][2]

        _stats[2] = 0
        for i in range(self.n_actions):
            _idx = _childs[i]
            if _ns[_idx][0] < 1:
                return False
            v, var = self.get_value(_idx)
            _stats[0][i] = _ns[_idx][0] / counter[_idx]
            _stats[1][i] = v + _ns[_idx][2] - root_score
            _stats[3][i] = v + _ns[_idx][2] - root_score
            _stats[4][i] = var
            _stats[5][i] = _ns[_idx][4]

        return _stats

    def get_value(self, node=None):

        if node is None:
            node = self.root

        if self.projection:
            _os = self.obs_stats[self.node_to_obs[node]]
            return _os[1], _os[2]
        else:
            _ns = self.arrs['node_stats'][node]
            return _ns[1], _ns[3]

    def remove_nodes(self):

        print('\nWARNING: REMOVING UNUSED NODES...', **perr)

        _c = get_all_childs(self.root, self.arrs['child'])
        self.occupied.clear()
        self.occupied.extend(_c)
        self.available.clear()
        self.available.extend(i for i in range(self.max_nodes) if i not in _c)

        print('Number of occupied nodes: {}'.format(len(self.occupied)), **perr)
        print('Number of available nodes: {}'.format(len(self.available)), **perr)

        if not self.benchmark and self.online and not self.projection:
            self.store_nodes(self.available)
            self.train_nodes()

        if self.saver:
            self.save_nodes(self.available)

        arrs = self.arrs.values()
        for idx in self.available:
            _g = self.game_arr[idx]

            self.node_index_dict.pop(_g, None)

            for arr in arrs:
                arr[idx].fill(0)

    def store_nodes(self, nodes, min_visits=20):

        print('Storing unused nodes...', **perr)

        states, values, variance, weights = self.memory

        g_arr = self.game_arr
        n_stats = self.arrs['node_stats']
        child = self.arrs['child']

        actions = self.n_actions

        m_idx = self.memory_index
        for idx in nodes:
            ns = n_stats[idx]
            if ns[0] < min_visits or any(n_stats[child[idx][i]][0] < 1 for i in range(actions)):
                continue
            states[m_idx, 0] = g_arr[idx].getState()
            values[m_idx] = ns[1]
            variance[m_idx] = ns[3]
            weights[m_idx] = ns[0]

            m_idx += 1
            if m_idx >= self.memory_size:
                break

        print('{} nodes stored.'.format(m_idx - self.memory_index), **perr)

        self.memory_index = m_idx

    def train_nodes(self, batch_size=128, iters_per_val=100, loss_threshold=1, val_fraction=0.1, patience=10, growth_rate=2500, max_iters=100000):

        print('Training...', **perr)

        states, values, variance, weights = self.memory
        weights = weights / weights[:self.memory_index].mean()
        p_dummy = np.empty((batch_size, 7), dtype=np.float32)

        d_size = self.memory_index
        m_size = min((self.n_trains + 1) * growth_rate, self.memory_size)
        if d_size >= m_size:
            print('Enough training data ({} >= {}), proceed to training.'.format(d_size, m_size), **perr)
        else:
            print('Not enough training data ({} < {}), collecting more data.'.format(d_size, m_size), **perr)
            return None

        self.n_trains += 1

        self.model.training = True
        val_size = int(d_size * val_fraction)
        batch_val = [
                states[d_size-val_size:d_size],
                values[d_size-val_size:d_size],
                variance[d_size-val_size:d_size],
                np.empty((val_size, 7), dtype=np.float32),
                weights[d_size-val_size:d_size]]

        print('Training data size: {}    Validation data size: {}'.format(d_size-val_size, val_size), **perr)

        iters = fails = 0
        l_val_min = float('inf')
        while fails < patience and iters < max_iters:
            l_avg = 0
            for it in range(iters_per_val):
                b_idx = np.random.choice(d_size - val_size, size=batch_size)
                batch = [states[b_idx], values[b_idx], variance[b_idx], p_dummy, weights[b_idx]]
                loss = self.model.train(batch)
                l_avg += loss['loss']
            iters += iters_per_val
            l_avg /= iters_per_val
            l_val = self.model.compute_loss(batch_val)
            l_val_mean, l_val_std = l_val['loss'], l_val['loss_std'] / val_size ** 0.5
            #self.model.update_scheduler()
            if l_val_mean - l_val_min < l_val_std * loss_threshold:
                if l_val_mean < l_val_min:
                    self.model.save(verbose=False)
                    l_val_min = l_val_mean
                fails = 0
            else:
                fails += 1
            print('Iteration:{:6d}  training loss:{:.3f} validation loss:{:.3f}±{:.3f}'.format(iters, l_avg, l_val_mean, l_val_std), **perr)
        #self.model.save(verbose=False)
        self.model.load()
        self.model.update_scheduler()
        self.model.training = False

        self.memory_index = 0

        print('Training complete.', **perr)

    def new_obs(self, state, nidx):

        key = state.tobytes()

        idx = self.obs_index_dict.get(key)

        if not idx:

            if not self.obs_available:
                self.remove_obs()

            idx = self.obs_available.pop()

            self.obs_arr[idx] = state

            self.obs_index_dict[key] = idx

            self.obs_occupied.append(idx)

        self.node_to_obs[nidx] = idx

        return idx

    def remove_obs(self):

        print('\nWARNING: REMOVING UNUSED OBSERVATION NODES...', **perr)

        _c = get_occupied_obs(self.root, self.arrs['child'], self.node_to_obs)
        self.obs_occupied.clear()
        self.obs_occupied.extend(_c)
        self.obs_available.clear()
        self.obs_available.extend(i for i in range(self.max_nodes) if i not in _c)

        print('Number of occupied observation nodes: {}'.format(len(self.obs_occupied)), **perr)
        print('Number of available observation nodes: {}'.format(len(self.obs_available)), **perr)

        if not self.benchmark and self.online:
            self.store_obs(self.obs_available)
            self.train_nodes()

        for idx in self.obs_available:
            s = self.obs_arr[idx]

            self.obs_index_dict.pop(s.tobytes(), None)

            self.obs_stats[idx].fill(0)

    def store_obs(self, obs, min_visits=20):

        print('Storing unused observations...', **perr)

        states, values, variance, weights = self.memory

        stats = self.obs_stats

        m_idx = self.memory_index
        for idx in obs:
            s = stats[idx]
            if s[0] < min_visits:
                continue
            states[m_idx, 0] = self.obs_arr[idx]
            values[m_idx] = s[1]
            variance[m_idx] = s[2]
            weights[m_idx] = s[0]

            m_idx += 1
            if m_idx >= self.memory_size:
                break

        print('{} observations stored.'.format(m_idx - self.memory_index), **perr)

        self.memory_index = m_idx
