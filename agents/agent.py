import numpy as np
from sys import stderr
from agents.core import get_all_childs
from collections import deque

perr = dict(file=stderr, flush=True)


class Agent:

    def __init__(self, sims=100, init_nodes=500000, backend='pytorch',
                 env=None, env_args=None, n_actions=7,
                 saver=None, stochastic_inference=False,
                 min_visits=30, benchmark=False, **kwargs):

        self.sims = sims

        self.init_nodes = init_nodes

        self.backend = backend

        self.env = env
        self.env_args = env_args

        self.episode = 0

        self.min_visits = min_visits

        self.n_actions = n_actions

        self.saver = saver

        self.stochastic_inference = stochastic_inference

        self.benchmark = benchmark

        self.init_array()
        self.init_model()

    def init_array(self):

        self.arrs = {
                'child': np.zeros((self.init_nodes, self.n_actions), dtype=np.int32),
                'child_stats': np.zeros((self.init_nodes, 6, self.n_actions), dtype=np.float32),
                'node_stats': np.zeros((self.init_nodes, 5), dtype=np.float32),
                'node_ep': np.zeros((self.init_nodes, ), dtype=np.int32)
                }

        self.game_arr = [self.env(*self.env_args) for i in range(self.init_nodes)]

        self.available = deque(range(1, self.init_nodes), maxlen=self.init_nodes)
        self.occupied = deque([0], maxlen=self.init_nodes)

        self.node_index_dict = dict()

        self.max_nodes = self.init_nodes

    def init_model(self):

        if self.backend == 'pytorch':
            from model.model_pytorch import Model
            self.model = Model()
            self.model.load()

            if self.stochastic_inference:
                self.inference = lambda state: self.model.inference_stochastic(state[None, None, :, :])
            else:
                self.inference = lambda state: self.model.inference(state[None, None, :, :])

        else:
            self.model = None

    def evaluate(self, node):

        state = node.game.getState()

        return self.evaluate_state(state)

    def evaluate_state(self, state):

        v, var, p = self.inference(state)

        return v[0][0], var[0][0], p[0]

    def expand_nodes(self, n_nodes=10000):

        print('\nWARNING: ADDING EXTRA NODES...', **perr)

        for k, arr in self.arrs.items():
            _s = arr.shape
            _new_s = [_ for _ in _s]
            _new_s[0] = n_nodes
            _temp_arr = np.zeros(_new_s, dtype=arr.dtype)
            self.arrs[k] = np.concatenate([arr, _temp_arr])

        self.game_arr += [self.env(*self.env_args) for i in range(n_nodes)]
        self.available += [i for i in range(self.max_nodes, self.max_nodes+n_nodes)]
        self.max_nodes += n_nodes

    def new_node(self, game):

        idx = self.node_index_dict.get(game)

        if not idx:

            if self.available:
                idx = self.available.pop()
            else:
                self.remove_nodes()
                if self.available:
                    idx = self.available.pop()
                else:
                    self.expand_nodes()
                    idx = self.available.pop()

            _g = self.game_arr[idx]

            _g.copy_from(game)

            self.arrs['node_ep'][idx] = self.episode

            self.node_index_dict[_g] = idx

            self.occupied.append(idx)

        return idx

    def mcts(self, root_index):
        pass

    def play(self):

        for i in range(self.sims):
            self.mcts(self.root)

        self.stats = self.compute_stats()

        if np.all(self.stats[3] == 0):
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.stats[3])

        return action

    def compute_stats(self):
        _stats = np.zeros((6, self.n_actions))

        _childs = self.arrs['child'][self.root]
        _ns = self.arrs['node_stats']

        for i in range(self.n_actions):
            _idx = _childs[i]
            _stats[0][i] = _ns[_idx][0]
            _stats[1][i] = _ns[_idx][1]
            _stats[2][i] = 0
            _stats[3][i] = _ns[_idx][1]
            _stats[4][i] = _ns[_idx][3]
            _stats[5][i] = _ns[_idx][4]

        return _stats

    def get_prob(self):

        return self.stats[0] / np.sum(self.stats[0])

    def get_stats(self):

        return np.copy(self.stats)

    def get_value(self):

        print('\nWATNING: get_value not implemented for this agent', **perr)

        return 0, 0

    def remove_nodes(self):

        print('\nWARNING: REMOVING UNUSED NODES...', **perr)

        _c = get_all_childs(self.root, self.arrs['child'])
        self.occupied.clear()
        self.occupied.extend(_c)
        self.available.clear()
        self.available.extend(i for i in range(self.max_nodes) if i not in _c)

        print('Number of occupied nodes: {}'.format(len(self.occupied)), **perr)
        print('Number of available nodes: {}\n'.format(len(self.available)), **perr)

        if self.saver:
            self.save_nodes(self.available)

        arrs = self.arrs.values()
        for idx in self.available:
            _g = self.game_arr[idx]

            self.node_index_dict.pop(_g, None)

            for arr in arrs:
                arr[idx].fill(0)

    def save_nodes(self, nodes_to_save):

        saver = self.saver

        node_stats = self.arrs['node_stats']

        node_ep = self.arrs['node_ep']

        for idx in nodes_to_save:

            if node_stats[idx][0] < self.min_visits:
                continue

            _tmp_stats = self.compute_stats(idx)
            if _tmp_stats is False:
                continue

            _g = self.game_arr[idx]

            v, var = self.get_value(idx)

            saver.add_raw(node_ep[idx],
                          _g.getState(),
                          _tmp_stats[0] / _tmp_stats[0].sum(),
                          np.argmax(_tmp_stats[1]),
                          _g.combo,
                          _g.line_clears,
                          _g.line_stats,
                          _g.score,
                          _tmp_stats,
                          v,
                          var)

    def save_occupied(self):

        if self.saver:
            self.save_nodes(self.occupied)

    def set_root(self, game):

        self.root = self.new_node(game)

    def update_root(self, game, episode=0):

        self.episode = episode

        self.set_root(game)

        self.arrs['node_stats'][self.root][2] = game.score

    def close(self):

        if self.saver:
            self.save_occupied()
            self.saver.close()
