import numpy as np
import agents.helper
from agents.cppmodule.core import get_all_childs
from sys import stderr
from collections import deque

perr = dict(file=stderr, flush=True)


class Agent:

    def __init__(self, n_actions=7, benchmark=False, **kwargs):

        self.episode = 0
        self.n_actions = n_actions
        self.benchmark = benchmark

    def play(self):
        raise NotImplementedError('update_root not implemented')

    def get_action(self):
        raise NotImplementedError('get_action not implemented')

    def get_prob(self):
        raise NotImplementedError('get_action not implemented')

    def update_root(self, game):
        raise NotImplementedError('update_root not implemented')

    def close(self):
        raise NotImplementedError('close not implemented')


class TreeAgent(Agent):

    def __init__(self, sims=100, max_nodes=500000, env=None, env_args=None,
                 node_saver=None, projection=True, min_visits=30, **kwargs):

        super().__init__(**kwargs)

        self.sims = sims

        self.max_nodes = max_nodes

        self.env = env
        self.env_args = env_args

        self.episode = 0

        self.min_visits = min_visits

        self.node_saver = node_saver

        self.projection = projection

        self.init_array()

    def init_array(self):

        self.arrays = {
                'child': np.zeros((self.max_nodes, self.n_actions), dtype=np.int32),
                'visit': np.zeros(self.max_nodes, dtype=np.int32),
                'value': np.zeros(self.max_nodes, dtype=np.float32),
                'variance': np.zeros(self.max_nodes, dtype=np.float32),
                'episode': np.zeros(self.max_nodes, dtype=np.int32),
                'score': np.zeros(self.max_nodes, dtype=np.float32),
                'end': np.zeros(self.max_nodes, dtype=bool)
                }

        self.game_arr = [self.env(*self.env_args) for i in range(self.max_nodes)]

        self.available = deque(range(1, self.max_nodes), maxlen=self.max_nodes)
        self.occupied = deque([0], maxlen=self.max_nodes)

        self.node_index_dict = dict()

        if self.projection:
            self.node_to_obs = np.zeros(self.max_nodes, dtype=np.int32)
            self.obs_arrays = {
                    'state': np.zeros((self.max_nodes, *self.env_args[0]), dtype=np.int8),
                    'visit': np.zeros(self.max_nodes, dtype=np.int32),
                    'value': np.zeros(self.max_nodes, dtype=np.float32),
                    'variance': np.zeros(self.max_nodes, dtype=np.float32),
                    'end': np.zeros(self.max_nodes, dtype=bool),
                }
            self.obs_index_dict = dict()
            self.obs_available = deque(range(1, self.max_nodes), maxlen=self.max_nodes)
            self.obs_occupied = deque([0], maxlen=self.max_nodes)

    def new_node(self, game):

        idx = self.node_index_dict.get(game)

        if not idx:

            if not self.available:
                self.remove_nodes()

            idx = self.available.pop()

            _g = self.game_arr[idx]

            _g.copy_from(game)

            self.arrays['episode'][idx] = self.episode
            self.arrays['score'][idx] = game.score

            self.node_index_dict[_g] = idx

            self.occupied.append(idx)

            if self.projection:

                state = _g.getState()
                key = state.tobytes()

                o_idx = self.obs_index_dict.get(key)
                if not o_idx:

                    o_idx = self.obs_available.pop()

                    self.obs_arrays['state'][o_idx] = state
                    self.obs_arrays['end'][o_idx] = _g.end

                    self.obs_index_dict[key] = o_idx
                    self.obs_occupied.append(o_idx)

                self.node_to_obs[idx] = o_idx

        return idx

    def mcts(self):

        raise NotImplementedError('mcts not implemented for this agent.')

    def expand(self, game):

        idx = self.new_node(game)

        _g = self.g_tmp
        child = self.arrays['child'][idx]
        for i in range(self.n_actions):
            _g.copy_from(game)
            _g.play(i)
            child[i] = self.new_node(_g)

    def play(self):

        self.mcts(self.root, self.sims)

        return self.get_action()

    def compute_stats(self, idx=None):

        if idx is None:
            idx = self.root

        _stats = np.zeros((3, self.n_actions), dtype=np.float32)

        child = self.arrays['child'][idx]
        if self.projection:
            _c = [self.node_to_obs[i] for i in child]
            visit = self.obs_arrays['visit']
            value = self.obs_arrays['value']
            variance = self.obs_arrays['variance']
        else:
            _c = child
            visit = self.arrays['visit']
            value = self.arrays['value']
            variance = self.arrays['variance']
        score = self.arrays['score']
        score_diff = [score[i] - score[idx] for i in child]

        for i, c in enumerate(_c):
            _stats[0][i] = visit[c]
            _stats[1][i] = value[c] + score_diff[i]
            _stats[2][i] = variance[c]

        return _stats

    def get_action(self):

        self.stats = self.compute_stats()

        return np.argmax(self.stats[1])

    def get_prob(self):

        return self.stats[0] / np.sum(self.stats[0])

    def get_stats(self):

        return np.copy(self.stats)

    def get_value_and_variance(self, node=None):

        if node is None:
            node = self.root

        if self.projection:
            o = self.node_to_obs[node]
            return self.obs_arrays['value'][o], self.obs_arrays['variance'][o]
        else:
            return self.arrays['value'][node], self.arrays['variance'][node]

    def update_available(self, occupied):

        self.occupied.clear()
        self.occupied.extend(occupied)

        self.available.clear()
        self.available.extend(i for i in range(self.max_nodes) if i not in occupied)

        print('Number of occupied nodes: {}'.format(len(self.occupied)), **perr)
        print('Number of available nodes: {}'.format(len(self.available)), **perr)

        if self.projection:
            _o = set(self.node_to_obs[i] for i in occupied)
            self.obs_occupied.clear()
            self.obs_occupied.extend(_o)
            self.obs_available.clear()
            self.obs_available.extend(i for i in range(self.max_nodes) if i not in _o)

            print('Number of occupied observation nodes: {}'.format(len(self.obs_occupied)), **perr)
            print('Number of available observation nodes: {}'.format(len(self.obs_available)), **perr)

    def reset_arrays(self):

        arr = self.game_arr
        pop = self.node_index_dict.pop
        for idx in self.available:
            pop(arr[idx], None)

        for arr in self.arrays.values():
            arr[self.available] = 0

        if self.projection:
            arr = self.obs_arrays['state']
            pop = self.obs_index_dict.pop
            for idx in self.obs_available:
                pop(arr[idx].tobytes(), None)

            for arr in self.obs_arrays.values():
                arr[self.obs_available] = 0

    def remove_nodes(self):

        print('\nWARNING: REMOVING UNUSED NODES...', **perr)

        occupied = get_all_childs(self.root, self.arrays['child'])

        self.update_available(occupied)

        if self.node_saver:
            self.save_nodes(self.available)

        self.reset_arrays()

    def save_nodes(self, nodes_to_save):

        saver = self.node_saver

        episode = self.arrays['episode']
        game = self.game_arr
        visit = self.arrays['visit']
        value = self.arrays['value']
        variance = self.arrays['variance']
        end = self.arrays['end']

        for idx in nodes_to_save:

            if visit[idx] < self.min_visits or end[idx]:
                continue

            stats = self.compute_stats(idx)

            _g = game[idx]

            saver.add_raw(episode[idx],
                          _g.getState(),
                          stats[0] / stats[0].sum(),
                          np.argmax(stats[1]),
                          _g.combo,
                          _g.line_clears,
                          _g.line_stats,
                          _g.score,
                          stats,
                          value[idx],
                          variance[idx])

    def save_occupied(self):

        if self.node_saver:
            self.save_nodes(self.occupied)

    def update_root(self, game):

        self.root = self.new_node(game)

        if game.end:
            self.episode += 1

    def close(self):

        if self.node_saver:
            self.save_occupied()
            self.node_saver.close()
