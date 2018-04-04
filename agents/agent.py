import numpy as np
import sys
from core import get_all_childs

n_actions = 6

class Agent:
    def __init__(self,sims,init_nodes=1000000):

        self.sims = sims
        
        self.init_nodes = init_nodes
        
        child_arr = np.zeros((init_nodes,n_actions),dtype=np.int32)
        node_stats_arr = np.zeros((init_nodes,3),dtype=np.float32)

        self.arrs = {
                'child':child_arr,
                'node_stats':node_stats_arr,
                }

        self.game_arr = [None] * init_nodes

        self.available = [i for i in range(1,init_nodes)]

        self.occupied = [0]

        self.node_index_dict = dict()

        self.max_nodes = init_nodes

    def expand_nodes(self,n_nodes=10000):

        sys.stdout.write('Adding more nodes...\n')
        sys.stdout.flush()

        for k, arr in self.arrs.items():
            _s = arr.shape
            _new_s = [_ for _ in _s]
            _new_s[0] = n_nodes
            _temp_arr = np.zeros(_new_s,dtype=arr.dtype)
            self.arrs[k] = np.concatenate([arr,_temp_arr])

        self.game_arr += [None] * n_nodes
        self.available += [i for i in range(self.max_nodes,self.max_nodes+n_nodes)]
        self.max_nodes += n_nodes

    def new_node(self,game):

        if game in self.node_index_dict:

            return self.node_index_dict[game]

        else:

            if self.available:
                idx = self.available.pop()
            else:
                self.remove_nodes()
                if self.available:
                    ifx = self.available.pop()
                else:
                    self.expand_nodes()
                    idx = self.available.pop()

            _g = game.clone()

            self.node_index_dict[_g] = idx

            self.game_arr[idx] = _g

            self.occupied.append(idx)

            return idx

    def mcts(self,root_index):
        pass

    def play(self):

        for i in range(self.sims):
            self.mcts(self.root)

        self.stats = self.compute_stats()

        if np.all(self.stats[3] == 0):
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(self.stats[3])

        return action

    def compute_stats(self):

        _stats = np.zeros((4,n_actions))

        _childs = self.arrs['child'][self.root]
        _ns = self.arrs['node_stats']

        for i in range(n_actions):
            _idx = _childs[i]
            _stats[0][i] = _ns[_idx][0]
            _stats[1][i] = _ns[_idx][1]
            _stats[2][i] = 0
            _stats[3][i] = _ns[_idx][1] / _ns[_idx][0]

        return _stats

    def get_prob(self):

        return self.stats[0] / np.sum(self.stats[0])

    def get_stats(self):

        return self.stats

    def remove_nodes(self):

        _c = get_all_childs(self.root,self.arrs['child'])

        for idx in self.occupied:

            if idx not in _c:

                _g = self.game_arr[idx]

                del self.node_index_dict[_g]

                self.game_arr[idx] = None

                for k, arr in self.arrs.items():
                    arr[idx] = 0

                self.occupied.remove(idx)
                self.available.append(idx)

    def set_root(self,game):

        self.root = self.new_node(game)

    def update_root(self,game):

        self.set_root(game)


