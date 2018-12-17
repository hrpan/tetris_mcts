import collections
import numpy as np
from agents.agent import Agent
from agents.core import select_index_3, backup_trace_3, choose_action
n_actions = 6
eps = 1e-7

class ValueSim(Agent):

    def __init__(self, conf, sims, tau=None, backend='pytorch', env=None, env_args=None):

        super().__init__(sims=sims,backend=backend,env=env, env_args=env_args)

        self.g_tmp = env(*env_args)

    def mcts(self,root_index):

        _child = self.arrs['child']
        _node_stats = self.arrs['node_stats']

        trace = select_index_3(root_index, _child, _node_stats)

        leaf_index = trace[-1]

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore()

        if not leaf_game.end:

            v, var, p = self.evaluate_state(leaf_game.getState())

            _node_stats[leaf_index][3] = var

            value += v
            
            _g = self.g_tmp
            for i in range(n_actions):
                _g.copy_from(leaf_game)
                _g.play(i)
                _n = self.new_node(_g)
                _child[leaf_index][i] = _n
                _node_stats[_n][2] = _g.getScore()

        backup_trace_3(trace, _node_stats, value)

    def compute_stats(self):
        _stats = np.zeros((6, n_actions), dtype=np.float32)

        _childs = self.arrs['child'][self.root]
        _ns = self.arrs['node_stats']

        counter = collections.Counter(_childs)        

        _stats[2] = 0
        for i in range(n_actions):
            _idx = _childs[i]
            if counter[_idx] == 0:
                _stats[0][i] = 0
                _stats[1][i] = 0
            else:
                _stats[0][i] = _ns[_idx][0] / counter[_idx]
                _stats[1][i] = _ns[_idx][1] * _ns[_idx][0]
            _stats[3][i] = _ns[_idx][1]
            _stats[4][i] = _ns[_idx][3]
            _stats[5][i] = _ns[_idx][4]

        return _stats

    def get_value(self):

        _ns = self.arrs['node_stats'][self.root]

        return _ns[1], _ns[3] 

    def update_root(self, game):
        
        self.root = self.new_node(game)
        self.arrs['node_stats'][self.root][2] = game.getScore()
