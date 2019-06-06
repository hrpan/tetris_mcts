import collections
import numpy as np
from agents.agent import Agent
from agents.core import *

eps = 1e-7

class ValueSimBayes(Agent):

    def __init__(self, conf, sims, tau=None, backend='pytorch', env=None, env_args=None, min_n=2, n_actions=7):

        super().__init__(sims=sims,backend=backend,env=env, env_args=env_args, n_actions=n_actions)

        self.g_tmp = env(*env_args)

        self.min_n = min_n

    def mcts(self,root_index):

        _child = self.arrs['child']
        _node_stats = self.arrs['node_stats']

        trace = select_index_bayes(root_index, _child, _node_stats, self.min_n)

        leaf_index = trace[-1]

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore()

        if not leaf_game.end:

            v, var, p = self.evaluate_state(leaf_game.getState())

            value += v
            
            _g = self.g_tmp
            for i in range(self.n_actions):
                _g.copy_from(leaf_game)
                _g.play(i)
                _n = self.new_node(_g)
                _child[leaf_index][i] = _n
                _node_stats[_n][2] = _g.getScore()

        backup_trace_welford(trace, _node_stats, value)

    def compute_stats(self, node=None):

        if node == None:
            node = self.root

        _stats = np.zeros((6, self.n_actions), dtype=np.float32)

        _childs = self.arrs['child'][node]
        _ns = self.arrs['node_stats']

        counter = collections.Counter(_childs)        

        root_score = _ns[node][2]        

        _stats[2] = 0
        for i in range(self.n_actions):
            _idx = _childs[i]
            if _ns[_idx][0] < 1:
                return False
            _stats[0][i] = _ns[_idx][0] / counter[_idx]
            _stats[1][i] = _ns[_idx][1] + _ns[_idx][2] - root_score 
            _stats[3][i] = _ns[_idx][1] + _ns[_idx][2] - root_score
            _stats[4][i] = _ns[_idx][3] / _ns[_idx][0]
            _stats[5][i] = _ns[_idx][4]

        return _stats

    def get_value(self, node=None):
        
        if node == None:
            node = self.root

        _ns = self.arrs['node_stats'][node]

        return _ns[1], _ns[3] / _ns[0] 

