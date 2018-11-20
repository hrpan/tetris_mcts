import collections
import numpy as np
from agents.agent import Agent
from agents.core import select_index, backup_trace, choose_action
n_actions = 6

class ValueSim(Agent):

    def __init__(self, conf, sims, tau=None, backend='pytorch', env=None, env_args=None):

        super().__init__(sims=sims,backend=backend,env=env, env_args=env_args)

        self.g_tmp = env(*env_args) 
    def mcts(self,root_index):

        _child = self.arrs['child']
        _node_stats = self.arrs['node_stats']

        trace = select_index(root_index, _child, _node_stats)

        leaf_index = trace[-1]
        #leaf_index = select_index(root_index,self.arrs['child'],self.arrs['child_stats'])

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore() #- self.game_arr[root_index].getScore()

        if not leaf_game.end:

            v, p = self.evaluate_state(leaf_game.getState())

            value += v
            
            _g = self.g_tmp
            for i in range(n_actions):
                _g.copy_from(leaf_game)
                _g.play(i)
                _n = self.new_node(_g)
                _child[leaf_index][i] = _n
                _node_stats[_n][2] = _g.getScore()

        backup_trace(trace, _node_stats, value)

    def compute_stats(self):
        _stats = np.zeros((6, n_actions))

        _childs = self.arrs['child'][self.root]
        _ns = self.arrs['node_stats']

        counter = collections.Counter(_childs)        

        _stats[2] = 0
        for i in range(n_actions):
            _idx = _childs[i]
            _stats[0][i] = _ns[_idx][0] / counter[_idx]
            _stats[1][i] = _ns[_idx][1] / counter[_idx]
            _stats[3][i] = _ns[_idx][1] / _ns[_idx][0]
            _stats[4][i] = _ns[_idx][3]
            _stats[5][i] = _ns[_idx][4]

        return _stats
        
