import sys
import numpy as np
from agents.agent import Agent
from agents.core import select_index_2, backup_trace, choose_action, fill_child_stats, update_child_info, get_all_child_2

n_actions = 6

np.set_printoptions(precision=2, suppress=True)

class ValueSim2(Agent):

    def __init__(self,conf,sims,tau=None,backend='pytorch'):

        init_nodes = 50000

        super().__init__(sims=sims, backend=backend, init_nodes=init_nodes)

        child_length = 1000
        child_info_arr = np.zeros((init_nodes, n_actions, child_length, 2), dtype=np.uint16)
        self.arrs['child_info'] = child_info_arr

    def mcts(self,root_index):

        _g = self.game_arr[root_index].clone()

        trace, action = select_index_2(_g, self.node_index_dict, self.arrs['node_stats'], self.arrs['child_info'])

        leaf_game = _g

        value = leaf_game.getScore() #- self.game_arr[root_index].getScore()

        idx = self.new_node(leaf_game)

        self.arrs['node_stats'][idx][2] = value

        trace.append(idx)

        update_status = update_child_info(trace, action, self.arrs['child_info'])

        if not update_status:
            sys.stderr.write('\nWARNING: CHILD_INFO FULL\n')

        if not leaf_game.end:

            v, p = self.evaluate_state(leaf_game.getState())

            value += v

           
        
        backup_trace(trace, self.arrs['node_stats'], value)

    def compute_stats(self):

        return fill_child_stats(self.root,
            self.arrs['node_stats'],
            self.arrs['child_info']) 

    def expand_nodes(self, n_nodes=100000):

        super().expand_nodes(n_nodes)

    def remove_nodes(self):

        sys.stderr.write('\nWARNING: REMOVING UNUSED NODES...\n')

        c_set = get_all_child_2(self.root, self.arrs['child_info'])

        self.occupied = list(c_set)
        sys.stderr.write('Number of occupied nodes: ' + str(len(self.occupied)) + '\n')
        self.available = list(set(range(self.max_nodes)) - c_set)
        sys.stderr.write('Number of available nodes: ' + str(len(self.available)) + '\n')

        for idx in self.available:
            _g = self.game_arr[idx]

            del self.node_index_dict[_g]
            
            self.arrs['child'][idx] = 0
            self.arrs['child_stats'][idx] = 0
            self.arrs['node_stats'][idx] = 0
            self.arrs['child_info'][idx] = 0
        
