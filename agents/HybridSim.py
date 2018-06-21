import numpy as np
from agents.agent import Agent
from agents.core import select_index, backup_trace, choose_action
np.set_printoptions(precision=2)
n_actions = 6

class HybridSim(Agent):

    def __init__(self,conf,sims,depth=30,tau=None,backend='pytorch'):

        super().__init__(sims=sims,backend=backend)

        self.depth = depth 

    def mcts(self,root_index):
        trace = select_index(root_index,self.arrs['child'],self.arrs['node_stats'])

        leaf_index = trace[-1]
        #leaf_index = select_index(root_index,self.arrs['child'],self.arrs['child_stats'])

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore() #- self.game_arr[root_index].getScore()


        if not leaf_game.end:

            _g = leaf_game.clone()

            _depth = 0
            
            _ma = 0
            _record = []
            
            while not _g.end and _depth < self.depth:
                v, p = self.evaluate_state(_g.getState())
                _record.append(v+_g.getScore())
                if _depth == 0:
                    _ma = (v+_g.getScore())
                else:
                    _ma = _ma * 0.9 + (v+_g.getScore()) * 0.1
                _act = choose_action(p)
                _g.play(_act)
                _depth += 1

            
            if _g.end:
                v = 0
                _ma *= 0.9
                _record.append(0)
                print(_record)            
            else:           
                v, p = self.evaluate_state(leaf_game.getState())
                _ma = _ma * 0.9 + (v+_g.getScore()) * 0.1
                _record.append(v+_g.getScore())
            print("%.3f %.3f %.3f %.3f %.3f %.3f"%(np.mean(_record),np.std(_record),np.amax(_record),np.amin(_record),_ma,_ma-np.mean(_record)))
            value = _g.getScore() + v

            for i in range(n_actions):
                _g = leaf_game.clone()
                _g.play(i)
                _n = self.new_node(_g)

                self.arrs['child'][leaf_index][i] = _n
                self.arrs['node_stats'][_n][2] = _g.getScore()
            
        
        backup_trace(trace,self.arrs['node_stats'],value)


