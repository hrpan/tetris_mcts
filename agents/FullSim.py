
from agents.agent import Agent
from agents.core import select_index, backup_trace, choose_action

n_actions = 6

class FullSim(Agent):

    def __init__(self,conf,sims,tau=None,backend='pytorch'):

        super().__init__(sims=sims,backend=backend)

    def mcts(self,root_index):
        trace = select_index(root_index,self.arrs['child'],self.arrs['node_stats'])

        leaf_index = trace[-1]
        #leaf_index = select_index(root_index,self.arrs['child'],self.arrs['child_stats'])

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore() #- self.game_arr[root_index].getScore()


        if not leaf_game.end:

            v, p = self.evaluate_state(leaf_game.getState())

            _g = leaf_game.clone()

            while not _g.end:
                v, p = self.evaluate_state(_g.getState())
                _act = choose_action(p)
                _g.play(_act)

            value = _g.getScore()

            for i in range(n_actions):
                _g = leaf_game.clone()
                _g.play(i)
                _n = self.new_node(_g)

                self.arrs['child'][leaf_index][i] = _n
                self.arrs['node_stats'][_n][2] = _g.getScore()
            
        
        backup_trace(trace,self.arrs['node_stats'],value)


