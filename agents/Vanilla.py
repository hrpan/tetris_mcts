from agents.agent import Agent
from agents.core import select_index, backup_trace, choose_action

from random import randint

n_actions = 6

class Vanilla(Agent):
    def __init__(self, conf, sims, tau=None, env=None, env_args=None):

        super().__init__(sims=sims, backend=None, env=env, env_args=env_args)

        self.g_tmp = env(*env_args)

    def mcts(self,root_index):
        trace = select_index(root_index,self.arrs['child'],self.arrs['node_stats'])

        leaf_index = trace[-1]
        #leaf_index = select_index(root_index,self.arrs['child'],self.arrs['child_stats'])

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore() #- self.game_arr[root_index].getScore()

        if not leaf_game.end:

            #v, p = self.evaluate_state(leaf_game.getState())

            _g = self.g_tmp
            _g.copy_from(leaf_game)

            while not _g.end:
                _act = randint(0,n_actions-1)
                _g.play(_act)

            value = _g.getScore()

            for i in range(n_actions):
                _g.copy_from(leaf_game)
                _g.play(i)
                _n = self.new_node(_g)

                self.arrs['child'][leaf_index][i] = _n
                self.arrs['node_stats'][_n][2] = _g.getScore()
            
        
        backup_trace(trace,self.arrs['node_stats'],value)


