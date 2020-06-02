import numpy as np
from agents.ValueSim import ValueSim
from agents.cppmodule.core import get_all_childs, select_trace_obs, backup_trace_obs_LP
from agents.cppmodule.core import get_unique_child_obs


class ValueSimLP(ValueSim):

    def __init__(self, **kwargs):

        super().__init__(min_visits_to_store=25, **kwargs)

    def mcts(self, root_index, sims):

        child = self.arrays['child']
        score = self.arrays['score']
        games = self.game_arr
        eval = self.model.inference

        if self.projection:
            state = self.obs_arrays['state']
            visit = self.obs_arrays['visit']
            value = self.obs_arrays['value']
            variance = self.obs_arrays['variance']
            end = self.arrays['end']
            n_to_o = self.node_to_obs
            s_args = [root_index, child, visit, value, variance, score, n_to_o, 1]
            selection = select_trace_obs
            mixture = False
            averaged = True
            b_args = [
                None, visit, value, variance, n_to_o, score,
                end, None, None, None, None, self.gamma, mixture, averaged]
            backup = backup_trace_obs_LP
        else:
            visit = self.arrays['visit']
            value = self.arrays['value']
            variance = self.arrays['variance']
            s_args = [root_index, child, visit, value, variance, score]
            selection = select_trace
            b_args = [None, visit, value, variance, score, None, None, self.gamma]
            backup = backup_trace

        for i in range(sims):
            trace = selection(*s_args)

            leaf_index = trace[-1]

            leaf_game = games[leaf_index]

            if not leaf_game.end:

                self.expand(leaf_game)

                _c, _o = get_unique_child_obs(leaf_index, child, score, n_to_o)

                states = state[_o]

                v, var = eval(states[:, None, :, :])
                v, var = v.ravel(), var.ravel()
            else:
                _c = _o = []
                v = var = np.empty(0, dtype=np.float32)

            b_args[0] = trace
            b_args[-7] = _c
            b_args[-6] = _o
            b_args[-5] = v
            b_args[-4] = var
            backup(*b_args)

            #print(i, visit[n_to_o[self.root]], value[n_to_o[self.root]], variance[n_to_o[self.root]])
            #_ro = n_to_o[child[self.root]]
            #print(visit[_ro].astype(int), score[child[self.root]], value[_ro].astype(int), variance[_ro].astype(int))
            #input()
