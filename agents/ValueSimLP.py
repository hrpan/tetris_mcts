import numpy as np
from agents.ValueSim import ValueSim
from agents.cppmodule.core import get_all_childs, select_trace_obs, backup_trace_obs
from agents.cppmodule.core import get_unique_child_obs


class ValueSimLP(ValueSim):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def evaluate_states(self, states):

        v, var = self.model.inference(states[:, None, :, :])

        return np.squeeze(v, axis=1), np.squeeze(var, axis=1)

    def mcts(self, root_index, sims):

        child = self.arrays['child']
        score = self.arrays['score']

        if self.projection:
            visit = self.obs_arrays['visit']
            value = self.obs_arrays['value']
            variance = self.obs_arrays['variance']
            n_to_o = self.node_to_obs
            s_args = [root_index, child, visit, value, variance, score, n_to_o, 1]
            selection = select_trace_obs
            b_args = [None, visit, value, variance, n_to_o, score, None, None, self.gamma]
            #backup = backup_trace_mixture_obs
            backup = backup_trace_obs
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

            leaf_game = self.game_arr[leaf_index]

            _value, _variance = leaf_game.score, 0
            if not leaf_game.end:

                self.expand(leaf_game)

                _c, _o = get_unique_child_obs(leaf_index, child, score, n_to_o)

                states = self.obs_arrays['state'][_o]

                v, var = self.evaluate_states(states)

                _variance = np.mean(var)

                vmean = 0
                for __c, __o, __v, __var in zip(_c, _o, v, var):
                    if visit[__o] > 0:
                        vmean += value[__o] + (score[__c] - _value)
                        continue
                    visit[__o] = 1
                    value[__o] = __v
                    variance[__o] = __var
                    vmean += __v + (score[__c] - _value)
                _value += vmean / len(v)

            b_args[0] = trace
            b_args[-3] = _value
            b_args[-2] = _variance
            backup(*b_args)
            #print(i, visit[n_to_o[self.root]], value[n_to_o[self.root]], variance[n_to_o[self.root]])
            #input()
