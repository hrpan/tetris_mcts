import agents.helper
from agents.agent import TreeAgent
from agents.core import select_trace, backup_trace
from random import randint
from agents.cppmodule.core import backup_trace_obs, select_trace_obs


class Vanilla(TreeAgent):
    def __init__(self, gamma=0.99, **kwargs):

        super().__init__(projection=True, **kwargs)

        self.gamma = gamma

        self.g_tmp = self.env(*self.env_args)

    def mcts(self, root_index, sims):

        child = self.arrays['child']
        score = self.arrays['score']

        if self.projection:
            visit = self.obs_arrays['visit']
            value = self.obs_arrays['value']
            variance = self.obs_arrays['variance']
            n_to_o = self.node_to_obs
            s_args = [root_index, child, visit, value, variance, score, n_to_o, 5]
            selection = select_trace_obs
            b_args = [None, visit, value, variance, n_to_o, score, None, None, self.gamma]
            #backup = backup_trace_mixture_obs
            backup = backup_trace_obs
        else:
            visit = self.arrays['visit']
            value = self.arrays['value']
            variance = self.arrays['variance']
            s_args = [root_index, child, visit, value, variance, score, 5]
            selection = select_trace
            b_args = [None, visit, value, variance, score, None, None, self.gamma]
            backup = backup_trace

        for i in range(sims):
            trace = selection(*s_args)
            leaf_index = trace[-1]

            leaf_game = self.game_arr[leaf_index]

            if not leaf_game.end:

                _g = self.g_tmp
                _g.copy_from(leaf_game)
                while not _g.end:
                    _act = randint(0, self.n_actions-1)
                    _g.play(_act)
                _value = _g.score
                _variance = 1e3

                self.expand(leaf_game)
            else:
                _value, _variance = leaf_game.score, 0

            b_args[0] = trace
            b_args[-3] = _value
            b_args[-2] = _variance
            backup(*b_args)
