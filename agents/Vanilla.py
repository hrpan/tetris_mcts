from agents.agent import TreeAgent
from agents.core import *
from agents.core_projection import *

from random import randint


class Vanilla(TreeAgent):
    def __init__(self, **kwargs):

        super().__init__(projection=False, **kwargs)

        self.g_tmp = self.env(*self.env_args)

    def mcts(self, root_index):

        if self.projection:
            trace = select_trace_obs(
                root_index,
                self.arrays['child'],
                self.obs_arrays['visit'],
                self.obs_arrays['value'],
                self.obs_arrays['variance'],
                self.arrays['score'],
                self.node_to_obs,
                low=5)
        else:
            trace = select_trace(
                root_index,
                self.arrays['child'],
                self.arrays['visit'],
                self.arrays['value'],
                self.arrays['variance'],
                self.arrays['score'],
                low=5)

        leaf_index = trace[-1]

        leaf_game = self.game_arr[leaf_index]

        if not leaf_game.end:

            _g = self.g_tmp
            _g.copy_from(leaf_game)

            while not _g.end:
                _act = randint(0, self.n_actions-1)
                _g.play(_act)

            value = _g.score
            var = 1e3
            
            self.expand(leaf_game)

        else:
            value = leaf_game.score
            var = 0

        if self.projection:
            backup_trace_obs(
                trace,
                self.obs_arrays['visit'],
                self.obs_arrays['value'],
                self.obs_arrays['variance'],
                self.node_to_obs,
                self.arrays['score'],
                value, var)
        else:
            backup_trace_welford_v2(
                trace,
                self.arrays['visit'],
                self.arrays['value'],
                self.arrays['variance'],
                self.arrays['score'],
                value, var)
