from agents.agent import Agent
from agents.core import select_trace, backup_trace, policy_max

from random import randint


class Vanilla(Agent):
    def __init__(self, **kwargs):

        kwargs['backend'] = None

        super().__init__(**kwargs)

        self.g_tmp = self.env(*self.env_args)

    def mcts(self, root_index):

        child = self.arrs['child']
        node_stats = self.arrs['node_stats']

        trace = select_trace(root_index, child, node_stats, policy=policy_max)

        leaf_index = trace[-1]

        leaf_game = self.game_arr[leaf_index]

        if not leaf_game.end:

            _g = self.g_tmp
            _g.copy_from(leaf_game)

            while not _g.end:
                _act = randint(0, self.n_actions-1)
                _g.play(_act)

            value = _g.getScore()

            for i in range(self.n_actions):
                _g.copy_from(leaf_game)
                _g.play(i)
                _n = self.new_node(_g)

                child[leaf_index][i] = _n
                node_stats[_n][2] = _g.getScore()
        else:
            value = leaf_game.getScore()

        backup_trace(trace, node_stats, value)
