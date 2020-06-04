import agents.helper
from agents.cppmodule.agent import MCTSAgent
from random import randint

def random_playout(game):
    while not game.end:
        game.play(randint(0, 7))
    return game.score, 1e5

class VanillaC(MCTSAgent):

    def __init__(self, sims=100, max_nodes=500000, projection=True, gamma=0.99, benchmark=False, LP=False, **kwargs):
        super().__init__(sims, max_nodes, projection, gamma, benchmark, random_playout, 1, False)
    
    def close(self):
        pass
