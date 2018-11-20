from os.path import dirname
import os
import sys
sys.path.append('../../pyTetris')
#from pyTetris import Tetris
from nbTetris import Tetris
import numpy as np
import argparse
from util.Data import DataSaver
from importlib import import_module

class ScoreTracker:
    def __init__(self):
        self.scores = []

    def append(self, score):
        self.scores.append(score)

    def getStats(self):
        _min = np.amin(self.scores)
        _max = np.amax(self.scores)
        _mean = np.mean(self.scores)
        _std = np.std(self.scores)
        return _min, _max, _mean, _std

    def printStats(self):
        sys.stdout.write('\rGames played:%3d    min/max/mean/std:%5.2f/%5.2f/%5.2f/%5.2f'
            %(len(self.scores),
            np.amin(self.scores),
            np.amax(self.scores),
            np.mean(self.scores),
            np.std(self.scores)))
        sys.stdout.flush()

    def reset(self):
        self.scores = []

"""
ARGUMENTS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--agent_type', default='Vanilla', type=str, help='Which agent to use')
parser.add_argument('--app', default=1, type=int, help='Actions-per-drop')
parser.add_argument('--cycle', default=0, type=int, help='Number of cycle')
parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor')
parser.add_argument('--interactive', default=False, help='Text interactive interface', action='store_true')
parser.add_argument('--mcts_const', default=5.0, type=float, help='PUCT constant')
parser.add_argument('--mcts_sims', default=500, type=int, help='Number of MCTS sims')
parser.add_argument('--mcts_tau', default=1.0, type=float, help='Temperature constant')
parser.add_argument('--ngames', default=50, type=int, help='Number of episodes to play')
parser.add_argument('--printboard', default=False, help='Print board', action='store_true')
parser.add_argument('--save', default=False, help='Save self-play episodes', action='store_true')
parser.add_argument('--save_dir', default='./data/',type=str, help='Directory for save')
parser.add_argument('--save_file', default='data', type=str, help='Filename to save')
parser.add_argument('--selfplay', default=False, help='Agent selfplay', action='store_true')
args = parser.parse_args()

agent_type = args.agent_type
app = args.app
cycle = args.cycle
gamma = args.gamma
interactive = args.interactive
mcts_sims = args.mcts_sims
mcts_const = args.mcts_const
mcts_tau = args.mcts_tau
ngames = args.ngames
printboard = args.printboard
save = args.save
save_dir = args.save_dir
save_file = args.save_file
selfplay = args.selfplay

"""
SOME INITS
"""
env_args = ((22,10), app)
game = Tetris(*env_args)

if selfplay:
    _agent_module = import_module('agents.'+agent_type)
    Agent = getattr(_agent_module,agent_type)
    agent = Agent(mcts_const, mcts_sims, tau=mcts_tau, env=Tetris, env_args=env_args)
    agent.set_root(game)

if save:
    saver = DataSaver(save_dir, save_file, cycle)

tracker = ScoreTracker()

"""
MAIN GAME LOOP
"""
while True:
    if interactive:
        game.printState()
        print('Current score:%d'%game.getScore())
        action = int(input('Play:'))

    elif selfplay:

        if printboard:
            game.printState()

        action = agent.play()

        if save:
            saver.add(ngames,action,agent,game)    

    game.play(action)
    
    if selfplay:
        agent.update_root(game)

    if game.end:
        if interactive:
            play_more = input('Play more?')
            if play_more == 'y':
                game.reset()
            else:
                break
        else:
            ngames -= 1

            if save:
                saver.save_episode()
            tracker.append(game.getScore())
            tracker.printStats()

            if ngames == 0:
                break
            else:
                game.reset()
                agent.set_root(game)

sys.stdout.write('\n')
sys.stdout.flush()


if save:
   saver.close() 
