from os.path import dirname
import os
import sys
sys.path.append('../../pyTetris')
from pyTetris import Tetris
import numpy as np
import argparse
import random
import pandas as pd

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

    def reset(self):
        self.scores = []

"""
ARGUMENTS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--agent_type', default='Vanilla', type=str, help='Which agent to use')
parser.add_argument('--app', default=1, type=int, help='Actions-per-drop')
parser.add_argument('--cycle', default=-1, type=int, help='Number of cycle')
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

game = Tetris(boardsize=(22,10),actions_per_drop=app)

if selfplay:
    _agent_module = __import__('agents.'+agent_type, globals(), locals(), [agent_type], 0)
    Agent = getattr(_agent_module,agent_type)
    agent = Agent(mcts_const,mcts_sims,tau=mcts_tau)
    agent.set_root(game)

if save:
    df = pd.DataFrame(columns=['board','policy','action','score','cum_score','child_stats'])
    df_ep = pd.DataFrame(columns=['board','policy','action','score','child_stats'])

tracker = ScoreTracker()
n_games = ngames
"""
MAIN GAME LOOP
"""
while True:
    if interactive:
        game.printState()
        print('Current score:%d'%game.getScore())
        action = int(input('Play:'))

    if printboard:
        game.printState()

    if selfplay:
        action = agent.play()
        if save:
            _dict = {'board':game.getState(),
                    'policy':agent.get_prob(),
                    'action':action,
                    'score':game.getScore(),
                    'child_stats':agent.get_stats()}
            df_ep = df_ep.append(_dict,ignore_index=True)

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
            tracker.append(game.getScore())
            n_games -= 1
            if save:
                score = game.getScore()
                df_ep['cum_score'] = score - df_ep['score']
                df = pd.concat([df,df_ep],ignore_index=True)
                df_ep = df_ep[0:0]
            sys.stdout.write('\rGames remaining:%d min/max/mean/std:%.3f/%.3f/%.3f/%.3f'%(n_games, *tracker.getStats()))
            sys.stdout.flush()
            if n_games == 0:
                break
            else:
                game.reset()
                agent.set_root(game)

sys.stdout.write('\n')
sys.stdout.flush()


if save:
    df['cycle']=cycle
    _save = save_dir + save_file + str(cycle)
    if os.path.isfile(_save):
        df_old = pd.read_pickle(_save)
        df = pd.concat([df_old,df],ignore_index=True)
    df.to_pickle(_save)
