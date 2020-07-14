from pyTetris import Tetris
import numpy as np
import argparse
from util.gui import GUI
from util.Data import DataSaver
from importlib import import_module


class ScoreTracker:
    def __init__(self):
        self.scores = []
        self.lines = []

    def append(self, score, line):
        self.scores.append(score)
        self.lines.append(line)

    def getStats(self):
        _min = np.amin(self.scores)
        _max = np.amax(self.scores)
        _mean = np.mean(self.scores)
        _std = np.std(self.scores)
        return _min, _max, _mean, _std

    def printStats(self):
        print('\rGames played:{:>3}    min/max/mean/std:{:5.2f}({:5.2f})/{:5.2f}'
              '({:5.2f})/{:5.2f}({:5.2f})/{:5.2f}({:5.2f})'.format(
                     len(self.scores),
                     np.amin(self.scores),
                     np.amin(self.lines),
                     np.amax(self.scores),
                     np.amax(self.lines),
                     np.mean(self.scores),
                     np.mean(self.lines),
                     np.std(self.scores),
                     np.std(self.lines)),
              end='', flush=True)

    def reset(self):
        self.scores = []


"""
ARGUMENTS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--agent_type', default=None, type=str, help='Which agent to use')
parser.add_argument('--app', default=1, type=int, help='Actions-per-drop')
parser.add_argument('--benchmark', default=False, help='Benchmark mode for agent', action='store_true')
parser.add_argument('--cycle', default=0, type=int, help='Number of cycle')
parser.add_argument('--endless', default=False, help='Endless plays', action='store_true')
parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor')
parser.add_argument('--gui', default=False, help='A simple GUI', action='store_true')
parser.add_argument('--interactive', default=False, help='Text interactive interface', action='store_true')
parser.add_argument('--mcts_const', default=5.0, type=float, help='PUCT constant')
parser.add_argument('--mcts_sims', default=50, type=int, help='Number of MCTS sims')
parser.add_argument('--mcts_tau', default=1.0, type=float, help='Temperature constant')
parser.add_argument('--min_visit', default=40, type=int, help='Minimum visits for node storage')
parser.add_argument('--ngames', default=50, type=int, help='Number of episodes to play')
parser.add_argument('--online', default=False, help='Online agent training', action='store_true')
parser.add_argument('--printboard', default=False, help='Print board', action='store_true')
parser.add_argument('--print_board_to_file', default=False, help='Print board to file', action='store_true')
parser.add_argument('--realtime_status', default=False, help='Save realtime game status through numpy memmap', action='store_true')
parser.add_argument('--save', default=False, help='Save self-play episodes', action='store_true')
parser.add_argument('--save_dir', default='./data/', type=str, help='Directory for save')
parser.add_argument('--save_file', default='data', type=str, help='Filename to save')
parser.add_argument('--save_tree', default=False, help='Save expanded tree nodes', action='store_true')
parser.add_argument('--tetris_randomizer', default=0, type=int, help='Queue randomizer used by Tetris (0: bag, 1: uniform)')
parser.add_argument('--tetris_scoring', default=0, type=int, help='Scoring system used by Tetris (0: official guideline, 1: line clears)')
args = parser.parse_args()

"""
SOME INITS
"""
env_args = ((20, 10), args.app, args.tetris_scoring, args.tetris_randomizer)
game = Tetris(*env_args)

ngames = 0

if args.agent_type:
    _agent_module = import_module('agents.' + args.agent_type)
    Agent = getattr(_agent_module, args.agent_type)
    agent_args = dict(
            sims=args.mcts_sims,
            env=Tetris,
            env_args=env_args,
            benchmark=args.benchmark,
            online=args.online,
            min_visit=args.min_visit)
    agent = Agent(**agent_args)
    agent.update_root(game)
else:
    agent = None

if args.save:
    saver = DataSaver(args.save_dir, args.save_file, args.cycle)

if args.save_tree:
    agent.saver = DataSaver(args.save_dir, 'tree', args.cycle)

tracker = ScoreTracker()

if args.gui:
    G = GUI()

if args.print_board_to_file:
    board_output = open('board_output', 'wb')

if args.realtime_status:
    _board = np.memmap('./tmp/board', dtype=np.int8, mode='w+', shape=(20, 10))
    _combo = np.memmap('./tmp/combo', dtype=np.int32, mode='w+', shape=(1, ))
    _score = np.memmap('./tmp/score', dtype=np.int32, mode='w+', shape=(1, ))
    _lines = np.memmap('./tmp/lines', dtype=np.int32, mode='w+', shape=(1, ))
    _line_stats = np.memmap('./tmp/line_stats', dtype=np.int32, mode='w+', shape=(4, ))
"""
MAIN GAME LOOP
"""
while True:
    if args.interactive:
        game.printState()
        print('Current score: {}'.format(game.score))
        action = int(input('Play:'))

    elif agent:

        if args.printboard:
            game.printState()

        action = agent.play()

        if args.save:
            saver.add(ngames, action, agent, game)

    if args.gui:
        G.update_canvas(game.getState())

    if args.print_board_to_file:
        board_output.truncate(0)
        board_output.seek(0)
        board_output.write(game.getState().tostring())
        board_output.flush()

    if args.realtime_status:
        _board[:] = game.getState()[:]
        _combo[:] = game.combo
        _lines[:] = game.line_clears
        _score[:] = game.score
        _line_stats[:] = game.line_stats[:] 

    game.play(action)

    if agent:
        agent.update_root(game)

    if game.end:
        if args.interactive:
            play_more = input('Play more?')
            if play_more == 'y':
                game.reset()
            else:
                break
        elif args.endless:
            ngames += 1
            print('Episode: {:>5} Score: {:>10} Lines Cleared: {:>10}'.format(ngames, game.score, game.line_clears), flush=True)
            game.reset()
            agent.update_root(game)
        else:
            ngames += 1

            tracker.append(game.score, game.line_clears)
            tracker.printStats()

            if ngames >= args.ngames:
                break
            else:
                game.reset()
                agent.update_root(game)

print(flush=True)

agent.close()

if args.save:
    saver.close()
