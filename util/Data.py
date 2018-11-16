import tables
import numpy as np

class State(tables.IsDescription):
    episode     = tables.Int32Col()
    board       = tables.Int8Col(shape=(22, 10))
    policy      = tables.Float32Col(shape=(6,))
    action      = tables.Int8Col()
    score       = tables.Int32Col()
    child_stats = tables.Float32Col(shape=(6, 6))
    cycle       = tables.Int32Col() 


class DataSaver:

    def __init__(self, save_dir, save_file, cycle):

        file_name = save_dir + save_file + str(cycle)

        self.cycle = cycle

        self.file = tables.open_file(file_name, mode='a')
        
        if self.file.__contains__('/State'):
            self.table = self.file.root.State
        else:
            self.table = self.file.create_table(self.file.root, 'State', State)

        self.state = self.table.row

    def add(self, episode, action, agent, game):
        
        self.state['episode'] = episode
        self.state['board'] = game.getState()
        self.state['policy'] = agent.get_prob()
        self.state['action'] = action
        self.state['score'] = game.getScore()
        self.state['child_stats'] = agent.get_stats()         
        self.state['cycle'] = self.cycle

        self.state.append()

    def close(self):
    
        self.file.close()

    def save_episode(self):

        self.table.flush()


class DataLoader:
    
    def __init__(self, list_of_files):
    
        self.files = [ tables.open_file(f, mode='r') for f in list_of_files]

        self.tables = [f.root.State for f in self.files]
    
        self.episode = np.concatenate([t.col('episode') for t in self.tables])
        self.board = np.concatenate([t.col('board') for t in self.tables])
        self.policy = np.concatenate([t.col('policy') for t in self.tables])
        self.action = np.concatenate([t.col('action') for t in self.tables])
        self.score = np.concatenate([t.col('score') for t in self.tables])
        self.child_stats = np.concatenate([t.col('child_stats') for t in self.tables])
        self.cycle = np.concatenate([t.col('cycle') for t in self.tables])

        self.length = len(self.episode)

        for f in self.files:
            f.close()

        def bound_index(self, index):
            if index >= self.length:
                return self.length - 1
            elif index < 0:
                return 0
            else:
                return index

        def getBoard(self, index)
            index = self.bound_index(index)
            return self.board[index]

        def getPolicy(self, index)
            index = self.bound_index(index)
            return self.policy[index]

        def getCycle(self, index)
            index = self.bound_index(index)
            return self.cycle[index]

        def getScore(self, index)
            index = self.bound_index(index)
            return self.score[index]
