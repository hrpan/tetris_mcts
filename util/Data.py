import tables
import numpy as np

n_actions = 7


def keyFile(s):
    for i, c in enumerate(s):
        if c.isdigit() and s[i-4:i] == 'data':
            break
    return int(s[i:])


class State(tables.IsDescription):
    episode        = tables.Int32Col()
    board          = tables.Int8Col(shape=(20, 10))
    policy         = tables.Float32Col(shape=(n_actions,))
    action         = tables.Int8Col()
    combo          = tables.Int32Col()
    lines          = tables.Int32Col()
    line_stats     = tables.Int32Col(shape=(4,))
    score          = tables.Int32Col()
    child_stats    = tables.Float32Col(shape=(3, n_actions))
    cycle          = tables.Int32Col()
    value          = tables.Float32Col()
    variance       = tables.Float32Col()


class Loss(tables.IsDescription):
    loss_train               = tables.Float32Col()
    loss_train_value         = tables.Float32Col()
    loss_train_variance      = tables.Float32Col()
    loss_train_policy        = tables.Float32Col()
    loss_validation          = tables.Float32Col()
    loss_validation_value    = tables.Float32Col()
    loss_validation_variance = tables.Float32Col()
    loss_validation_policy   = tables.Float32Col()
    loss_ewc                 = tables.Float32Col()
    cycle                    = tables.Int32Col()


class DataSaver:

    def __init__(self, save_dir, save_file, cycle, chunksize=500):

        file_name = save_dir + save_file + str(cycle)

        self.chunksize = chunksize

        self.cycle = cycle

        self.file = tables.open_file(file_name, mode='a')

        self.iter = 0

        if self.file.__contains__('/State'):
            self.table = self.file.root.State
        else:
            self.table = self.file.create_table(self.file.root, 'State', State)

        self.state = self.table.row

    def add(self, episode, action, agent, game):

        def check(key):
            return hasattr(agent, key) and callable(getattr(agent, key))

        self.iter += 1
        self.state['episode'] = episode
        self.state['board'] = game.getState()
        if check('get_prob'):
            self.state['policy'] = agent.get_prob()
        self.state['action'] = action
        self.state['combo'] = game.combo
        self.state['lines'] = game.line_clears
        self.state['line_stats'] = game.line_stats
        self.state['score'] = game.score
        if check('get_stats'):
            self.state['child_stats'] = agent.get_stats()
        self.state['cycle'] = self.cycle
        if check('get_value_and_variance'):
            v, var = agent.get_value_and_variance()
        elif check('get_value'):
            v = agent.get_value()
            var = 0
        else:
            v, var = 0, 0
        self.state['value'] = v
        self.state['variance'] = var

        self.state.append()

        self.iter = self.iter % self.chunksize

        if self.iter == 0:
            self.table.flush()

    def add_raw(self, episode, board,
                policy, action, combo,
                lines, line_stats, score, child_stats,
                value, variance):

        self.iter += 1
        self.state['episode'] = episode
        self.state['board'] = board
        self.state['policy'] = policy
        self.state['action'] = action
        self.state['combo'] = combo
        self.state['lines'] = lines
        self.state['line_stats'] = line_stats
        self.state['score'] = score
        self.state['child_stats'] = child_stats
        self.state['cycle'] = self.cycle
        self.state['value'] = value
        self.state['variance'] = variance

        self.state.append()

        self.iter = self.iter % self.chunksize

        if self.iter == 0:
            self.table.flush()

    def close(self):

        self.table.flush()

        self.file.close()

    def save_episode(self):

        self.table.flush()


class DataLoader:

    def __init__(self, list_of_files):

        list_of_files = sorted(list_of_files, key=keyFile)

        self.files = [tables.open_file(f, mode='r') for f in list_of_files]

        self.tables = [f.root.State for f in self.files]

        for c in self.tables[0].colnames:
            setattr(self, c, np.concatenate([t.col(c) for t in self.tables]))

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

    def getBoard(self, index):
        index = self.bound_index(index)
        return self.board[index]

    def getPolicy(self, index):
        index = self.bound_index(index)
        return self.policy[index]

    def getCycle(self, index):
        index = self.bound_index(index)
        return self.cycle[index]

    def getScore(self, index):
        index = self.bound_index(index)
        return self.score[index]

    def getLines(self, index):
        index = self.bound_index(index)
        return self.lines[index]

    def getCombo(self, index):
        index = self.bound_index(index)
        return self.combo[index]


class LossSaver:

    def __init__(self, cycle):

        self.cycle = cycle

        self.file = tables.open_file('data/loss', mode='a')

        if self.file.__contains__('/Loss'):
            self.table = self.file.root.Loss
        else:
            self.table = self.file.create_table(self.file.root, 'Loss', Loss)

        self.loss = self.table.row

        self.iter = 0
        self.chunksize = 512

    def add(self, losses):
        for l in losses:
            self.loss['loss_train'] = l[0]
            self.loss['loss_train_value'] = l[1]
            self.loss['loss_train_variance'] = l[2]
            self.loss['loss_train_policy'] = l[3]
            self.loss['loss_validation'] = l[4]
            self.loss['loss_validation_value'] = l[5]
            self.loss['loss_validation_variance'] = l[6]
            self.loss['loss_validation_policy'] = l[7]
            self.loss['loss_ewc'] = l[8]
            self.loss['cycle'] = self.cycle

            self.loss.append()

        self.table.flush()

    def add_raw(self, l):

        self.iter += 1

        self.loss['loss_train'] = l[0]
        self.loss['loss_train_value'] = l[1]
        self.loss['loss_train_variance'] = l[2]
        self.loss['loss_train_policy'] = l[3]
        self.loss['loss_validation'] = l[4]
        self.loss['loss_validation_value'] = l[5]
        self.loss['loss_validation_variance'] = l[6]
        self.loss['loss_validation_policy'] = l[7]
        self.loss['loss_ewc'] = l[8]
        self.loss['cycle'] = self.cycle

        self.loss.append()

        self.iter = self.iter % self.chunksize
        if self.iter == 0:
            self.table.flush()

    def close(self):
        self.table.flush()
        self.file.close()
