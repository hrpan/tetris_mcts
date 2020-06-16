import torch
import re
import sys
import os
import shutil
import filecmp
import numpy as np
from collections import defaultdict
from shutil import copyfile

sys.path.append('../')


class BoardParser:
    def __init__(self):

        self.file = open('../board_output', 'rb')

        self.data = None

    def update(self):

        s = self.file.read()

        if len(s) == 200:
            self.data = np.fromstring(s, dtype=np.int8).reshape(20, 10)

        self.file.seek(0)


class StatusParser:
    def __init__(self):

        self.board = np.memmap('../tmp/board', mode='r', dtype=np.int8, shape=(20, 10))
        self.combo = np.memmap('../tmp/combo', mode='r', dtype=np.int32, shape=(1, ))
        self.lines = np.memmap('../tmp/lines', mode='r', dtype=np.int32, shape=(1, ))
        self.score = np.memmap('../tmp/score', mode='r', dtype=np.int32, shape=(1, ))
        self.line_stats = np.memmap('../tmp/line_stats', mode='r', dtype=np.int32, shape=(4, ))


class Parser:
    def __init__(self, filename):

        self.filename = filename

        self.last_update = -1

    def check_update(self):

        latest_update = os.path.getmtime(self.filename)

        if latest_update > self.last_update:
            self.last_update = latest_update
            self.parse()
            return True
        return False

    def parse(self):
        score_re = 'Episode:\s*(?P<episode>\d*)\s*' \
                   'Score:\s*(?P<score>\d*)\s*' \
                   'Lines Cleared:\s*(?P<lines>\d*)'
        train_re = 'Iteration:\s*(?P<iter>\d*)\s*' \
                   'training loss:\s*(?P<t_loss>\d*\.\d*)\s*' \
                   'validation loss:\s*(?P<v_loss>\d*\.\d*)±\s*(?P<v_loss_err>\d*\.\d*|nan)\s*' \
                   'gradient norm:\s*(?P<g_norm>\d*\.\d*)'
        datasize_re = 'Training data size:\s*(?P<tsize>\d*)\s*' \
                      'Validation data size:\s*(?P<vsize>\d*)'
        queue_re = 'Memory usage: (?P<filled>\d*) / (?P<size>\d*).*'

        self.data = defaultdict(list)
        size = 0
        filled = 0
        rm_since_last_game = 0

        with open(self.filename) as f:
            lc_avg_tmp = []
            sc_avg_tmp = []
            data_accum = 0
            training = False
            for line in f.readlines():
                match_score_re = re.search(score_re, line)
                match_train_re = re.search(train_re, line)
                match_datasize_re = re.search(datasize_re, line)
                match_queue_re = re.search(queue_re, line)
                if match_score_re:
                    d = match_score_re.groupdict()
                    lc = int(d['lines'])
                    sc = int(d['score'])
                    self.data['line_cleared'].append(lc)
                    self.data['score'].append(sc)
                    self.data['data_accumulated'].append(data_accum)
                    lc_avg_tmp.append(lc)
                    sc_avg_tmp.append(sc)
                    rm_since_last_game = 0
                elif match_train_re:
                    d = match_train_re.groupdict()
                    self.data['training_loss'].append(float(d['t_loss']))
                    self.data['validation_loss'].append(float(d['v_loss']))
                    if d['v_loss_err'] == 'nan':
                        self.data['validation_loss_err'].append(0)
                    else:
                        self.data['validation_loss_err'].append(float(d['v_loss_err']))
                    self.data['g_norm'].append(float(d['g_norm']))
                    #print(d['g_norm'])
                elif match_datasize_re:
                    d = match_datasize_re.groupdict()
                    tsize = int(d['tsize'])
                    vsize = int(d['vsize'])
                    data_accum += (tsize + vsize)
                elif match_queue_re:
                    d = match_queue_re.groupdict()
                    filled = int(d['filled'])
                    size = int(d['size'])
                elif 'REMOVING UNUSED' in line:
                    rm_since_last_game += 1
                elif 'proceed to training' in line:
                    training = True
                    if lc_avg_tmp:
                        mean = np.average(lc_avg_tmp)
                        std = np.std(lc_avg_tmp) / np.sqrt(len(lc_avg_tmp))
                        self.data['line_cleared_per_train'].append((mean, std))
                        lc_avg_tmp.clear()
                    else:
                        if self.data['line_cleared_per_train']:
                            self.data['line_cleared_per_train'].append(
                                    self.data['line_cleared_per_train'][-1])
                        else:
                            self.data['line_cleared_per_train'].append((0, 0))
                    if sc_avg_tmp:
                        mean = np.average(sc_avg_tmp)
                        std = np.std(sc_avg_tmp) / np.sqrt(len(sc_avg_tmp))
                        self.data['score_per_train'].append((mean, std))
                        sc_avg_tmp.clear()
                    else:
                        if self.data['score_per_train']:
                            self.data['score_per_train'].append(
                                    self.data['score_per_train'][-1])
                        else:
                            self.data['score_per_train'].append((0, 0))
                elif 'Training complete' in line:
                    training = False
            if lc_avg_tmp:
                mean = np.average(lc_avg_tmp)
                std = np.std(lc_avg_tmp) / np.sqrt(len(lc_avg_tmp))
                self.data['line_cleared_per_train'].append((mean, std))
            if sc_avg_tmp:
                mean = np.average(sc_avg_tmp)
                std = np.std(sc_avg_tmp) / np.sqrt(len(sc_avg_tmp))
                self.data['score_per_train'].append((mean, std))

            if not training:
                flocal = './model_checkpoint'
                ftarget = '../pytorch_model/model_checkpoint'

                ex_local = os.path.isfile(flocal)
                ex_target = os.path.isfile(ftarget)

                if ex_target and ((ex_local and not filecmp.cmp(flocal, ftarget)) or not ex_local):
                    copyfile(ftarget, flocal)

        self.data['filled'] = filled
        self.data['size'] = size
        self.data['rm_since_last_game'] = rm_since_last_game


class ModelParser:
    def __init__(self, distributional=True):

        self.last_update = -1

        self.data = {}

        self.distributional = distributional

    def check_update(self):
        flocal = './model_checkpoint'
        if os.path.isfile(flocal):
            latest = os.path.getmtime(flocal)
            if latest > self.last_update:
                print('New model found, updating...', flush=True)
                self.last_update = latest
                state = torch.load(flocal, map_location=torch.device('cpu'))
                model_state = state['model_state_dict']
                self.parse_state(model_state)
                return True
        return False

    def parse(self, model):
        self.parse_state(model.state_dict())

    def parse_state(self, model_state):
        self.data = {}
        for k, v in model_state.items():
            if 'weight' in k:
                k = k.replace('.weight', '')
                k = k.replace('seq.', '')
                self.data[k] = v.cpu().numpy().ravel()
