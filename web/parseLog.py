import re, sys, os, shutil, time, glob, filecmp
import numpy as np
sys.path.append('../')
import model.model_pytorch as M
from datetime import datetime as dt
from importlib import reload

class BoardParser:
    def __init__(self):
        
        self.file = open('../board_output', 'rb')

        self.data = None

    def update(self):

        s = self.file.read()
        
        if len(s) == 220:
            self.data = np.fromstring(s, dtype=np.int8).reshape(22, 10)

        self.file.seek(0)

class Parser:
    def __init__(self, filename):

        self.filename = filename
    
        self.last_log_update = -1

    def check_update(self):

        latest_log_update = os.path.getmtime(self.filename)

        if latest_log_update > self.last_log_update:
            self.last_log_update = latest_log_update
            self.parse()
            return True
        return False
        
    def parse(self): 
        score_re = 'Episode:\s*(?P<episode>\d*)\s*' \
                'Score:\s*(?P<score>\d*)\s*' \
                'Lines Cleared:\s*(?P<lines>\d*)'   
        train_re = 'Iteration:\s*(?P<iter>\d*)\s*' \
                'training loss:\s*(?P<t_loss>\d*.\d*)\s*' \
                'validation loss:\s*(?P<v_loss>\d*.\d*)\s*'
        datasize_re = 'Training data size:\s*(?P<tsize>\d*)\s*' \
                'Validation data size:\s*(?P<vsize>\d*)'

        line_cleared = []
        line_cleared_per_train = []
        score = []
        score_per_train = []
        data_accumulated = []
        training_loss = []
        validation_loss = []

        with open(self.filename) as f: 
            lc_avg_tmp = []
            sc_avg_tmp = []
            data_accum = 0
            for line in f.readlines():
                match_score_re = re.search(score_re, line)
                match_train_re = re.search(train_re, line)
                match_datasize_re = re.search(datasize_re, line)
                if match_score_re:
                    d = match_score_re.groupdict()
                    lc = int(d['lines'])
                    sc = int(d['score'])
                    line_cleared.append(lc)
                    score.append(sc)
                    lc_avg_tmp.append(lc)
                    sc_avg_tmp.append(sc)
                    data_accumulated.append(data_accum)
                elif match_train_re:
                    d = match_train_re.groupdict()
                    tl = float(d['t_loss'])
                    vl = float(d['v_loss'])
                    training_loss.append(tl)
                    validation_loss.append(vl)
                elif match_datasize_re:
                    d = match_datasize_re.groupdict()
                    tsize = int(d['tsize'])
                    vsize = int(d['vsize'])
                    data_accum += (tsize + vsize)
                if 'proceed to training' in line:
                    if lc_avg_tmp:
                        line_cleared_per_train.append((np.average(lc_avg_tmp), np.std(lc_avg_tmp)/np.sqrt(len(lc_avg_tmp))))
                        lc_avg_tmp.clear()
                    else:
                        line_cleared_per_train.append(line_cleared_per_train[-1]) 
                    if sc_avg_tmp:
                        score_per_train.append((np.average(sc_avg_tmp), np.std(sc_avg_tmp)/np.sqrt(len(sc_avg_tmp))))
                        sc_avg_tmp.clear()
                    else:
                        score_per_train.append(score_per_train[-1]) 
            if lc_avg_tmp:
                line_cleared_per_train.append((np.average(lc_avg_tmp), np.std(lc_avg_tmp)/np.sqrt(len(lc_avg_tmp))))
            if sc_avg_tmp:
                score_per_train.append((np.average(sc_avg_tmp), np.std(sc_avg_tmp)/np.sqrt(len(sc_avg_tmp))))

            if 'line' in locals() and not 'loss' in line:
                flocal = './model_checkpoint'
                ftarget = '../pytorch_model/model_checkpoint'
                
                ex_local = os.path.isfile(flocal)
                ex_target = os.path.isfile(ftarget)

                if ex_target and ((ex_local and not filecmp.cmp(flocal, ftarget)) or not ex_local):
                    shutil.copyfile(ftarget, flocal)

        self.data = dict(
                line_cleared=line_cleared, 
                line_cleared_per_train=line_cleared_per_train,
                score=score,
                score_per_train=score_per_train,
                data_accumulated=data_accumulated,
                training_loss=training_loss,
                validation_loss=validation_loss)

class ModelParser:
    def __init__(self):

        self.last_update = -1
        
        self.data = {}
        
    def check_update(self):

        if os.path.isfile('./model_checkpoint'):
            latest = os.path.getmtime('./model_checkpoint')
            if latest > self.last_update:
                self.last_update = latest
                reload(M)
                m = M.Model(use_cuda=False)
                m.load(filename='./model_checkpoint')
                self.parse(m)
                return True
        elif not self.data:
            m = M.Model(use_cuda=False)
            self.parse(m)
            return True
        return False

    def parse(self, model):
        self.data = {}
        for module in model.model.named_modules():
            if not module[0] or not hasattr(module[1], 'weight') or module[1].weight is None:
                continue

            self.data[module[0]] = module[1].weight.data.numpy().ravel()
        
        
