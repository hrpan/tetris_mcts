import numpy as np
import os
import sys
import glob
import argparse
import random
from collections import defaultdict
from util.Data import DataLoader, LossSaver
from math import ceil
eps = 0.001

"""
ARGS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default:32)')
parser.add_argument('--cycle', default=-1, type=int, help='Cycle (default:-1)')
parser.add_argument('--data_paths', default=[], nargs='*', help='Training data paths (default: empty list)')
parser.add_argument('--early_stopping', default=False, help='Use early stopping', action='store_true')
parser.add_argument('--early_stopping_patience', default=10, type=int, help='Early stopping patience (default:10)')
parser.add_argument('--epochs', default=10, type=int, help='Training epochs (default:10)')
parser.add_argument('--ewc', default=False, help='Elastic weight consolidation (default:False)', action='store_true')
parser.add_argument('--ewc_lambda', default=1, type=float, help='Elastic weight consolidation importance parameter(default:1)')
parser.add_argument('--last_nfiles', default=1, type=int, help='Use last n files in training only (default:1, -1 for all)')
parser.add_argument('--min_iters', default=-1, type=int, help='Min training iterations (default:-1, negative for unlimited)')
parser.add_argument('--max_iters', default=-1, type=int, help='Max training iterations (default:-1, negative for unlimited)')
parser.add_argument('--new', default=False, help='Create a new model instead of training the old one', action='store_true')
parser.add_argument('--validation', default=False, help='Validation set (default:False)', action='store_true')
parser.add_argument('--val_episodes', default=0, type=float, help='Number of validation episodes (default:0)')
parser.add_argument('--val_mode', default=0, type=int, help='Validation mode (0: random, 1:episodic, default:0)')
parser.add_argument('--val_set_size', default=0.05, type=float, help='Validation set size (fraction of total) (default:0.05)')
parser.add_argument('--val_set_size_max', default=-1, type=int, help='Maximum validation set size (default:-1, negative for unlimited)')
parser.add_argument('--val_total', default=25, type=int, help='Total number of validations (default:25)')
parser.add_argument('--save_loss', default=False, help='Save loss history', action='store_true')
parser.add_argument('--save_interval', default=100, type=int, help='Number of iterations between save_loss')
parser.add_argument('--td', default=False, help='Temporal difference update', action='store_true')
parser.add_argument('--weighted', default=False, help='Weighted loss', action='store_true')
parser.add_argument('--weighted_mode', default=0, type=int, help='0: inverse of variance, 1: number of visits')
args = parser.parse_args()

batch_size = args.batch_size
cycle = args.cycle
early_stopping = args.early_stopping
early_stopping_patience = args.early_stopping_patience
epochs = args.epochs
ewc = args.ewc
ewc_lambda = args.ewc_lambda
last_nfiles = args.last_nfiles
min_iters = args.min_iters
max_iters = args.max_iters
new = args.new
validation = args.validation
val_episodes = args.val_episodes
val_mode = args.val_mode
val_set_size = args.val_set_size
val_set_size_max = args.val_set_size_max
val_total = args.val_total
save_loss = args.save_loss
save_interval = args.save_interval
td = args.td
weighted = args.weighted
weighted_mode = args.weighted_mode

#========================
"""
LOAD DATA
"""
list_of_data = []
for path in args.data_paths:
    list_of_data += glob.glob(path)
list_of_data.sort(key=os.path.getmtime)

if last_nfiles > 0:
    list_of_data = list_of_data[-last_nfiles:]

if len(list_of_data) == 0:
    raise Exception('list_of_data is empty.')

loader = DataLoader(list_of_data)

if td:
    if eligibility_trace:
        _child_stats = loader.child_stats
        n = _child_stats[:,0]
        q = _child_stats[:,3]

        _episode = loader.episode
        _score = loader.score
        _v = np.sum(n * q, axis=1) / np.sum(n, axis=1)
        values = np.zeros(_v.shape)
        for idx, ep in enumerate(_episode):
            idx_r = idx
            weight = 1.0
            _sum = 0
            _weight_sum = 0
            while idx_r < len(_episode) and _episode[idx_r] == _episode[idx] :
                _sum += weight * ( _score[idx_r] + _v[idx_r] - _score[idx] )
                _weight_sum += weight
                idx_r += 1
                weight *= eligibility_trace_lambda
            values[idx] = _sum / _weight_sum
        weights = np.ones(values.shape)
    else:
        values = loader.value
        variance = loader.variance
        if weighted:
            if weighted_mode == 0:
                weights = 1 / (variance + eps)
            elif weighted_mode == 1:
                weights = np.sum(loader.child_stats[:, 0], axis=1)
                weights = weights / np.average(weights)
            elif weighted_mode == 2:
                weights = np.sum(loader.child_stats[:, 0], axis=1)
                weights = weights / np.average(weights)
                weights = weights / (variance + eps)
        else:
            weights = np.ones(values.shape)
else:
    values = np.zeros((len(loader.score), ), dtype=np.float32)
    idx = 0
    while idx < len(loader.episode):
        for idx_end in range(idx, len(loader.episode)):
            if loader.episode[idx] != loader.episode[idx_end]:
                idx_end -= 1
                break
        ep_score = loader.score[idx_end]
        for _i in range(idx, idx_end+1):
            values[_i] = ep_score - loader.score[_i]
        idx = idx_end + 1
    variance = loader.variance 
    weights = np.ones(values.shape)

if backend == 'pytorch':
    states = loader.board
    policy = loader.policy
    def gen_batch(idx):
        return (np.expand_dims(states[idx], 1).astype(np.float32, copy=False), 
            np.expand_dims(values[idx], -1).astype(np.float32, copy=False),
            np.expand_dims(variance[idx], -1).astype(np.float32, copy=False),
            policy[idx].astype(np.float32, copy=False), 
            np.expand_dims(weights[idx], -1).astype(np.float32, copy=False))
                

#=========================
"""
VALIDATION SET
"""
if validation:
    if val_mode == 0:
        if val_set_size <= 0: 
            t_idx = list(range(len(states)))
            v_idx = []
        elif val_set_size >= 1:
            t_idx = []
            v_idx = list(range(len(states)))
        else:
            n_val_data = int(len(states) * val_set_size)
            if val_set_size_max > 0:
                v_idx = np.random.choice(len(states), size=min(n_val_data, val_set_size_max), replace=False)
            else:
                v_idx = np.random.choice(len(states), size=n_val_data, replace=False)
            t_idx = [x for x in range(len(states)) if x not in v_idx] 
    elif val_mode == 1:
        if val_episodes <= 0:    
            t_idx = list(range(len(states)))
            v_idx = []
        else:
            v_idx = np.where(loader.episode < val_episodes + 1)[0]
            if val_set_size_max > 0:
                idx = np.random.choice(v_idx[0], size=min(len(v_idx[0]), val_set_size_max), replace=False)
                v_idx = v_idx[0][idx]
            t_idx = np.where(loader.episode >= val_episodes + 1)[0]
else:
    t_idx = list(range(len(states)))

n_data = len(t_idx)

#=========================
"""
MODEL SETUP
"""

if backend == 'pytorch':
    from model.model_pytorch import Model
    m = Model(training=True, weighted=weighted, ewc=ewc, ewc_lambda=ewc_lambda)
    if not new:
        m.load()
    train_step = lambda batch, step: m.train(batch)
    compute_loss = lambda batch: m.compute_loss(batch)
    scheduler_step = lambda **kwargs: m.update_scheduler(**kwargs)
#=========================

iters_per_epoch = n_data//batch_size

iters = int(epochs * iters_per_epoch)

if max_iters >= 0:
    iters = int(min(iters, max_iters))

if min_iters >= 0:
    iters = int(max(iters, min_iters))

val_interval = iters // val_total + 1

if save_loss:
    #loss/loss_v/loss_var/loss_p
    hist_shape = (int(ceil(iters/save_interval)), 9)
    loss_history = np.empty(hist_shape)

#=========================
"""
TRAINING ITERATION
"""
loss_ma = 0
decay = 0.99

def loss_by_chunk(idx, chunksize=1000):
    loss = [0] * 5
    val_idx = 0
    length = len(idx)
    while val_idx < length:
        if val_idx + chunksize < length:
            idx_s = idx[val_idx:val_idx+chunksize]
        else:
            idx_s = idx[val_idx:]
        b_val = gen_batch(idx_s)
        _loss = compute_loss(b_val)
        for i in range(5):
            loss[i] += len(b_val[0]) * _loss[i] / length
        val_idx += chunksize
    return loss

if early_stopping:
    if not validation or len(v_idx) == 0:
        raise Exception('Early stopping without validation?')

    loss_val_best = float('inf')

    _epoch = 0

    _p = 0
    sys.stdout.write('\n')

    loss_history = []

    while _p < early_stopping_patience:
        
        loss_avg = defaultdict(float)

        for i in range(iters_per_epoch):
            
            batch = gen_batch(np.random.choice(t_idx, size=batch_size))
                
            _loss = train_step(batch, i)

            for k in _loss:
                loss_avg[k] += _loss[k]

        for k in _loss:
            loss_avg[k] /= iters_per_epoch

        _epoch += 1
        
        loss_val = compute_loss(gen_batch(v_idx))

        scheduler_step(metrics=loss_val)

        if loss_val['loss'] < loss_val_best:
            loss_val_best = loss_val['loss']
            _p = 0  
            m.save(verbose=False)
            suffix = ' <----- current model '
        else:
            _p += 1
            suffix = ''

        print('epoch:%d loss: %.5f/(%.5f±%.5f) %s' %(_epoch, loss_avg['loss'], loss_val['loss'], loss_val['loss_std'] / len(v_idx) ** 0.5,suffix), flush=True)

        if save_loss:
            loss_history.append([
                loss_avg['loss'],
                loss_avg['loss_v'],
                loss_avg['loss_var'],
                loss_avg['loss_p'],
                loss_val['loss'],
                loss_val['loss_v'],
                loss_val['loss_var'],
                loss_val['loss_p'],
                loss_val['loss_ewc']])

    loss_history = np.array(loss_history)
else:
    for i in range(iters):

        batch = gen_batch(np.random.choice(t_idx, size=batch_size))

        if validation and i % val_interval == 0:
            loss_val = compute_loss(gen_batch(v_idx))
            scheduler_step(metrics=loss_val['loss'])
            sys.stdout.write('\n')

        loss = train_step(batch,i)

        loss_ma = decay * loss_ma + ( 1 - decay ) * loss['loss']

        print('iter:%d/%d loss: %.5f/%.5f(%.5f)'%(i+1,iters,loss_ma,loss_val['loss'], loss_val['loss_std']), end='\r', flush=True)
            
        if save_loss and i % save_interval == 0:
            _idx = i // save_interval
            loss_history[_idx] = (
                loss_avg['loss'],
                loss_avg['loss_v'],
                loss_avg['loss_var'],
                loss_avg['loss_p'],
                loss_val['loss'],
                loss_val['loss_v'],
                loss_val['loss_var'],
                loss_val['loss_p'],
                loss_val['loss_ewc'])

if ewc:
    m.compute_fisher(batch_train)

print(flush=True)

if not early_stopping:
    m.save()


if save_loss:
    loss_saver = LossSaver(cycle)
    loss_saver.add(loss_history)
    loss_saver.close()

sys.stdout.write('\n')
sys.stdout.flush()
