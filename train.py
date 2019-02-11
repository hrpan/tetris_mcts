import numpy as np
import pickle 
import os
import sys
import glob
import argparse
import random
from util.Data import DataLoader, LossSaver
from math import ceil
"""
ARGS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--backend', default='pytorch', type=str, help='DL backend')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default:32)')
parser.add_argument('--cycle', default=-1, type=int, help='Cycle (default:-1)')
parser.add_argument('--data_paths', default=[], nargs='*', help='Training data paths (default: empty list)')
parser.add_argument('--eligibility_trace', default=False, action='store_true', help='Use eligibility trace')
parser.add_argument('--eligibility_trace_lambda', default=0.9, type=float, help='Lambda in eligibility trace (default:0.9)')
parser.add_argument('--epochs', default=10, type=int, help='Training epochs (default:10)')
parser.add_argument('--last_nfiles', default=1, type=int, help='Use last n files in training only (default:1, -1 for all)')
parser.add_argument('--max_iters', default=-1, type=int, help='Max training iterations (default:100000, negative for unlimited)')
parser.add_argument('--new', default=False, help='Create a new model instead of training the old one', action='store_true')
parser.add_argument('--val_episodes', default=0, type=float, help='Number of validation episodes (default:0)')
parser.add_argument('--val_total', default=25, type=int, help='Total number of validations (default:25)')
parser.add_argument('--save_loss', default=False, help='Save loss history', action='store_true')
parser.add_argument('--save_interval', default=100, type=int, help='Number of iterations between save_loss')
parser.add_argument('--shuffle', default=False, help='Shuffle dataset', action='store_true')
parser.add_argument('--target_normalization', default=False, help='Standardizes the targets', action='store_true')
parser.add_argument('--td', default=False, help='Temporal difference update', action='store_true')
args = parser.parse_args()

backend = args.backend
batch_size = args.batch_size
cycle = args.cycle
data_paths = args.data_paths
eligibility_trace = args.eligibility_trace
eligibility_trace_lambda = args.eligibility_trace_lambda
epochs = args.epochs
last_nfiles = args.last_nfiles
max_iters = args.max_iters
new = args.new
val_episodes = args.val_episodes
val_total = args.val_total
save_loss = args.save_loss
save_interval = args.save_interval
shuffle = args.shuffle
target_normalization = args.target_normalization
td = args.td

#========================
""" 
LOAD DATA 
"""
list_of_data = []
for path in data_paths:
    list_of_data += glob.glob(path) 
list_of_data.sort(key=os.path.getmtime)

if last_nfiles > 0:
    list_of_data = list_of_data[-last_nfiles:]

if len(list_of_data) == 0:
    exit()

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
    else:
        values = loader.value
        variance = loader.variance
else:
    values = np.zeros((len(loader.score), ), dtype=np.float32)
    while idx < len(loader.episode):
        idx_end = idx + 1
        while idx_end < len(loader.episode):
            if idx == len(loader.episode) - 1 or loader.episode[idx] != loader.episode[idx+1]:
                ep_score = loader.score[idx_end]
                break
        for _i in range(idx, idx_end+1):
            values[_i] = ep_score - loader.score[_i]
        idx = idx_end
    variance = loader.variance 

if backend == 'pytorch':
    states = np.expand_dims(loader.board, 1).astype(np.float32)
    policy = loader.policy.astype(np.float32)
    values = np.expand_dims(values, -1).astype(np.float32)
elif backend == 'tensorflow':
    states = np.expand_dims(np.stack(loader['board'].values),-1)
    policy = loader.policy
    values = np.expand_dims(values,-1)

#========================
"""
Shuffle
"""
if shuffle:
    indices = np.random.permutation(len(states))

    states = states[indices]
    policy = policy[indices]
    values = values[indices]
    variance = variance[indices]
#=========================
"""
VALIDATION SET
"""
v_idx = np.where(loader.episode < val_episodes + 1)
t_idx = np.where(loader.episode >= val_episodes + 1)

n_data = len(t_idx[0])

batch_val = [states[v_idx], values[v_idx], variance[v_idx], policy[v_idx]]

batch_train = [states[t_idx], values[t_idx], variance[t_idx], policy[t_idx]]

#=========================
"""
TARGET NORMALIZATION
"""
if target_normalization:
    v_mean = batch_train[1].mean()
    v_std = batch_train[1].std()
    var_mean = batch_train[2].mean()
    var_std = batch_train[2].std()
    batch_train[1] = (batch_train[1] - v_mean) / v_std
    batch_train[2] = (batch_train[2] - var_mean) / var_std
    batch_val[1] = (batch_val[1] - v_mean) / v_std
    batch_val[2] = (batch_val[2] - var_mean) / var_std

#=========================
"""
MODEL SETUP
"""

if backend == 'pytorch':
    from model.model_pytorch import Model
    m = Model()
    if not new:
        m.load()
    train_step = lambda batch, step: m.train(batch)
    compute_loss = lambda batch: m.compute_loss(batch)
    scheduler_step = lambda val_loss: m.update_scheduler(val_loss)
    if target_normalization:
        m.v_mean = v_mean
        m.v_std = v_std
        m.var_mean = var_mean
        m.var_std = var_std
elif backend == 'tensorflow':
    from model.model import Model
    import tensorflow as tf
    sess = tf.Session()
    m = Model()
    if new:
        m.build_graph()
        sess.run(tf.global_variables_initializer())
    else:
        m.load(sess)
    train_step = lambda batch, step: m.train(sess,batch,step)
    compute_loss = lambda batch: m.compute_loss(sess,batch)
    scheduler_step = lambda val_loss: None
#=========================

iters_per_epoch = n_data//batch_size

if max_iters >= 0:
    iters = int(min( epochs * iters_per_epoch, max_iters))
else:
    iters = int(epochs * iters_per_epoch)

val_interval = iters // val_total + 1

if save_loss:
    #loss/loss_v/loss_var/loss_p
    hist_shape = (int(ceil(iters/save_interval)), 8)
    loss_history = np.empty(hist_shape)

#=========================
"""
TRAINING ITERATION
"""
loss_ma = 0
decay = 0.99
loss_val = 0

for i in range(iters):
    idx = np.random.randint(n_data,size=batch_size)

    batch = [_arr[idx] for _arr in batch_train]

    loss, loss_v, loss_var, loss_p = train_step(batch,i)
    
    loss_ma = decay * loss_ma + ( 1 - decay ) * loss
    
    if i % val_interval == 0 and val_episodes > 0:
        loss_val, loss_val_v, loss_val_var, loss_val_p = compute_loss(batch_val)
        scheduler_step(loss_val)
        
    sys.stdout.write('\riter:%d/%d loss: %.5f/%.5f'%(i,iters,loss_ma,loss_val))
    sys.stdout.flush()
        
    if i % save_interval == 0 and save_loss:
        _idx = i // save_interval
        loss_history[_idx] = (loss, 
            loss_v, 
            loss_var,
            loss_p,
            loss_val,
            loss_val_v,
            loss_val_var,
            loss_val_p)

sys.stdout.write('\n')
sys.stdout.flush()

if backend == 'tensorflow':
    m.save(sess)
elif backend == 'pytorch':
    m.save()


if save_loss:
    loss_saver = LossSaver(cycle)
    loss_saver.add(loss_history)
    loss_saver.close()

sys.stdout.write('\n')
sys.stdout.flush()
