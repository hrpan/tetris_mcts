import numpy as np
import pickle 
import os
import sys
import glob
import argparse
import random
import pandas as pd
from util.Data import DataLoader
from math import ceil
"""
ARGS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--backend', default='pytorch', type=str, help='DL backend')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default:32)')
parser.add_argument('--cycle', default=-1, type=int, help='Cycle (default:-1)')
parser.add_argument('--data_dir', default=['./data'], nargs='*', help='Training data directories (default:./data/ep*)')
parser.add_argument('--eligibility_trace', default=False, action='store_true', help='Use eligibility trace')
parser.add_argument('--eligibility_trace_lambda', default=0.9, type=float, help='Lambda in eligibility trace (default:0.9)')
parser.add_argument('--epochs', default=10, type=int, help='Training epochs (default:10)')
parser.add_argument('--last_ncycles', default=1, type=int, help='Use last n cycles in training only (default:1, -1 for all)')
parser.add_argument('--max_iters', default=-1, type=int, help='Max training iterations (default:100000, negative for unlimited)')
parser.add_argument('--new', default=False, help='Create a new model instead of training the old one', action='store_true')
parser.add_argument('--val_split', default=0.1, type=float, help='Split ratio for validation data')
parser.add_argument('--val_split_max', default=-1, type=int, help='Maximum size for validation data (negative for unlimited)')
parser.add_argument('--val_total', default=25, type=int, help='Total number of validations')
parser.add_argument('--sarsa', default=False, help='Sarsa update', action='store_true')
parser.add_argument('--save_loss', default=False, help='Save loss history', action='store_true')
parser.add_argument('--save_interval', default=100, type=int, help='Number of iterations between save_loss')
parser.add_argument('--shuffle', default=False, help='Shuffle dataset', action='store_true')
args = parser.parse_args()

backend = args.backend
batch_size = args.batch_size
cycle = args.cycle
data_dir = args.data_dir
eligibility_trace = args.eligibility_trace
eligibility_trace_lambda = args.eligibility_trace_lambda
epochs = args.epochs
last_ncycles = args.last_ncycles
max_iters = args.max_iters
new = args.new
val_split = args.val_split
val_split_max = args.val_split_max
val_total = args.val_total
sarsa = args.sarsa
save_loss = args.save_loss
save_interval = args.save_interval
shuffle = args.shuffle

#========================
""" 
LOAD DATA 
"""
list_of_data = []
for _d in data_dir:
    list_of_data += glob.glob(_d+'/data*') 
list_of_data.sort(key=os.path.getmtime)

if last_ncycles > 0:
    list_of_data = list_of_data[-last_ncycles:]

if len(list_of_data) == 0:
    exit()

loader = DataLoader(list_of_data)    

if sarsa:
    _child_stats = loader.child_stats
    n = _child_stats[:,0]
    q = _child_stats[:,3]
    if eligibility_trace:
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
        values = np.sum(n * q, axis=1) / np.sum(n, axis=1)
else:
    values = np.zeros((loader.score[0], ), dtype=np.float32)
    while idx < len(loader.episode):
        idx_end = idx + 1
        while idx_end < len(loader.episode):
            if idx == len(loader.episode) - 1 or loader.episode[idx] != loader.episode[idx+1]:
                ep_score = loader.score[idx_end]
                break
        for _i in range(idx, idx_end+1):
            values[_i] = ep_score - self.score[_i]
        idx = idx_end

if backend == 'pytorch':
    states = np.expand_dims(loader.board, 1).astype(np.float32)
    policy = loader.policy.astype(np.float32)
    values = np.expand_dims(values, -1).astype(np.float32)
elif backend == 'tensorflow':
    states = np.expand_dims(np.stack(loader.['board'].values),-1)
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

#=========================
"""
VALIDATION SET
"""
if val_split_max >= 0:
    n_data = int(max(len(states) * ( 1 - val_split ), len(states) - val_split_max))
else:
    n_data = int(len(states) * ( 1 - val_split ))

batch_val = [states[n_data:], values[n_data:], policy[n_data:]]

batch_train = [states[:n_data], values[:n_data], policy[:n_data]]

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

val_interval = iters // val_total

loss_ma = 0
decay = 0.99

loss_val = 0

if save_loss:
    #loss_train/loss_value/loss_policy/loss_val/loss_val_v/loss_val_p
    hist_shape = (int(ceil(iters/save_interval)),6)
    loss_history = np.empty(hist_shape)

#=========================
"""
TRAINING ITERATION
"""

for i in range(iters):
    idx = np.random.randint(n_data,size=batch_size)

    batch = [_arr[idx] for _arr in batch_train]

    loss, loss_v, loss_p = train_step(batch,i)
    
    loss_ma = decay * loss_ma + ( 1 - decay ) * loss
    
    if i % val_interval == 0 and val_split > 0:
        loss_val, loss_val_v, loss_val_p = compute_loss(batch_val)
        scheduler_step(loss_val)
        
    sys.stdout.write('\riter:%d/%d loss: %.5f/%.5f'%(i,iters,loss_ma,loss_val))
    sys.stdout.flush()
        
    if i % save_interval == 0 and save_loss:
        _idx = i // save_interval
        loss_history[_idx] = (loss,loss_v,loss_p,loss_val,loss_val_v,loss_val_p)

sys.stdout.write('\n')
sys.stdout.flush()

if backend == 'tensorflow':
    m.save(sess)
elif backend == 'pytorch':
    m.save()

if save_loss:
    loss_file = 'data/loss'
    cols = ['loss',
            'loss_value',
            'loss_policy',
            'loss_val',
            'loss_val_value',
            'loss_val_policy']
    df = pd.DataFrame(data=loss_history,columns=cols)
    if os.path.isfile(loss_file):
        df_old = pd.read_pickle(loss_file)
        df = pd.concat([df_old,df],ignore_index=True)
    df.to_pickle(loss_file)

sys.stdout.write('\n')
sys.stdout.flush()
