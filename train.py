import numpy as np
import pickle 
import os
import sys
import glob
import argparse
import random
import pandas as pd
from math import ceil
"""
ARGS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--backend',default='pytorch',type=str,help='DL backend')
parser.add_argument('--cycle',default=-1,type=int,help='Cycle (default:-1)')
parser.add_argument('--new',default=False,help='Create a new model instead of training the old one',action='store_true')
parser.add_argument('--batch_size',default=32,type=int,help='Batch size (default:32)')
parser.add_argument('--epochs',default=10,type=int,help='Training epochs (default:10)')
parser.add_argument('--max_iters',default=-1,type=int,help='Max training iterations (default:100000, negative for unlimited)')
parser.add_argument('--data_dir',default=['./data'],nargs='*',help='Training data directories (default:./data/ep*)')
parser.add_argument('--gamma',default=0.99,type=float,help='Gamma for discounted rewards')
parser.add_argument('--save_loss',default=False,help='Save loss history',action='store_true')
parser.add_argument('--save_interval',default=100,type=int,help='Number of iterations between save_loss')
parser.add_argument('--shuffle',default=False,help='Shuffle dataset',action='store_true')
parser.add_argument('--val_split',default=0.1,type=float,help='Split ratio for validation data')
parser.add_argument('--val_split_max',default=-1,type=int,help='Maximum size for validation data (negative for unlimited)')
#parser.add_argument('--val_interval',default=1000,type=int,help='Number of iterations between validation loss')
args = parser.parse_args()

backend = args.backend
cycle = args.cycle
new = args.new
batch_size = args.batch_size
epochs = args.epochs
data_dir = args.data_dir
gamma = args.gamma
max_iters = args.max_iters
save_loss = args.save_loss
save_interval = args.save_interval
shuffle = args.shuffle
val_split = args.val_split
val_split_max = args.val_split_max
#val_interval = args.val_interval

#========================
""" 
LOAD DATA 
"""
list_of_data = []
for _d in data_dir:
    list_of_data += glob.glob(_d+'/data*') 

dfs = []
for file_name in list_of_data:
    df = pd.read_pickle(file_name)
    dfs.append(df)

if dfs:    
    df = pd.concat(dfs,ignore_index=True)
else:
    exit()

b_shape = df['board'][0].shape

if backend == 'pytorch':
    states = np.expand_dims(np.stack(df['board'].values),1).astype(np.float32)
    policy = np.stack(df['policy'].values).astype(np.float32)
    labels = np.expand_dims(np.stack(df['cum_score'].values),-1).astype(np.float32)
elif backend == 'tensorflow':
    states = np.expand_dims(np.stack(df['board'].values),-1)
    policy = np.stack(df['policy'].values)
    labels = np.expand_dims(np.stack(df['cum_score'].values),-1)

#========================
"""
Shuffle
"""
if shuffle:
    indices = np.random.permutation(len(states))

    states = states[indices]
    policy = policy[indices]
    labels = labels[indices]

#=========================
"""
VALIDATION SET
"""
if val_split_max >= 0:
    n_data = int(max(len(states) * ( 1 - val_split ),len(states) - val_split_max))
else:
    n_data = int(len(state) * ( 1 - val_split ))

batch_val = [states[n_data:],labels[n_data:],policy[n_data:]]

batch_train = [states[:n_data],labels[:n_data],policy[:n_data]]

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

val_interval = iters_per_epoch

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

if backend =='tensorflow':
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
