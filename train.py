from model.model import Model
import tensorflow as tf
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
parser.add_argument('--cycle',default=-1,type=int,help='Cycle (default:-1)')
parser.add_argument('--new',default=False,help='Create a new model instead of training the old one',action='store_true')
parser.add_argument('--batch_size',default=32,type=int,help='Batch size (default:32)')
parser.add_argument('--epochs',default=10,type=int,help='Training epochs (default:10)')
parser.add_argument('--max_iters',default=100000,type=int,help='Max training iterations (default:100000)')
parser.add_argument('--data_dir',default=['./data'],nargs='*',help='Training data directories (default:./data/ep*)')
parser.add_argument('--nodes',default=False,help='Train on node stats instead of playouts',action='store_true')
parser.add_argument('--node_thresh',default=0,type=float,help='Minimum visits for a node to be trained (quantile)')
parser.add_argument('--n_episodes',default=-1,type=int,help='Number of training episodes (randomly chosen)')
parser.add_argument('--save_loss',default=False,help='Save loss history',action='store_true')
parser.add_argument('--save_interval',default=100,type=int,help='Number of iterations between save_loss')
parser.add_argument('--shuffle',default=False,help='Shuffle dataset',action='store_true')
parser.add_argument('--val_split',default=0.1,type=float,help='Split ratio for validation data')
parser.add_argument('--val_interval',default=1000,type=int,help='Number of iterations between validation loss')
args = parser.parse_args()

cycle = args.cycle
new = args.new
batch_size = args.batch_size
epochs = args.epochs
data_dir = args.data_dir
nodes = args.nodes
node_thresh = args.node_thresh
n_episodes = args.n_episodes
max_iters = args.max_iters
save_loss = args.save_loss
save_interval = args.save_interval
shuffle = args.shuffle
val_split = args.val_split
val_interval = args.val_interval

""" 
LOAD DATA 
"""
if nodes:
    list_of_data = []
    for _d in data_dir:
        list_of_data += glob.glob(_d+'/nodes*')

    states = []
    stats = []
    for file_name in list_of_data:
        with np.load(file_name) as _f:
            states.append(_f['arr_0'])
            stats.append(_f['arr_1'])

    states = np.concatenate(states)
    stats = np.concatenate(stats)

    node_visits = np.sum(stats[:,0],axis=1)
    q = np.percentile(node_visits,node_thresh)
    idx = np.where(node_visits > q)

    states = states[idx]
    stats = stats[idx]

    values = np.sum(stats[:,1],axis=1) / node_visits[idx]
    states = np.expand_dims(states,-1)
    labels = np.expand_dims(values,-1)
    policy = np.zeros((len(states),6))
else:
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


    states = np.expand_dims(np.stack(df['board'].values),-1)
    policy = np.stack(df['policy'].values)
    labels = np.stack(df['cum_score'].values)[:,None]

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

n_data = int(len(states) * ( 1 - val_split ))

batch_val = [states[n_data:],labels[n_data:],policy[n_data:]]

batch_train = [states[:n_data],labels[:n_data],policy[:n_data]]

#=========================


iters = int(min(epochs*n_data//batch_size,max_iters))

loss_ma = 0
decay = 0.99

loss_val = 0

if save_loss:
    #loss_train/loss_value/loss_policy/loss_val/loss_val_v/loss_val_p
    hist_shape = (int(ceil(iters/save_interval)),6)
    loss_history = np.empty(hist_shape)

with tf.Session() as sess:
    m = Model()
    if new:
        m.build_graph()
        sess.run(tf.global_variables_initializer())
    else:
        m.load(sess)
    
    for i in range(iters):

        idx = np.random.randint(n_data,size=batch_size)
        batch = [batch_train[0][idx],
                batch_train[1][idx],
                batch_train[2][idx]]
        
        loss, loss_v, loss_p = m.train(sess,batch,i)
        
        loss_ma = decay * loss_ma + ( 1 - decay ) * loss
        
        if i % val_interval == 0 and val_split > 0:
            loss_val, loss_val_v, loss_val_p = m.compute_loss(sess,batch_val)

        sys.stdout.write('\riter:%d/%d loss: %.5f/%.5f'%(i,iters,loss_ma,loss_val))
        sys.stdout.flush()
            
        if i % save_interval == 0 and save_loss:
            _idx = i // save_interval
            loss_history[_idx] = (loss,loss_v,loss_p,loss_val,loss_val_v,loss_val_p)
  
    m.save(sess)

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
