from model.model import Model
from numba import jit, deferred_type
import tensorflow as tf
import numpy as np
import random
import  sys
sys.path.append('/home/rick/project/pyTetris/')
from pyTetris import Tetris
from copy import deepcopy
import pickle

np.set_printoptions(precision=3,suppress=True)

n_actions = 6

eps = 1e-7

class Node:
    def __init__(self,parent=None,action=None,game=None):
        if parent is not None and action is not None:
            self.parent = [(parent,action)]
        else:
            self.parent = []
        self.child = []
        #n,r,p,q,max_r
        self.child_stats = np.zeros((5,n_actions))
        self.game = game

@jit(nopython=True)
def select(_stats):
    for i in range(n_actions):
        if _stats[0][i] == 0:
            return i

    _t = np.sum(_stats[0])

    #_var = _stats[4] / _t - ( _stats[1] / _t ) ** 2

    _c = np.max(_stats[4]) * np.sqrt( np.log(_t) / ( 2 * _stats[0] ) )
    
    #_m = 2 * np.sqrt( np.log(_t) / _t ) / _stats[2]

    #_v = _stats[3] + _c - _m
    _v = _stats[3] + _c 

    return np.argmax(_v)

@jit(nopython=True)
def backup_stats(_stats,action,value):
    _stats[0][action] += 1
    _stats[1][action] += value
    _stats[3][action] = _stats[1][action] / _stats[0][action]
    _stats[4][action] = max(_stats[4][action],value)

class Agent:
    def __init__(self,conf,sims,tau=None):
        self.sims = sims
        self.model = Model()
        self.node_dict = dict()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model.load(self.sess)

    def evaluate(self,node):

        state = node.game.getState()

        state = np.expand_dims(state,0)
        state = np.expand_dims(state,-1)

        _r = self.model.inference(self.sess,state)

        v = _r[0][0][0]
        p = _r[1][0]

        return v, p

    def mcts(self,root_node):

        curr_node = root_node

        #select
        while curr_node.child:
            #n,r,p,q
            _stats = curr_node.child_stats
            
            max_i = select(_stats)

            curr_node = curr_node.child[max_i]

        #expand
        value = curr_node.game.getScore() - root_node.game.getScore()
        if not curr_node.game.end:

            v, policy = self.evaluate(curr_node)

            value += v
            
            _stats = curr_node.child_stats
            _stats[2] = policy

            g_childs = curr_node.child
            for i in range(n_actions):
                _g = curr_node.game.clone()
                _g.play(i)
                if _g in self.node_dict:
                    _n = self.node_dict[_g]
                    _n.parent.append((curr_node,i))
                    __stats = _n.child_stats
                    backup_stats(_stats,i,value)
                else:
                    _n = Node(curr_node,i,_g)
                    self.node_dict[_g] = _n
                g_childs.append(_n)

            

        #backup
        backup_list = curr_node.parent.copy()
        traversed = set()
        i = 0
        while i < len(backup_list):
            p, a = backup_list[i]
            if (id(p),a) not in traversed:
                traversed.add((id(p),a))
                backup_stats(p.child_stats,a,value)
                backup_list += p.parent
            i += 1

    def play(self):

        for i in range(self.sims):
            self.mcts(self.root)
        #print(self.root.child_stats)

        #n_temp = self.root.child_stats[0] ** (1/self.tau)
        #self.prob = n_temp / np.sum(n_temp)
        #action = np.random.choice(n_actions,p=self.prob)
        
        action = np.argmax(self.root.child_stats[3])
        self.prob = np.zeros(n_actions)
        self.prob[action] = 1.
        #print(action,self.prob)
        return action

    def get_prob(self):
        return self.prob

    def get_score(self):
        _stats = self.root.child_stats
        return np.sum(_stats[1])/np.sum(_stats[0])

    def get_stats(self):
        return self.root.child_stats

    def set_root(self,game):
        self.root = Node(game=game)
        self.root.game = game.clone()
        if game not in self.node_dict:
            self.node_dict[game] = self.root

    def update_root(self,game):
        if game in self.node_dict:
            self.root = self.node_dict[game]
        else:
            self.set_root(game)

    def save_nodes(self,save_path):
        _dict = self.node_dict
        states = [g.getState() for g in _dict.keys()]
        stats = [n.child_stats for n in _dict.values()]

        np.savez(save_path+'nodes.npz',np.stack(states),np.stack(stats))
