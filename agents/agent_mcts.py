from model.model import Model
from numba import jit, deferred_type
import numpy as np
import random
import  sys
sys.path.append('/home/rick/project/pyTetris/')
from pyTetris import Tetris
from copy import deepcopy
import pickle

np.set_printoptions(precision=3,suppress=True)

n_actions = 6

class Tree:
    def __init__(self,parent,action=None):
        self.parent = parent
        self.child = None
        #n,r,p,q,max_r
        self.child_stats = np.zeros((5,n_actions))
        self.action = action  
        self.visited = False
        self.game = None
    def copy_and_play(self):
#        self.game = deepcopy(self.parent.game)
        self.game = self.parent.game.clone()
        self.game.play(self.action)
        self.visited = True

eps = 3e-1

@jit
def select(_stats):
    for i in range(n_actions):
        if _stats[0][i] == 0:
#        if _stats[0][i] < 10:
            return i

    _t = np.sum(_stats[0])

    #_var = _stats[4] / _t - ( _stats[1] / _t ) ** 2

    _c = np.amax(_stats[4]) * np.sqrt( np.log(_t) / ( 2 * _stats[0] ) )
    
    #_m = 2 * np.sqrt( np.log(_t) / _t ) / _stats[2]

    #_v = _stats[3] + _c - _m
    _v = _stats[3] + _c 

    return np.argmax(_v)

@jit
def backup(_stats,action,value):
    _stats[0][action] += 1
    _stats[1][action] += value
    _stats[3][action] = _stats[1][action] / _stats[0][action]
#    _stats[4][action] += value * value
    _stats[4][action] = np.amax((_stats[4][action],value))

class Agent:
    def __init__(self,conf,sims,tau=1):
        self.c = conf
        self.sims = sims
        self.tau = tau
        self.model = Model()
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
        if not curr_node.visited:
            curr_node.copy_and_play()

        value = curr_node.game.getScore() - root_node.game.getScore()
        
        if not curr_node.game.end:

            v, policy = self.evaluate(curr_node)

            value += v
            
            _stats = curr_node.child_stats
            #_stats[0] = eps
            _stats[2] = policy
            curr_node.child = [Tree(curr_node,action=i) for i in range(n_actions)]

        #curr_node.game.printState()
        #print(value,v)
        #backup
        while curr_node is not root_node:
            parent = curr_node.parent
            action = curr_node.action
            _stats = parent.child_stats
            backup(_stats,action,value)
            """
            _stats[0][action] += 1
            _stats[1][action] += value
            _stats[3][action] = _stats[1][action] / _stats[0][action]
            """
            #print(_stats)
            curr_node = parent

    def play(self):

        for i in range(self.sims):
            self.mcts(self.root)
        #print(self.root.child_stats)      
        #input()
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
        self.root = Tree(None,None)
        self.root.game = game.clone()
        self.root.visited = True

    def update_root(self,action,game):
        child = self.root.child
        if child[action].game.equiv(game):
            self.root = child[action]
        else:
            self.set_root(game)
