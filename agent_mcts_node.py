from model.model import Model
from numba import jit, jitclass
from numba import int32, float32
import tensorflow as tf
import numpy as np
import random
import  sys
sys.path.append('/home/rick/project/pyTetris/')
from pyTetris import Tetris
from os.path import isfile
np.set_printoptions(precision=3,suppress=True)

n_actions = 6

eps = 1e-7

class Node:
    """
    DEPRECATED
    """
    def __init__(self,parent=None,action=None,game=None):
        if parent is not None and action is not None:
            self.parent = [(parent,action)]
        else:
            self.parent = []
        self.child = []
        #n,r,p,q,max_r
        self.child_stats = np.zeros((5,n_actions))
        self.game = game

@jit(nopython=True,cache=True)
def select(_stats):
    """
    DEPRECATED
    """
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

@jit(nopython=True,cache=True)
def backup_stats(_stats,action,value):
    """
    DEPRECATED
    """
    _stats[0][action] += 1
    _stats[1][action] += value
    _stats[3][action] = _stats[1][action] / _stats[0][action]
    _stats[4][action] = max(_stats[4][action],value)

@jit(nopython=True,cache=True)
def select_index(index,child,child_stats):

    trace = []

    while True:

        _stats = child_stats[index]

        _t = np.sum(_stats[0])

        trace.append(index)

        is_leaf_node = True
        for i in range(n_actions):
            if child[index][i] != 0:
                is_leaf_node = False
                break

        if is_leaf_node:
            break

        has_unvisited_node = False
        for i in range(n_actions):
            #if child_stats[index][0][i] == 0:
            if child_stats[index][0][i] < np.ceil( 8 * np.log( _t + 1 ) ) + 1:
                index = child[index][i]
                has_unvisited_node = True
                break

        if has_unvisited_node:
            continue

        #_c = np.max(_stats[4]) * np.sqrt( np.log(_t) / ( 2 * _stats[0] ) )
        _var = _stats[4] / _stats[0] - _stats[3] * _stats[3]

        _c = np.sqrt( 16 * _var * np.log(_t) / ( _stats[0] ) )

        _v = _stats[3] + _c 
        
        _a = np.argmax(_v)

        index = child[index][_a]

    return trace

@jit(nopython=True,cache=True)
def backup_index(index,n_parents,parent,child_stats,value):
    traversed = set([(0,0)])
    _np = n_parents[index]
    to_traverse_idx = []
    to_traverse_act = []
    for i in range(_np):
        to_traverse_idx.append(parent[index][i][0])
        to_traverse_act.append(parent[index][i][1])
    i = 0
    while i < len(to_traverse_act):
        index = to_traverse_idx[i]
        act = to_traverse_act[i]
        if (index,act) not in traversed:
            traversed.add((index,act))
            _stats = child_stats[index]
            _stats[0][act] += 1
            _stats[1][act] += value
            _stats[3][act] = _stats[1][act] / _stats[0][act]
            #_stats[4][act] = max(_stats[4][act],value)
            _stats[4][act] += value * value
            _np = n_parents[index]
            for j in range(_np):
                to_traverse_idx.append(parent[index][j][0])
                to_traverse_act.append(parent[index][j][1])
        i += 1

@jit(nopython=True,cache=True)
def backup_trace(trace,n_parents,parent,child_stats,value):

    for idx in trace:
        for i in range(n_parents[idx]):
            _i = parent[idx][i][0]
            _a = parent[idx][i][1]
            child_stats[_i][0][_a] += 1
            child_stats[_i][1][_a] += value
            child_stats[_i][3][_a] = child_stats[_i][1][_a] / child_stats[_i][0][_a]
            child_stats[_i][4][_a] += value * value

@jit(nopython=True,cache=True)
def copy_from_child(stats,action,curr,child):
    stats[curr][0][action] = np.sum(stats[child][0])
    stats[curr][1][action] = np.sum(stats[child][1])
    stats[curr][3][action] = stats[curr][1][action] / (stats[curr][0][action] + eps)
    stats[curr][4][action] = np.sum(stats[child][4])

@jit(nopython=True,cache=True)
def add_parent_jit(index,index_p,n_parents,parent,max_parents):

        parent[index][n_parents[index]] = index_p
        n_parents[index] += 1

        return n_parents[index] == max_parents
            
class Agent:
    def __init__(self,conf,sims,tau=None):
        self.sims = sims
        self.model = Model()
        self.node_dict = dict()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model.load(self.sess)

        self.current_node = 1
        init_nodes = 1000000
        init_p_nodes = 100
        
        n_parents_arr = np.zeros((init_nodes,),dtype=np.uint16)
        parent_arr = np.zeros((init_nodes,init_p_nodes,2),dtype=np.int32)
        child_arr = np.zeros((init_nodes,n_actions),dtype=np.int32)
        child_stats_arr = np.zeros((init_nodes,5,n_actions),dtype=np.float32)

        self.arrs = {'n_parents': n_parents_arr,
                'parent':parent_arr,
                'child':child_arr,
                'child_stats':child_stats_arr}
        self.game_arr = [0,]
        self.node_index_dict = dict()
        self.max_parents = init_p_nodes
        self.max_nodes = init_nodes

    def expand_nodes(self,n_nodes=10000):
        print('expanding nodes...')
        for k, arr in self.arrs.items():
            _s = arr.shape
            _new_s = [_ for _ in _s]
            _new_s[0] = n_nodes
            _temp_arr = np.zeros(_new_s,dtype=arr.dtype)
            self.arrs[k] = np.concatenate([arr,_temp_arr])

        self.max_nodes += n_nodes

    def expand_parents(self,n_parents=10):
        print('expanding parents...')
        _arr = self.arrs['parent']
        _s = _arr.shape
        _new_s = [_ for _ in _s]
        _new_s[1] = n_parents
        _temp_arr = np.zeros(_new_s,dtype=_arr.dtype)
        self.arrs['parent'] = np.concatenate([_arr,_temp_arr],axis=1)
    
    def evaluate(self,node):

        state = node.game.getState()
        
        return self.evaluate_state(state)

    def evaluate_state(self,state):

        state = np.expand_dims(state,0)
        state = np.expand_dims(state,-1)

        _r = self.model.inference(self.sess,state)

        v = _r[0][0][0]
        p = _r[1][0]

        return v, p
   
    def mcts_index(self,root_index):
        trace = select_index(root_index,self.arrs['child'],self.arrs['child_stats'])

        leaf_index = trace[-1]
        #leaf_index = select_index(root_index,self.arrs['child'],self.arrs['child_stats'])

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore() - self.game_arr[root_index].getScore()


        if not leaf_game.end:

            v, p = self.evaluate_state(leaf_game.getState())

            value += v

            for i in range(n_actions):
                _g = leaf_game.clone()
                _g.play(i)
                _n = self.new_node(_g)
                copy_from_child(self.arrs['child_stats'],i,leaf_index,_n)
                self.add_parent(_n,(leaf_index,i))
                self.arrs['child'][leaf_index][i] = _n
        """
        backup_index(leaf_index,
                self.arrs['n_parents'],
                self.arrs['parent'],
                self.arrs['child_stats'],
                value)
        """
        backup_trace(trace,
                self.arrs['n_parents'],
                self.arrs['parent'],
                self.arrs['child_stats'],
                value)
    def mcts(self,root_node):
        """
        DEPRECATED
        """
        curr_node = root_node

        #select
        global max_depth
        depth = 0
        while curr_node.child:
            depth += 1
            #n,r,p,q
            _stats = curr_node.child_stats
            
            max_i = select(_stats)

            curr_node = curr_node.child[max_i]
        max_depth = max(depth,max_depth)
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
                    _stats[0][i] = np.sum(__stats[0])
                    _stats[1][i] = np.sum(__stats[1])
                    _stats[3][i] = _stats[1][i] / (_stats[0][i]+eps)
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
            self.mcts_index(self.root)
        action = np.argmax(self.arrs['child_stats'][self.root][3])
        self.prob = np.zeros(n_actions)
        self.prob[action] = 1.
        return action

    def get_prob(self):
        return self.prob

    def get_score(self):
        _stats = self.arrs['child_stats'][self.root]
        return np.sum(_stats[1])/np.sum(_stats[0])

    def get_stats(self):
        return self.arrs['child_stats'][self.root]

    def add_parent(self,index,parent):
        """
        _np = self.arrs['n_parents'][index]

        if _np + 1 > self.max_parents:
            self.expand_parents()
        self.arrs['n_parents'][index] += 1
        self.arrs['parent'][index][_np] = parent
        """
        if add_parent_jit(index,
                parent,
                self.arrs['n_parents'],
                self.arrs['parent'],
                self.max_parents):
            self.expand_parents()

    def new_node(self,game):
        
        if game in self.node_index_dict:

            return self.node_index_dict[game]

        else:
            idx = self.current_node

            self.node_index_dict[game] = idx

            if idx + 1 > self.max_nodes:
                self.expand_nodes()
            
            self.game_arr.append(game.clone())
            self.current_node += 1

            return idx

    def set_root(self,game):

        self.root = self.new_node(game)

        """
        self.root = Node(game=game)
        self.root.game = game.clone()
        if game not in self.node_dict:
            self.node_dict[game] = self.root
        """
    def update_root(self,game):
        if game in self.node_index_dict:
            self.root = self.node_index_dict[game]
        else:
            self.set_root(game)
        """
        if game in self.node_dict:
            self.root = self.node_dict[game]
        else:
            self.set_root(game)
        """
    def save_nodes(self,save_path):
        _dict = self.node_dict
        states = np.stack([self.game_arr[i].getState() for i in range(1,self.current_node)])
        stats = self.arrs['child_stats'][1:self.current_node]

        visits = np.sum(stats[:,0],axis=1)

        idx = np.where(visits > 0)

        states = states[idx]
        stats = stats[idx]
       
        count = 0
        while isfile(save_path+'/nodes_%d.npz'%count):
            count += 1

        np.savez(save_path+'/nodes_%d.npz'%count,np.stack(states),np.stack(stats))
