from model.model_pytorch import Model
from numba import jit, jitclass
from numba import int32, float32
import numpy as np
import random
import  sys
sys.path.append('/home/rick/project/pyTetris/')
from pyTetris import Tetris
from os.path import isfile
np.set_printoptions(precision=3,suppress=True)

n_actions = 6

eps = 1e-7

@jit(nopython=True,cache=True)
def select_index(index,child,node_stats):

    trace = []

    while True:

        trace.append(index)

        is_leaf_node = True

        _child_nodes = []
        for i in range(n_actions):
            if child[index][i] != 0:
                _child_nodes.append(child[index][i])

        len_c = len(_child_nodes)

        if len_c == 0:
            break

        has_unvisited_node = False

        _stats = np.zeros((2,len_c))

        for i in range(len_c):
            _idx = _child_nodes[i]
            _stats[0][i] = node_stats[_idx][0]
            _stats[1][i] = node_stats[_idx][1]
            if node_stats[_idx][0] == 0:
                index = _idx
                has_unvisited_node = True
                break

        if has_unvisited_node:
            continue

        _t = np.sum(_stats[0]) 

        _c = 1.0 * np.sqrt( np.log(_t) / _stats[0] )

        _q = _stats[1] / _stats[0]

        _v = _q + _c 
        
        _a = np.argmax(_v)

        index = _child_nodes[_a]

    return trace

@jit(nopython=True,cache=True)
def backup_trace(trace,node_stats,value):

    for idx in trace:
        v = value - node_stats[idx][2] 
        node_stats[idx][0] += 1
        node_stats[idx][1] += v

@jit(nopython=True,cache=True)
def get_all_childs(index,child):
    
    to_traverse = [index,]

    traversed = []

    i = 0
    while i < len(to_traverse):
        idx = to_traverse[i]
        if idx not in traversed:
            to_traverse += list(child[idx])   
            traversed.append(idx)
        i += 1
    return set(to_traverse)

@jit(nopython=True,cache=True)
def choose_action(p):
    _cdf = p.cumsum()

    rnd = np.random.rand()

    _a = np.searchsorted(_cdf,rnd)

    return _a

class Agent:
    def __init__(self,conf,sims,tau=None,gamma=0.99):
        self.sims = sims
        self.gamma = gamma
        self.model = Model()
        self.node_dict = dict()

        self.current_node = 1
        init_nodes = 500000
        
        child_arr = np.zeros((init_nodes,n_actions),dtype=np.int32)
        node_stats_arr = np.zeros((init_nodes,3),dtype=np.float32)

        self.arrs = {'child':child_arr,'node_stats':node_stats_arr}
        self.game_arr = [None] * init_nodes
        self.available = [i for i in range(1,init_nodes)]
        self.occupied = [0]
        self.node_index_dict = dict()
        self.max_nodes = init_nodes

    def expand_nodes(self,n_nodes=10000):
        print('expanding nodes...')
        for k, arr in self.arrs.items():
            _s = arr.shape
            _new_s = [_ for _ in _s]
            _new_s[0] = n_nodes
            _temp_arr = np.zeros(_new_s,dtype=arr.dtype)
            self.arrs[k] = np.concatenate([arr,_temp_arr])
        self.game_arr += [None] * n_nodes
        self.available += [i for i in range(self.max_nodes,self.max_nodes+n_nodes)]
        #self.occupied_arr += [False] * n_nodes
        self.max_nodes += n_nodes

    def evaluate(self,node):

        state = node.game.getState()
        
        return self.evaluate_state(state)

    def evaluate_state(self,state):

        state = np.expand_dims(state,0)
        state = np.expand_dims(state,0)

        v, p = self.model.inference(state)
        
        return v[0][0], p[0]

    def mcts_index(self,root_index):
        trace = select_index(root_index,self.arrs['child'],self.arrs['node_stats'])

        leaf_index = trace[-1]
        #leaf_index = select_index(root_index,self.arrs['child'],self.arrs['child_stats'])

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore() #- self.game_arr[root_index].getScore()


        if not leaf_game.end:

            v, p = self.evaluate_state(leaf_game.getState())

            _g = leaf_game.clone()

            while not _g.end:
                v, p = self.evaluate_state(_g.getState())
                _act = choose_action(p)
                _g.play(_act)

            value = _g.getScore()

            for i in range(n_actions):
                _g = leaf_game.clone()
                _g.play(i)
                _n = self.new_node(_g)

                self.arrs['child'][leaf_index][i] = _n
                self.arrs['node_stats'][_n][2] = _g.getScore()
            
        
        backup_trace(trace,self.arrs['node_stats'],value)

    def play(self):

        for i in range(self.sims):
            self.mcts_index(self.root)
        #print(len(get_all_childs(self.root,self.arrs['child'])),self.current_node)
        self.stats = self.compute_stats()
        if np.all(self.stats[3] == 0):
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(self.stats[3])
        self.prob = self.stats[0] / np.sum(self.stats[0])
        #self.prob = np.zeros(n_actions)
        #self.prob[action] = 1.
        return action

    def compute_stats(self):
        _stats = np.zeros((4,n_actions))
        _childs = self.arrs['child'][self.root]
        _ns = self.arrs['node_stats']
        for i in range(n_actions):
            _idx = self.arrs['child'][self.root][i]
            _stats[0][i] = _ns[_idx][0]
            _stats[1][i] = _ns[_idx][1]
            _stats[2][i] = 0
            _stats[3][i] = _ns[_idx][1] / _ns[_idx][0]
        
        return _stats

    def get_prob(self):
        return self.prob

    def get_score(self):
        _stats = self.arrs['child_stats'][self.root]
        return np.sum(_stats[1])/np.sum(_stats[0])

    def get_stats(self):
       
        return self.stats

    def new_node(self,game):
        
        if game in self.node_index_dict:

            return self.node_index_dict[game]

        else:

            if self.available:
                idx = self.available.pop()
            else:
                self.remove_nodes()
                if self.available:
                    idx = self.available.pop()
                else:
                    self.expand_nodes()
                    idx = self.available.pop()

            _g = game.clone()

            self.node_index_dict[_g] = idx
            
            self.game_arr[idx] = _g

            self.occupied.append(idx)

            return idx

    def remove_nodes(self):
        
        _c = get_all_childs(self.root,self.arrs['child'])
        for idx in self.occupied:
            if idx not in _c:
                _g = self.game_arr[idx]

                del self.node_index_dict[_g]

                self.game_arr[idx] = None

                for k, arr in self.arrs.items():

                    arr[idx] = 0
                
                self.occupied.remove(idx)
                self.available.append(idx)
                #self.occupied_arr[i] = False

    def set_root(self,game):

        self.root = self.new_node(game)

    def update_root(self,game):
        if game in self.node_index_dict:
            self.root = self.node_index_dict[game]
        else:
            self.set_root(game)

        #self.remove_nodes()

    def save_nodes(self,save_path):
        """
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
        """
