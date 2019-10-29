import collections
import numpy as np
from agents.agent import Agent
from agents.core import *
import sys
import torch.optim as optim
eps = 1e-7

class ValueSimOnline(Agent):

    def __init__(self, conf, sims, tau=None, backend='pytorch', env=None, env_args=None, backup_alpha=0.01, n_actions=7, memory_size=250000, benchmark=False):

        super().__init__(sims=sims,backend=backend,env=env, env_args=env_args, n_actions=n_actions, init_nodes=500000, stochastic_inference=False)

        #self.model.ewc = True

        self.g_tmp = env(*env_args)

        self.backup_alpha = backup_alpha

        self.benchmark = benchmark

        self.memory_size = memory_size

        self.memory = [[], [], [], []]

        self.n_trains = 0

    def mcts(self,root_index):

        _child = self.arrs['child']
        _node_stats = self.arrs['node_stats']

        trace = select_index_clt(root_index, _child, _node_stats)

        leaf_index = trace[-1]

        leaf_game = self.game_arr[leaf_index]

        value = leaf_game.getScore()

        if not leaf_game.end:

            v, var, p = self.evaluate_state(leaf_game.getState())

            _node_stats[leaf_index][3] = var

            value += v
            
            _g = self.g_tmp
            for i in range(self.n_actions):
                _g.copy_from(leaf_game)
                _g.play(i)
                _n = self.new_node(_g)
                _child[leaf_index][i] = _n
                _node_stats[_n][2] = _g.getScore()

        #backup_trace_3(trace, _node_stats, value, alpha=self.backup_alpha)
        backup_trace_welford_v2(trace, _node_stats, value)

    def compute_stats(self, node=None):

        if node == None:
            node = self.root

        _stats = np.zeros((6, self.n_actions), dtype=np.float32)

        _childs = self.arrs['child'][node]
        _ns = self.arrs['node_stats']

        counter = collections.Counter(_childs)        

        root_score = _ns[node][2]        

        _stats[2] = 0
        for i in range(self.n_actions):
            _idx = _childs[i]
            if _ns[_idx][0] < 1:
                return False
            _stats[0][i] = _ns[_idx][0] / counter[_idx]
            _stats[1][i] = _ns[_idx][1] + _ns[_idx][2] - root_score
            _stats[3][i] = _ns[_idx][1] + _ns[_idx][2] - root_score
            _stats[4][i] = _ns[_idx][3]
            _stats[5][i] = _ns[_idx][4]

        return _stats

    def get_value(self, node=None):

        if node == None:
            node = self.root

        _ns = self.arrs['node_stats'][node]

        return _ns[1], _ns[3] 

    def remove_nodes(self):

        sys.stderr.write('\nWARNING: REMOVING UNUSED NODES...\n')

        _c = get_all_childs(self.root,self.arrs['child'])
        self.occupied.clear()
        self.occupied.extend(_c)
        self.available.clear()
        a_app = self.available.append
        for i in range(self.max_nodes): 
            if i not in _c:
                a_app(i)
        sys.stderr.write('Number of occupied nodes: ' + str(len(self.occupied)) + '\n')
        sys.stderr.write('Number of available nodes: ' + str(len(self.available)) + '\n')
        sys.stderr.flush()

        if not self.benchmark:
            self.store_nodes(self.available)
            self.train_nodes()

        for idx in self.available:
            _g = self.game_arr[idx]

            #del self.node_index_dict[_g]
            self.node_index_dict.pop(_g, None)

            self.arrs['child'][idx].fill(0)
            self.arrs['child_stats'][idx].fill(0)
            self.arrs['node_stats'][idx].fill(0)

    def store_nodes(self, nodes, min_visits=30):

        sys.stderr.write('Storing unused nodes...\n')
        sys.stderr.flush()

        states, values, variance, weights = self.memory

        g_arr = self.game_arr        
        n_stats = self.arrs['node_stats']
        c_stats = self.arrs['child_stats']

        for idx in nodes:
            if n_stats[idx][0] < min_visits:
                continue
            states.append(g_arr[idx].getState())
            values.append(n_stats[idx][1])
            variance.append(n_stats[idx][3])
            weights.append(n_stats[idx][0])

    def train_nodes(self, batch_size=128, iters_per_val=500, loss_threshold=5, val_fraction=0.1, patience=20):

        sys.stderr.write('Training...\n')
        sys.stderr.flush()

        states, values, variance, weights = self.memory

        states = np.expand_dims(np.array(states, dtype=np.float32), 1)
        values = np.expand_dims(np.array(values, dtype=np.float32), -1)
        variance = np.expand_dims(np.array(variance, dtype=np.float32), -1)
        variance += 1
        weights = np.expand_dims(np.array(weights, dtype=np.float32), -1)
        weights /= weights.mean()
        p_dummy = np.empty((len(variance), 7), dtype=np.float32)

        #if len(states) < self.memory_size:
        #    b_idx = np.random.choice(len(states), size=1024)
        #    batch = [states[b_idx], values[b_idx], variance[b_idx], p_dummy[b_idx], weights[b_idx]]

        #    sample_loss = self.model.compute_loss(batch)[0]

        #    if sample_loss < loss_threshold:
        #        sys.stderr.write('Low sample loss ({:.2f} < {:.2f}), collecting more data before training ({} / {}).\n'.format(sample_loss, loss_threshold, len(states), self.memory_size))
        #        sys.stderr.flush()
        #        return None
        #    else:
        #        sys.stderr.write('High sample loss ({:.2f} > {:.2f}), starting premature training.\n'.format(sample_loss, loss_threshold))
        #        sys.stderr.flush()
        #else:
        #    sys.stderr.write('Enough training data ({} > {}), proceed to training.\n'.format(len(states), self.memory_size))
        #    sys.stderr.flush()

        m_size = min(self.n_trains * 15000, self.memory_size)
        if len(states) > m_size:
            sys.stderr.write('Enough training data ({} > {}), proceed to training.\n'.format(len(states), m_size))
            sys.stderr.flush()
        else:
            sys.stderr.write('Not enough training data ({} < {}), collecting more data.\n'.format(len(states), m_size))
            sys.stderr.flush()
            return None

        self.n_trains += 1

        self.model.training = True
        #self.model.ewc_lambda = 1.0 / len(states)
        batch = [states, values, variance, p_dummy, weights]
        val_size = int(len(states) * val_fraction)
        batch_val = [states[-val_size:], values[-val_size:], variance[-val_size:], p_dummy[-val_size:], weights[-val_size:]]

        sys.stderr.write('Training data size: {}    Validation data size: {}\n'.format(len(states)-val_size, val_size))
        sys.stderr.flush()

        iters = 0
        l_val_min = float('inf')
        fails = 0
        while fails < patience:
            l_avg = 0
            for it in range(iters_per_val):
                b_idx = np.random.choice(len(states) - val_size, size=batch_size)
                batch = [states[b_idx], values[b_idx], variance[b_idx], p_dummy[b_idx], weights[b_idx]]              
                loss = self.model.train(batch) 
                l_avg += loss[0]
            iters += iters_per_val
            l_avg /= iters_per_val
            l_val = self.model.compute_loss(batch_val)[0]
            #self.model.update_scheduler()
            if l_val < l_val_min:
                self.model.save(verbose=False)
                l_val_min = l_val
                fails = 0
            else:
                fails += 1
            sys.stderr.write('Iteration:{:6d}  training loss:{:.3f} validation loss:{:.3f}\n'.format(iters, l_avg, l_val))
            sys.stderr.flush()
        #self.model.compute_fisher([states, values, variance, p_dummy, weights])
        #self.model.get_fisher_from_adam()
        #self.model.save(verbose=False)
        #self.model.p0 = [p.clone() for p in self.model.model.parameters()]
        self.model.load()
        self.model.update_scheduler()
        self.model.training = False

        self.memory = [[], [], [], []]

        sys.stderr.write('Training complete.\n')
        sys.stderr.flush()
