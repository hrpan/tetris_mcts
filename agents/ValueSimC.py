import agents.helper
from agents.cppmodule.agent import OnlineMCTSAgent
from model.model_vv import Model_VV as Model
import numpy as np

def training(state, value, variance, visit, d_size, model):

    np.savez('./data/dump', states=state[:d_size], values=value[:d_size], variance=variance[:d_size], weights=visit[:d_size])
    model.train_data([state[:d_size], value[:d_size], 
                      variance[:d_size], visit[:d_size]],
                     iters_per_val=100, batch_size=512, max_iters=50000,
                     sample_replacement=False)
    model.training(False)

class ValueSimC(OnlineMCTSAgent):

    def __init__(
            self, sims=100, max_nodes=100000, online=False, memory_size=500000, memory_growth_rate=5000,
            min_visit=15, projection=True, gamma=0.99, benchmark=False, leaf_parallel=True, **kwargs):

        self.model = Model()
        self.model.load()
        self.model.training(False)
        
        inference = self.model.inference

        super().__init__(
            sims, max_nodes, online, memory_size, memory_growth_rate, min_visit,
            projection, gamma, benchmark, inference, 0, 
            lambda state, val, var, vis, size: training(state, val, var, vis, size, self.model), leaf_parallel)

    def close(self):
        pass
