import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.onnx
import torch.utils.data as D
#import onnx
import os, sys, subprocess
import numpy as np
from caffe2.python import workspace
from model.bbb import BBB
IMG_H, IMG_W, IMG_C = (22,10,1)

EXP_PATH = './pytorch_model/'

n_actions = 7


def convOutShape(shape_in, kernel_size, stride):

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    return ((shape_in[0] - kernel_size[0]) // stride[0] + 1, (shape_in[1] - kernel_size[1]) // stride[1] + 1 )

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        
        kernel_size = 3
        stride = 1
        filters = 32

        #self.conv1 = nn.Conv2d(1, filters, kernel_size, stride)
        self.conv1 = nn.Conv2d(1, filters, kernel_size=4, stride=(2, 1))
        self.bn1 = nn.BatchNorm2d(filters)
        #self.gn1 = nn.GroupNorm(4, filters, affine=False)
        #_shape = convOutShape((22,10), kernel_size, stride)
        _shape = convOutShape((22,10), 4, stride=(2, 1))

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        #self.gn2 = nn.GroupNorm(4, filters, affine=False)
        _shape = convOutShape(_shape, kernel_size, stride)

        flat_in = _shape[0] * _shape[1] * filters
 
        n_fc1 = 128
        self.fc1 = nn.Linear(flat_in, n_fc1)
        flat_out = n_fc1
        #flat_out = flat_in
        
        #self.fc_p = nn.Linear(flat_out, n_actions)
        self.fc_v = nn.Linear(flat_out, 1)
        self.fc_var = nn.Linear(flat_out, 1)
        torch.nn.init.normal_(self.fc_v.bias, mean=1000, std=.1)
        torch.nn.init.normal_(self.fc_var.bias, mean=6, std=.1)

    def forward(self, x):
        #x = self.bn1(F.relu(self.conv1(x)))
        #x = self.bn2(F.relu(self.conv2(x)))
        #x = self.gn1(F.relu(self.conv1(x)))
        #x = self.gn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))

        #policy = F.softmax(self.fc_p(x), dim=1)
        policy = torch.ones((x.shape[0], 7)) / 7
        #value = torch.exp(self.fc_v(x))
        value = self.fc_v(x)
        var = torch.exp(self.fc_var(x)) + 1
        #var = F.softplus(self.fc_var(x)) + 1
        return value, var, policy

def convert(x):
    return torch.from_numpy(x.astype(np.float32))

class Dataset(D.Dataset):
    def __init__(self, data):
        #states, values, variance, policy, weights
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return [d[index] for d in self.data]  


class Model:
    def __init__(self, training=False, new=True, weighted_mse=False, mle_gauss=True, use_variance=True, use_policy=False, loss_type='mae', use_onnx=False, use_cuda=True, mc_iters=50):

        self.use_cuda = use_cuda

        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


        self.training = training

        self.model = Net()
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()       
        self.mc_iters = mc_iters
        #reparam = [{'params': params} for name, params in self.model.named_parameters()]
        self.optimizer = BBB(self.model.parameters(), lr=1e-2, rho_init=-3)
        self.scheduler = None

        self.v_mean = 0
        self.v_std = 1

        self.var_mean = 0
        self.var_std = 1

        self.weighted_mse = weighted_mse

        self.mle_gauss = mle_gauss

        self.use_policy = use_policy
        self.use_variance = use_variance

        if loss_type == 'mae':
            self.l_func = lambda x, y: torch.abs(x-y)
        else:
            self.l_func = lambda x, y: (x - y) ** 2

        self.use_onnx = use_onnx

    def _loss(self, batch):
       
        state, value, variance, policy, weight = batch
        if type(state) is not torch.Tensor:
            state = torch.from_numpy(batch[0])
            value = torch.from_numpy(batch[1])
            variance = torch.from_numpy(batch[2])
            policy = torch.from_numpy(batch[3])
            weight = torch.from_numpy(batch[4])

        if self.use_cuda and torch.cuda.is_available():
            state = state.cuda()
            value = value.cuda()
            variance = variance.cuda()
            policy = policy.cuda()
            weight = weight.cuda()
       
        _v, _var, _p = self.model(state)

        if self.mle_gauss:
            #loss = (weight * (torch.log(_var) + variance / _var + ((value - _v) ** 2 ) / _var - 1 - torch.log(variance))).mean()
            loss = (weight * (torch.log(_var) + variance / _var + ((value - _v) ** 2 ) / variance - 1 - torch.log(variance))).mean()
            return loss, torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0])
        else:
            if self.weighted_mse:
                loss_v = (weight * self.l_func(_v, value)).mean()
                if self.use_variance:
                    loss_var = (weight * self.l_func(_var, variance)).mean()
                else:
                    loss_var = torch.FloatTensor([0])
            else:
                loss_v = self.l_func(_v, value).mean()
                if self.use_variance:
                    loss_var = self.l_func(_var, variance).mean()
                else:
                    loss_var = torch.FloatTensor([0])

            if self.use_policy:
                loss_p = F.kl_div(torch.log(_p),policy)
            else:
                loss_p = torch.FloatTensor([0])

            loss = loss_v + loss_var + loss_p
            return loss, loss_v, loss_var, loss_p

    def compute_loss(self, batch):

        self.model.eval()

        with torch.no_grad():
            losses = self._loss(batch)
        
        result = [l.item() for l in losses]

        result.append(0)

        return result

    def train(self, batch):

        self.model.train()

        opt = self.optimizer
        result = [0, ]
        for mc_iter in range(self.mc_iters):
            opt.set_weights()
            losses = self._loss(batch)
            opt.zero_grad()
            losses[0].backward()
            opt.aggregate_grads(batch_size=len(batch[0]))
            result[0] += losses[0].item()
        opt.step()
        #print(self.model.conv1.weight)
        result[0] /= self.mc_iters
        return result
  
    def inference(self,batch):

        self.model.eval() 

        b = torch.from_numpy(batch).float()
        if self.use_cuda and torch.cuda.is_available():
            b = b.cuda()

        with torch.no_grad():
            output = self.model(b)

        result = [o.cpu().numpy() for o in output]
        #result = self.model_caffe.run([batch.astype(np.float32)])

        return result

    def inference_stochastic(self,batch):

        self.model.eval()

        #state = convert(batch)
        b = torch.from_numpy(batch).float()
        if self.use_cuda and torch.cuda.is_available():
            b = b.cuda()

        with torch.no_grad():
            output = self.model(b)

        result = [o.cpu().numpy() for o in output]
        v = np.random.normal(result[0], np.sqrt(result[1]))
        result[0] = v
        return result


    def update_scheduler(self, **kwarg):

        if self.scheduler:
            self.scheduler.step(**kwarg)

    def save(self, verbose=True):

        if verbose:
            sys.stdout.write('Saving model...\n')
            sys.stdout.flush()

        if not os.path.isdir(EXP_PATH):
            if verbose:
                sys.stdout.write('Export path does not exist, creating a new one...\n')
                sys.stdout.flush()
            os.mkdir(EXP_PATH)

        filename = EXP_PATH + 'model_checkpoint'

        full_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'v_mean': self.v_mean,
                    'v_std': self.v_std,
                    'var_mean': self.var_mean,
                    'var_std': self.var_std,
                }

        if self.scheduler:
            full_state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(full_state, filename)

        if self.use_onnx:
            self.save_onnx()

    def save_onnx(self, verbose=False):

        dummy = torch.ones((1, 1, 22, 10))
        
        if not os.path.isdir(EXP_PATH):
            if verbose:
                sys.stdout.write('Export path does not exist, creating a new one...\n')
                sys.stdout.flush()
            os.mkdir(EXP_PATH)

        torch.onnx.export(self.model, dummy, EXP_PATH + 'model.onnx')

        output_arg = '-o ' + EXP_PATH + 'pred.pb'
        init_arg = '--init-net-output ' + EXP_PATH + 'init.pb'
        file_arg = EXP_PATH + 'model.onnx'

        subprocess.run('convert-onnx-to-caffe2' + ' ' + output_arg + ' ' + init_arg + ' ' + file_arg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def load(self, filename=EXP_PATH + 'model_checkpoint'):

        #filename = EXP_PATH + 'model_checkpoint'

        if os.path.isfile(filename):
            sys.stdout.write('Loading model...\n')
            sys.stdout.flush()
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.v_mean = checkpoint['v_mean'] 
            self.v_std = checkpoint['v_std'] 
            self.var_mean = checkpoint['var_mean'] 
            self.var_std = checkpoint['var_std']
            self.p0 = [p.clone() for p in self.model.parameters()]
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            sys.stdout.write('Checkpoint not found, using default model\n')
            sys.stdout.flush()

        if not self.training and self.use_onnx:
            self.load_onnx()

    def load_onnx(self):
        
        init_filename = EXP_PATH + 'init.pb'
        pred_filename = EXP_PATH + 'pred.pb'

        sys.stdout.write('Loading ONNX model...\n')
        sys.stdout.flush()

        if not (os.path.isfile(init_filename) or os.path.isfile(pred_filename)):
            self.save_onnx()

        with open(init_filename, mode='r+b') as f:
            init_net = f.read()

        with open(pred_filename, mode='r+b') as f:
            pred_net = f.read()

        self.model_caffe = workspace.Predictor(init_net, pred_net)

