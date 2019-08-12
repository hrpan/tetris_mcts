import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.onnx
import onnx
import os, sys, subprocess
import numpy as np
from caffe2.python import workspace

IMG_H, IMG_W, IMG_C = (22,10,1)

EXP_PATH = './pytorch_model/'

n_actions = 7

def convOutShape(shape_in,kernel_size,stride):
    return ((shape_in[0] - kernel_size) // stride + 1, (shape_in[1] - kernel_size) // stride + 1 )

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        
        kernel_size = 3
        stride = 1
        filters = 32

        self.conv1 = nn.Conv2d(1, filters, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        _shape = convOutShape((22,10), kernel_size, stride)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        _shape = convOutShape(_shape, kernel_size, stride)

        flat_in = _shape[0] * _shape[1] * filters
 
        n_fc1 = 128
        self.fc1 = nn.Linear(flat_in, n_fc1)
        flat_out = n_fc1
        #flat_out = flat_in

        
        self.fc_p = nn.Linear(flat_out, n_actions)
        self.fc_v = nn.Linear(flat_out, 1)
        self.fc_var = nn.Linear(flat_out, 1)
        torch.nn.init.normal_(self.fc_var.weight, mean=0., std=.1)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))

        #policy = F.softmax(self.fc_p(x), dim=1)
        policy = torch.ones((x.shape[0], 7)) / 7
        value = self.fc_v(x)
#        var = F.softplus(self.fc_var(x))
        var = F.elu(self.fc_var(x)) + 1
        return value, var, policy

def convert(x):
    return torch.from_numpy(x.astype(np.float32))

class Model:
    def __init__(self, training=False, new=True, weighted_mse=False, ewc=False, ewc_lambda=1, mle_gauss=True, use_variance=True, use_policy=False, loss_type='mae', use_onnx=False, use_cuda=True):

        self.use_cuda = use_cuda

        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


        self.training = training

        self.model = Net()
        if self.use_cuda:
            self.model = self.model.cuda()        
        #self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-3, amsgrad=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.scheduler = None

        self.v_mean = 0
        self.v_std = 1

        self.var_mean = 0
        self.var_std = 1

        self.weighted_mse = weighted_mse

        self.ewc = ewc
        self.ewc_lambda = ewc_lambda

        self.mle_gauss = mle_gauss

        self.fisher = None

        self.use_policy = use_policy
        self.use_variance = use_variance

        if loss_type == 'mae':
            self.l_func = lambda x, y: torch.abs(x-y)
        else:
            self.l_func = lambda x, y: (x - y) ** 2

        self.use_onnx = use_onnx

    def _loss(self, batch):

        state = torch.from_numpy(batch[0])
        value = torch.from_numpy(batch[1])
        variance = torch.from_numpy(batch[2])
        policy = torch.from_numpy(batch[3])
        weight = torch.from_numpy(batch[4])

        if self.use_cuda:
            state = state.cuda()
            value = value.cuda()
            variance = variance.cuda()
            policy = policy.cuda()
            weight = weight.cuda()

        _v, _var, _p = self.model(state)
        if self.mle_gauss:
            loss = (weight * (torch.log(_var) + variance / _var + ((value - _v) ** 2 ) / _var - 1 - torch.log(variance))).mean()
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

    def compute_ewc_loss(self):

        if self.fisher:
            ewc_loss = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
            for i, p in enumerate(self.model.parameters()):
                ewc_loss = ewc_loss + 0.5 * self.ewc_lambda * torch.sum(self.fisher[i] * (p - self.p0[i]) ** 2)
            return ewc_loss
        else:
            return torch.tensor([0.], dtype=torch.float32, requires_grad=True)

    def compute_loss(self, batch):

        self.model.eval()

        losses = self._loss(batch)
        
        result = [l.data.numpy() for l in losses]

        if self.ewc:
            result.append(self.compute_ewc_loss().data.numpy())
        else:
            result.append(0)

        return result

    def train(self, batch):

        self.model.train()

        self.optimizer.zero_grad()

        losses = self._loss(batch)

        if self.ewc:
            ewc_loss = self.compute_ewc_loss()
            l = losses[0] + ewc_loss
        else:
            l = losses[0]

        l.backward()

        #U.clip_grad_norm_(self.model.parameters(), 1e1)
      
        self.optimizer.step()

        result = [l.item() for l in losses]

        if self.ewc:
            result.append(ewc_loss.item())
            result[0] += ewc_loss.item()
        else:
            result.append(0)

        return result
    
    def compute_fisher(self, batch):
        
        self.model.train()
        
        fisher = [torch.zeros(p.data.shape) for p in self.model.parameters()]

        for i in range(len(batch[0])):

            self.optimizer.zero_grad()

            loss = self._loss([b[i:i+1] for b in batch]) 

            loss = loss[0] + self.compute_ewc_loss() 

            loss.backward()

            for j, p in enumerate(self.model.parameters()):
                if p.grad is not None:
                    fisher[j] += torch.pow(p.grad.data, 2) / len(batch[0])
            
        self.fisher = fisher
    
    def inference(self,batch):

        self.model.eval() 

        b = torch.from_numpy(batch).float()
        if self.use_cuda:
            b = b.cuda()

        with torch.no_grad():
            output = self.model(b)

        result = [o.cpu().numpy() for o in output]
        #result = self.model_caffe.run([batch.astype(np.float32)])

        return result

    def inference_stochastic(self,batch):

        self.model.eval()

        state = convert(batch)
        
        with torch.no_grad():
            output = self.model(state)

        result = [o.data.numpy() for o in output]
        v = np.random.normal(output[0], np.sqrt(output[1]))
        result[0] = v
        return result


    def update_scheduler(self,val_loss):

        if self.scheduler:
            self.scheduler.step(val_loss)

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
                    'fisher': self.fisher,
                }
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

    def load(self):

        filename = EXP_PATH + 'model_checkpoint'

        if os.path.isfile(filename):
            sys.stdout.write('Loading model...\n')
            sys.stdout.flush()
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.v_mean = checkpoint['v_mean'] 
            self.v_std = checkpoint['v_std'] 
            self.var_mean = checkpoint['var_mean'] 
            self.var_std = checkpoint['var_std']
            self.fisher = checkpoint['fisher'] 
            self.p0 = [p.clone() for p in self.model.parameters()]
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

