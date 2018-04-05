import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import sys
from torch.autograd import Variable

IMG_H, IMG_W, IMG_C = (22,10,1)

EXP_PATH = './pytorch_model/'

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        kernel_size = 4
        filters = 32
        self.conv1 = nn.Conv2d(1,filters,kernel_size)
        self.conv2 = nn.Conv2d(filters,filters,kernel_size)

        self.flat_in = ( IMG_H - ( kernel_size - 1 ) * 2 ) * ( IMG_W - ( kernel_size - 1 ) * 2 ) * filters
        self.flat_out = 128
        self.fc1 = nn.Linear(self.flat_in,self.flat_out)

        self.fc_p = nn.Linear(self.flat_out,6)
        self.fc_v = nn.Linear(self.flat_out,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.flat_in)
        x = F.relu(self.fc1(x))

        policy = F.softmax(self.fc_p(x),dim=1)
        value = self.fc_v(x)

        return value, policy

def convert(x):
    return Variable(torch.from_numpy(x.astype(np.float32)))

class Model:
    def __init__(self,new=True):

        self.model = Net()

        self.optimizer = optim.SGD(self.model.parameters(),lr=0.1,momentum=0.9)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=10,verbose=True)
        #self.optim = optim.Adam(self.model.parameters(),amsgrad=True)

    def _loss(self,batch):

        state = convert(batch[0])
        value = convert(batch[1])
        policy = convert(batch[2])

        _v, _p = self.model(state)

        #loss_v = F.mse_loss(_v,value)
        loss_v = Variable(torch.FloatTensor([0]))
        loss_p = F.kl_div(torch.log(_p),policy)

        loss = loss_v + loss_p
        return loss, loss_v, loss_p

    def compute_loss(self,batch):

        self.model.eval()

        loss, loss_v, loss_p = self._loss(batch)
        
        return loss.data.numpy(), loss_v.data.numpy(), loss_p.data.numpy()

    def train(self,batch):

        self.model.train()

        self.optimizer.zero_grad()

        loss, loss_v, loss_p = self._loss(batch)

        loss.backward()

        self.optimizer.step()

        return loss.data.numpy(), loss_v.data.numpy(), loss_p.data.numpy()

    def inference(self,batch):

        self.model.eval()

        state = convert(batch)

        output = self.model(state)

        return output[0].data.numpy(), output[1].data.numpy()

    def update_scheduler(self,val_loss):

        self.scheduler.step(val_loss)

    def save(self):

        sys.stdout.write('Saving model...\n')
        sys.stdout.flush()

        if not os.path.isdir(EXP_PATH):
            sys.stdout.write('Export path does not exist, creating a new one...\n')
            sys.stdout.flush()
            os.mkdir(EXP_PATH)

        filename = EXP_PATH + 'model_checkpoint'

        full_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

        torch.save(full_state, filename)

    def load(self):

        filename = EXP_PATH + 'model_checkpoint'

        if os.path.isfile(filename):
            sys.stdout.write('Loading model...\n')
            sys.stdout.flush()
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            sys.stdout.write('Checkpoint not found, using default model\n')
            sys.stdout.flush()
