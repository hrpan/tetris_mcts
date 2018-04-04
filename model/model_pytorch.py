import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.autograd import Variable

IMG_H, IMG_W, IMG_C = (22,10,1)

EXP_PATH = './pytorch_model/'

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        kernel_size = 3

        self.conv1 = nn.Conv2d(1,16,kernel_size)
        self.conv2 = nn.Conv2d(16,16,kernel_size)

        self.flat_in = ( IMG_H - ( kernel_size - 1 ) * 2 ) * ( IMG_W - ( kernel_size - 1 ) * 2 ) * 16
        
        self.fc1 = nn.Linear(self.flat_in,128)

        self.fc_p = nn.Linear(128,6)
        self.fc_v = nn.Linear(128,1)

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

        #self.optim = optim.SGD(self.model.parameters(),lr=0.1,momentum=0.9)
        self.optim = optim.Adam(self.model.parameters())

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

        self.optim.zero_grad()

        loss, loss_v, loss_p = self._loss(batch)

        loss.backward()

        self.optim.step()

        return loss.data.numpy(), loss_v.data.numpy(), loss_p.data.numpy()

    def inference(self,batch):

        self.model.eval()

        state = convert(batch)

        output = self.model(state)

        return output[0].data.numpy(), output[1].data.numpy()

    def save(self):

        print('Saving model...')

        filename = EXP_PATH + 'model_checkpoint'

        full_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                }

        torch.save(full_state, filename)

    def load(self):

        filename = EXP_PATH + 'model_checkpoint'

        if os.path.isfile(filename):
            print('Loading model...')
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
        else:
            print('Checkpoint not found, using default model')
