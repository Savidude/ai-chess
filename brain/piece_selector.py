import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

from . import constants


def init_weights(layer, magnitude):
    classname = layer.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = layer.in_features
        y = 1.0 / math.sqrt(n)
        layer.weight.data.uniform_(-y, y)
        layer.bias.data.fill_(0)
    else:
        y = 1/(10 ** magnitude)
        layer.weight.data.uniform_(-y, y)
        layer.bias.data.fill_(0)


class pCNN(nn.Module):

    def __init__(self):
        super(pCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        init_weights(self.conv1, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        init_weights(self.conv2, 6)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3)
        init_weights(self.conv3, 6)

        self.fc1 = nn.Linear(in_features=self.count_CNN_output_neurons(constants.BOARD_DIMENSIONS), out_features=128)
        init_weights(self.fc1, 0)
        self.out = nn.Linear(in_features=128, out_features=constants.MOVE_SELECTOR_NUM_ACTIONS)
        init_weights(self.out, 0)

    def count_CNN_output_neurons(self, board_dim):
        if not next(self.parameters()).is_cuda:
            x = Variable(torch.rand(1, *board_dim))  # create a fake board with the given dimensions
        else:
            x = Variable(torch.rand(1, *board_dim).to(device='cuda'))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.data.view(1, -1).size(1)  # Return the size of the flattened layer

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = F.relu(self.conv3(t))

        t = t.reshape(-1, self.count_CNN_output_neurons(constants.BOARD_DIMENSIONS))  # Flatten convolutional layer
        t = F.relu(self.fc1(t))
        t = self.out(t)
        return t
