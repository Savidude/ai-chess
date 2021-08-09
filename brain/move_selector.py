import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import constants


class mCNN(nn.Module):

    def __init__(self):
        super(mCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3)

        self.fc1 = nn.Linear(in_features=self.count_CNN_output_neurons(constants.BOARD_DIMENSIONS), out_features=128)
        self.out = nn.Linear(in_features=128, out_features=constants.MOVE_SELECTOR_NUM_ACTIONS)

    def count_CNN_output_neurons(self, board_dim):
        x = Variable(torch.rand(1, *board_dim))  # create a fake board with the given dimensions
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


class MoveSelector:
    def __init__(self, device):
        self.network = mCNN().to(device=device)
        self.loss_func = nn.MSELoss()
