import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math
from .torchdiffeq import odeint
from .torchdiffeq import odeint_adjoint
from .torchdiffeq import odeint

class Spiral(nn.Module):
  transform_matrix = torch.tensor([[-0.1, 1.0], [-2.0, -0.1]])

  def forward(self, t, y):
    return torch.mm(y, self.transform_matrix)


class SimpleNetODE(nn.Module):
    def __init__(self, net):
        super(ODEFunc, self).__init__()
        self.net = net

    def forward(self, t, y):
        return self.net(y)



