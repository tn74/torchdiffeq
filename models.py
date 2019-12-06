import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math
from .torchdiffeq import odeint
from .torchdiffeq import odeint_adjoint
import numpy as np

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


class DenseLayers1DimODE(nn.Module):
    def __init__(self, state_shape, layers=[50]):
        super().__init__()
        self.state_shape = state_shape
        self.one_d_length = self.state_shape[1]
        self.in_size = np.prod(state_shape)
        layers = [self.in_size] + layers + [self.in_size]
        layers = [nn.Linear(a, b) for a, b in zip(layers[:-1], layers[1:])]
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        yshape = y.size()
        y = y.reshape(yshape[0], -1)
        y = self.net(y)
        y = y.reshape(*yshape)
        return y

class ConvLayers1DimODE(nn.Module):
    def __init__(self, state_shape, kernels=2):
        super().__init__()
        self.state_shape = state_shape
        self.one_d_length = self.state_shape[1]
        self.total_inputs = np.prod(state_shape)
        self.conv = nn.Conv2d(1, kernels, 3, padding=1) # 1 input plane, 2 output planes, 3 size kernel, 1 size padding to keep the output planes same size as input
        testvec = torch.randn(size=state_shape).reshape(1, 1, *state_shape)
        conv_out  = np.prod(self.conv(testvec).size())
        self.net = nn.Sequential(
            nn.Linear(conv_out, 2 * self.total_inputs),
            nn.Linear(2 * self.total_inputs, self.total_inputs),
        )

    def forward(self, t, y):
        yshape = y.size()
        y = y.reshape(*([yshape[0]] + [1] + list(yshape[1:])))
        y = self.conv(y)
        y = y.reshape(yshape[0], -1)
        y = self.net(y)
        y = y.reshape(*yshape)
        return y

