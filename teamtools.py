import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math

if __name__ != "__main__":
    from .torchdiffeq import odeint_adjoint
    from .torchdiffeq import odeint
else:
    from torchdiffeq import odeint_adjoint
    from torchdiffeq import odeint

def train(model, train_loader, optimizer, ode_propogator=odeint,
          niters = 1000,
          test_freq = 20,
          gpu = 0):
    ii = 0
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = train_loader()
        pred_y = ode_propogator(model, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                loss = torch.mean(torch.abs(pred_y - batch_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

if __name__ == "__main__":
    class Config:
      data_size = 1000
      batch_time = 10
      batch_size = 20
      niters = 1000
      test_freq = 20
      viz = True
      gpu = 0
      adjoint = 0
    args = Config()

    class Spiral(nn.Module):
      angle = 45
      # transform_matrix = 0.9 * torch.Tensor([
      #     [math.cos(angle * math.pi/180),   -math.sin(angle * math.pi/180)],
      #     [math.cos(angle * math.pi/180),    math.cos(angle * math.pi/180)]
      # ])
      transform_matrix = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])

      def forward(self, t, y):
        return torch.mm(y, self.transform_matrix)

    t = torch.linspace(0., 5., args.data_size)
    true_y0 = torch.tensor([[2., 0.]])
    with torch.no_grad():
        true_y = odeint(Spiral(), true_y0, t, method='dopri5')

    def get_batch():
        s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
        batch_y0 = true_y[s]  # (M, D)
        batch_t = t[:args.batch_time]  # (T)
        batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
        return batch_y0, batch_t, batch_y

    class ODEFunc(nn.Module):
        def __init__(self):
            super(ODEFunc, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 50),
                nn.Tanh(),
                nn.Linear(50, 2),
                # nn.Linear(2, 2)
            )

            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)

        def forward(self, t, y):
            return self.net(y)
    model = ODEFunc()
    train_loader = get_batch
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, optimizer)




