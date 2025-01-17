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
import matplotlib.pyplot as plt


if __name__ != "__main__":
    from .torchdiffeq import odeint_adjoint
    from .torchdiffeq import odeint
else:
    from torchdiffeq import odeint_adjoint
    from torchdiffeq import odeint

def train(model, train_loader, optimizer, ode_propogator=odeint,
      niters = 1000,
      test_freq = 20,
      ):
    """
    Given a model (function that takes in arguments (t, y_n) and returns dy_n\dt),
    train the weights to compute arbitray derivative
    """
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


class TruthSampler():
  def __init__(self, t, dataset, batch_time, batch_size):
    """ Sample dataset train method provided in teamtools

    dataset - torch.Tensor shape = (# of samples, length of a sample, *(shape of single datapoint))
    batch_time - how long should the evolution be for a training sample in the batch
    batch_size - how many evolutions go in a single batch?
    """
    self.dataset = dataset
    sz = dataset.size()
    self.sample_count, self.sim_size, self.dp_size = sz[0], sz[1], sz[2:]
    if len(self.dp_size) == 1:
      self.dp_size = tuple([1] + list(self.dp_size))
    self.t = t
    self.batch_time = batch_time
    self.batch_size = batch_size

  def sample(self):
    start_ind = np.random.choice(
        np.arange(self.sim_size - self.batch_time, dtype=np.int64),
        self.batch_size,
        )
    sample_ind = np.random.choice(
        np.arange(self.sample_count, dtype=np.int64),
        self.batch_size
        )
    ret = []
    for sample_index, start_index in zip(sample_ind, start_ind):
      ret.append(self.dataset[sample_index, start_index: start_index + self.batch_time])
    batch_y = torch.stack(ret).transpose(0, 1)
    batch_y0 = batch_y[0].reshape(self.batch_size, *self.dp_size)
    batch_t = self.t[:self.batch_time]
    return batch_y0, batch_t, batch_y


def visualize_2d(t, true_y0, true_y, model):
    """Given ODE initial conditions for a 2D time series vector,
    the real output and the model we are visualizing, return a matplotlib figure visualizing the model"""
    with torch.no_grad():
        pred_y = odeint(model, true_y0, t).detach()
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-', label="True")
    ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--', label="Pred")
    ax_traj.set_xlim(t.min(), t.max())
    ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
    ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    dydt = model(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)
    fig.tight_layout()
    return fig


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




