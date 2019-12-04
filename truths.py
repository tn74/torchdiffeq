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


def heat_equation_pde_one_dim(L, n, T0, left_bound, right_bound, dx, alpha, t_final, dt):
  ans = []
  x = np.linspace(dx/2, L-dx/2, n)
  T = np.ones(n) * T0
  ans.append(T)
  dTdt = np.empty(n)
  t = np.arange(0, t_final, dt)
  for j in range(1, len(t)):
    for i in range(1,n-1):
      dTdt[i] = alpha*(-(T[i] - T[i-1])/dx**2  + (T[i+1]-T[i])/dx**2)
      dTdt[0] = alpha*(-(T[0] - left_bound)/dx**2  + (T[1]-T[0])/dx**2)
      dTdt[n-1] = alpha*(-(T[n-1] - T[n-2])/dx**2  + (right_bound-T[n-1])/dx**2)
    T = T + dTdt*dt
    ans.append(T)
  a = np.concatenate(
      [left_bound * np.ones((len(ans), 1)), np.array(ans),
       right_bound * np.ones((len(ans), 1))], axis=1)
  ret = torch.from_numpy(a)
  ret = ret.reshape(1, len(t), 1, len(x) + 2).type(torch.FloatTensor)
  return ret


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


