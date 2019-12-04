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


class heatequation1D():
	data = None
	def __init__(self, L=0.1, hsi=set([0,2]), T0=[40,0,30], room_temp=3, dx=0.01, alpha=0.0001, t_final=50, dt=0.1):
		"""
        Construct a new 'headequation1D' object.

        :param L: Length of rod
        :param hsi: the indexes of the heat sources on the rod.
        :param T0: Initial temperatures.
        :param room_temp: Room temperature.
        :param dx: Distance between two discrete points on rod.
        :param alpha: Temperature transfer constant of rod (Based on material)
        :param t_final: Final time
        :param dt: Difference between two discrete times when temperature is measured.
        :return: returns nothing
        """
		self.L = L
		self.hsi = hsi
		self.T0 = T0
		self.room_temp = 0
		self.dx = dx
		self.alpha = alpha
		self.t_final = t_final
		self.dt = dt
		self.n = len(T0)


	def singleDimDelta(self,t0,t1,t2):
		return self.alpha*(-(t1-t0)/self.dx**2 + (t2 - t1)/self.dx**2)

	def generate_data(self):
		ans = []
		self.x = np.linspace(self.dx/2, self.L-self.dx/2, self.n)
		dTdt = np.empty(self.n)
		t = np.arange(0, self.t_final, self.dt)
		T = np.asarray(self.T0)
		ans.append(T)
		for j in range(1, len(t)):
			for i in range(0,self.n):
				# If rod's position is heat-source, temperature is constant.
				if (i in self.hsi):
					dTdt[i] = 0
				# If rod's position isn't heat-source, temperature will change.
				else:
					dTdt[i] = self.singleDimDelta(T[i-1], T[i], T[i+1])
			# If left bound of rod isn't heat source, use room temperature as left bound.
			if 0 not in self.hsi:
				dTdt[0] = self.singleDimDelta(self.room_temp, T[0], T[1])
			# If right bound of rod isn't heat source, use room temperature as right bound.
			if self.n-1 not in self.hsi:
				dTdt[self.n-1] = self.singleDimDelta(T[self.n-2], T[self.n-1], self.room_temp)
			T = T + dTdt*self.dt
			ans.append(T)
		self.data = np.array(ans)
		return self.data

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


