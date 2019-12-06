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
import random


class heatequation2D():
    data = None
    def __init__(self, w=10, h=10, dx=0.1, dy=0.1, D=4):
        """
        Construct a new 'headequation1D' object.

        :param w: plate width
        :param h: plate height
        :param dx: intervals in x direction
        :param dy: intervals in y direction.
        :param D: thermal diffusivity of material
        :return: returns nothing
        """
        # plate size, mm
        self.D = D
        self.dx = dx
        self.dy = dy
        self.nx = int(w/dx)
        self.ny = int(w/dy)
        self.dx2 = dx*dx
        self.dy2 = dy*dy
        self.dt = self.dx2 * self.dy2 / (2 * D * (self.dx2 + self.dy2))
        self.u0 = np.zeros((self.nx, self.ny,2))
        self.u = np.zeros((self.nx, self.ny, 2))

    def initializeRing(self, r=2,cx=2,cy=2):
        r2 = r**2
        for i in range(self.nx):
            for j in range(self.ny):
                p2 = (i*self.dx-cx)**2 + (j*self.dy-cy)**2
                if p2 < r2:
                    self.u0[i,j,0] = 700
        print(np.shape(self.u0))

    def initializeRing_random(self, r=2,cx=2,cy=2):
        r2 = r**2
        for i in range(self.nx):
            for j in range(self.ny):
                p2 = (i*self.dx-cx)**2 + (j*self.dy-cy)**2
                if p2 < r2:
                    self.u0[i,j,0] = float(random.randint(300,1000))
                    # Heat sources are initialized here.
                    self.u0[i,j,1] = 1 if random.randint(1,3) == 1 else 0

    def initializeBox_random(self, r=2,cx=2,cy=2):
        for i in range(self.nx):
            for j in range(self.ny):
                if abs(cx - i) < r and abs(cy - j) < r:
                    self.u0[i,j,0] = float(random.randint(300,1000))
                    # Heat sources are initialized here.
                    self.u0[i,j,1] = 1 if random.randint(1,3) == 1 else 0

    def initializeRandom(self):
        for i in range(self.nx):
            for j in range(self.ny):
                if random.randint(1,3) == 1:
                    self.u0[i,j,0] = float(random.randint(300,1000))
                    # Heat sources are initialized here.
                    self.u0[i,j,1] = 1 if random.randint(1,3) == 1 else 0

    def do_timestep(self,u0, u):
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1, 0] = u0[1:-1, 1:-1, 0] + (1-u0[1:-1, 1:-1, 1])*(self.D * self.dt * (
              (u0[2:, 1:-1, 0] - 2*u0[1:-1, 1:-1, 0] + u0[:-2, 1:-1, 0])/self.dx2
              + (u0[1:-1, 2:, 0] - 2*u0[1:-1, 1:-1, 0] + u0[1:-1, :-2, 0])/self.dy2))
        u0 = u.copy()
        return u0, u

    def generate_data(self, nsteps=10):
        ans = []
        ans.append(self.u0)
        seconds = time.time()
        for m in range(nsteps):
            self.u0, self.u = self.do_timestep(self.u0, self.u)
            ans.append(self.u.copy())
        print("Seconds since epoch =", time.time()-seconds)
        self.data = np.array(ans)
        return self.data

# heatequation2D_inst = heatequation2D(w=10, h=10, dx=0.1, dy=0.1, D=4)
# heatequation2D_inst.initializeRing_random(r=2,cx=5,cy=5);
# data = heatequation2D_inst.generate_data(nsteps=1000);

def generate_truth_sampler_he2D(size = 10, num_init_states=5, dx = 0.1, D=4):
    ans = []
    for i in range(num_init_states):
        print(i)
        # Instantiate heat equation model.
        heatequation2D_inst = heatequation2D(w=size, h=size, dx=dx, dy=dx, D=4)

        # Setup initial State
        curr_init_state = random.randint(1,3)

        # Initialize Rings
        if (curr_init_state == 1):
            numRings = random.randint(1,3)
            for i in range(numRings):
                heatequation2D_inst.initializeRing_random(r=random.randint(1,4),cx=random.randint(4,6),cy=random.randint(4,6))

        # Initialize Boxes
        if (curr_init_state == 2):
            numBoxes = random.randint(1,3)
            for i in range(numBoxes):
                heatequation2D_inst.initializeBox_random(r=random.randint(1,4),cx=random.randint(4,6),cy=random.randint(4,6))

        # Initialize Random
        if (curr_init_state == 3):
            heatequation2D_inst.initializeRandom()

        # Generate data and append to list
        data = heatequation2D_inst.generate_data(nsteps=1000)
        ans.append(data)

    # Stack list
    res = np.stack(ans)

    return res


class heatequation1D():
	data = None
	def __init__(self, L=0.1, T0=[40,0,30], dx=0.01, alpha=0.0001, t_final=50, dt=0.1):
		"""
        Construct a new 'headequation1D' object.

        :param L: Length of rod
        :param T0: Initial temperatures.
        :param dx: Distance between two discrete points on rod.
        :param alpha: Temperature transfer constant of rod (Based on material)
        :param t_final: Final time
        :param dt: Difference between two discrete times when temperature is measured.
        :return: returns nothing
        """
		self.L = L
		self.T0 = T0
		self.room_temp = 0
		self.dx = dx
		self.alpha = alpha
		self.t_final = t_final
		self.dt = dt
		self.n = len(T0[0])

	def singleDimDelta(self,t0,t1,t2):
		return self.alpha*(-(t1-t0)/self.dx**2 + (t2 - t1)/self.dx**2)

	def generate_data(self):
		ans = []
		self.x = np.linspace(self.dx/2, self.L-self.dx/2, self.n)
		t = np.arange(0, self.t_final, self.dt)
		T = np.asarray(self.T0)
		ans.append(T)
		for j in range(1, len(t)):
			dTdt = np.zeros((2,self.n))
			for i in range(self.n-1):
				# If rod's position is heat-source, temperature is constant.
				if (self.T0[1][i] == 1):
					dTdt[0][i] = 0
				# If rod's position isn't heat-source, temperature will change.
				else:
					dTdt[0][i] = self.singleDimDelta(T[0][i-1], T[0][i], T[0][i+1])
			# If left bound of rod isn't heat source, use room temperature as left bound.
			if self.T0[1][0] == 0:
				dTdt[0][0] = self.singleDimDelta(self.room_temp, T[0][0], T[0][1])
			# If right bound of rod isn't heat source, use room temperature as right bound.
			if self.T0[1][self.n-1] == 0:
				dTdt[0][self.n-1] = self.singleDimDelta(T[0][self.n-2], T[0][self.n-1], self.room_temp)
			T = T + dTdt*self.dt
			ans.append(T)
		self.data = np.array(ans)
		return self.data

# T0 = [[40, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 20], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
# heat_eqn_inst1_2 = heatequation1D(L=0.1, T0=T0, dx=0.01, alpha=0.0001, t_final=100, dt=0.1)
# heat_eqn_inst1_y1_2 = heat_eqn_inst1_2.generate_data()


def generate_truth_sampler_he1D(rod_size=12, num_init_states=100, L=0.1, dx=0.01, alpha=0.0001, t_final=100, dt=0.1):
	ans = []
	for i in range(num_init_states):
		T0 = np.zeros((2,rod_size))
		numHeatSources = random.randint(0,int(rod_size/3)) + 1
		heatSourceIndexes = np.random.choice(np.arange(0,rod_size), numHeatSources, replace=False)
		for index in np.nditer(heatSourceIndexes):
			T0[1][index] = 1
		heat_values = np.random.uniform(low=0, high=273, size=rod_size)
		T0[0] = heat_values
		heatequation1D_obj = heatequation1D(L=L, T0=T0, dx=dx, alpha=alpha, t_final=t_final, dt=dt)
		data = heatequation1D_obj.generate_data()
		ans.append(data)
	res = np.stack(ans)
	return res


def one_dim_heat_equation_pde(L, n, T0, left_bound, right_bound, dx, alpha, t_final, dt):
  ans = []
  x = np.linspace(dx/2, L-dx/2, n)
  T = np.ones(n) * T0
  ans.append(T)
  dTdt = np.empty(n)
  t = np.arange(0, t_final, dt)
  for j in range(1, len(t)):
    plt.clf()
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

def generate_one_dim_heat_dataset(samples=20, L=0.1, n=10, T0=0, dx=0.01, alpha=0.0001, t_final=30, dt=0.1):
    bounds = np.random.uniform(low=0, high=273, size=(samples, 2))
    samples = []
    for low, high in bounds:
        samples.append(one_dim_heat_equation_pde(L, n, T0, low, high, dx, alpha, t_final, dt))
    return torch.cat(samples)

# torch_y_true = heat_equation_pde(L=0.1, n=10, T0=0, left_bound=40, right_bound=20, dx=0.01, alpha=0.0001, t_final=30, dt=0.1)
# print(torch_y_true.size())
# print(torch_y_true)
# extra = torch.ones_like(torch_y_true)
# extra[:, :, :, 0] = 0
# extra[:, :, :, -1] = 0
# dp = torch.cat([torch_y_true, extra], 2)
# dset = torch.cat([dp for i in range(100)], 0)



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


