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

from torchdiffeq.torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq.torchdiffeq import odeint


def train(model, train_loader, ode_propogator,
          data_size = 1000,
          batch_time = 10,
          batch_size = 20,
          niters = 1000,
          test_freq = 20,
          viz = True,
          gpu = 0,
          adjoint = 0):
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    end = time.time()
    ii = 0

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = train_loader()
        pred_y = ode_propogator(model, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1
