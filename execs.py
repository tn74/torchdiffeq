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
from .torchdiffeq import odeint
from .torchdiffeq import odeint_adjoint


def train(model, train_loader, optimizer, ode_propogator=odeint,
      niters = 1000,
      test_freq = 20,
      ):
    """
    Given a model (function that takes in arguments (t, y_n) and returns dy_n\dt),
    train the weights to compute arbitray derivative
    """
    losses = []
    ii = 0
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = train_loader()
        pred_y = ode_propogator(model, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                loss = torch.mean(torch.abs(pred_y - batch_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1
    return losses
