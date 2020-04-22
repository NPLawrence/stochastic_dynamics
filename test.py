#just a sanity check

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

# from rootfind_autograd import rootfind_module

from myrelu import MyReLU

import simple_model as model
import generate_data as gen_datat

torch.set_grad_enabled(True)

# relu = MyReLU.apply
fhat = nn.Sequential(nn.Linear(2, 50), nn.Tanh(),
                    nn.Linear(50, 50), nn.Tanh(),
                    nn.Linear(50, 50), MyReLU(),
                    nn.Linear(50, 2))

layer_sizes = np.array([2, 100, 1])
V = model.MakePSD(model.ICNN(layer_sizes),2)


input = torch.randn(1,2)
W = V(fhat(input))
# output = V(input)
W.backward(torch.ones_like(W))


# for name, weight in V.named_parameters():
#     print(name, weight.grad)
#
# for name, weight in fhat.named_parameters():
#     print(name, weight.grad)
