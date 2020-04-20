import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pylab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import simple_model as model
import rootfind_model

import lyapunov_NN as L
import visualize_trajectory as vis

# from rootfind_autograd import rootfind_module

layer_sizes = np.array([2, 100, 1])
#
fhat = model.fhat(np.array([2, 50, 50, 2]))
# fhat = nn.Sequential(nn.Linear(2, 50), nn.Tanh(),
#                     nn.Linear(50, 50), nn.Tanh(),
#                     nn.Linear(50, 50), nn.Tanh(),
#                     nn.Linear(50, 2))
#
V = L.MakePSD(L.ICNN(layer_sizes),2)

# PATH_V = './saved_models/simple_test_V.pth'
# PATH_f = './saved_models/simple_test_f.pth'
#
# V = torch.load(PATH_V)
# fhat = torch.load(PATH_f)
# f = model.dynamics_simple(fhat,V)
f = model.dynamics_nonincrease(fhat,V)


# f_net = Net()
# net.load_state_dict(torch.load(PATH))


# f = model.dynamics_rootfind(fhat,V)

# f = rootfind_model.rootfind_module(fhat,V)

# f = model.dynamics_stochastic(fhat,V)
# x0 = 2.5*torch.randn([1,2], dtype = torch.float)
x0 = torch.tensor([[[1,1]]], dtype = torch.float)

plotting = vis.plot_dynamics(f,V,x0)

for i in range(1):

    plotting.plot_trajectory()

plt.show()
