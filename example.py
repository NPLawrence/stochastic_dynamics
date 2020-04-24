#Plots a random instance of a stable model

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
import dynamics_plotting as vis

# from rootfind_autograd import rootfind_module

layer_sizes = np.array([2, 50, 1])
#
fhat = nn.Sequential(nn.Linear(2, 50), nn.Tanh(),
                    nn.Linear(50, 50), nn.Tanh(),
                    nn.Linear(50, 2))
#
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,2)
PATH_ICNN = './saved_models/rootfind_test_ICNN.pth'
PATH_V = './saved_models/rootfind_test_V.pth'
PATH_f = './saved_models/rootfind_test_f.pth'
# PATH_f = './saved_models/simple_test_fhat.pth'

# PATH_ICNN = './saved_models/simple_test_ICNN.pth'
# PATH_V = './saved_models/simple_test_V.pth'
# PATH_f = './saved_models/simple_test_f.pth'
#
# the_model = TheModelClass(*args, **kwargs)
ICNN.load_state_dict(torch.load(PATH_ICNN))
V.load_state_dict(torch.load(PATH_V))
fhat.load_state_dict(torch.load(PATH_f))
# f = fhat
# f = model.dynamics_simple(fhat,V)
# f = model.dynamics_nonincrease(fhat,V)
# f = model.dynamics_rootfind(fhat,V)
f = rootfind_model.rootfind_module(fhat,V)
# f = model.dynamics_stochastic(fhat,V)
x0 = 5*torch.randn([1,2], dtype = torch.float)




# x0 = torch.tensor([[[5,5]]], dtype = torch.float)

plotting = vis.plot_dynamics(f,V,x0)

for i in range(1):

    plotting.plot_trajectory()

plt.show()
