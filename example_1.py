import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pylab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import simple_model as model
import visualize_trajectory as vis

layer_sizes = np.array([2, 100, 1])

fhat = model.fhat(np.array([2, 50, 50, 2]))
V = model.MakePSD(model.ICNN(layer_sizes),2)
# f = model.dynamics_simple(fhat,V)
f = model.dynamics_nonincrease(fhat,V)
# f = model.dynamics_rootfind(fhat,V)
# f = model.dynamics_stochastic(fhat,V)
# x0 = 2.5*torch.randn([1,2], dtype = torch.float)
x0 = torch.tensor([[[3,2]]], dtype = torch.float)

plotting = vis.plot_dynamics(f,V,x0)

for i in range(1):

    plotting.plot_trajectory()

plt.show()
