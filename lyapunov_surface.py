import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

import pylab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import simple_model as model
import rootfind_model
import stochastic_model

import lyapunov_NN as L
import dynamics_plotting as vis

layer_sizes = np.array([2, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
Z = np.ndarray(X.shape)

for i in range(0, len(x)):
    for j in range(0, len(x)):
        z = torch.tensor([[X[i][j],Y[i][j]]], dtype = torch.float)
        Z[i][j] = (V(z))
# zs = np.array(fun(np.ravel(X), np.ravel(Y)))
# Z = zs.reshape(X.shape)
#
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
