#Simple script for plotting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import pylab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import convex_model as model
import rootfind_model
import stochastic_model

import lyapunov_NN as L
import dynamics_plotting as vis
import generate_data



add_state = True

#rootfind default
# fhat = nn.Sequential(nn.Linear(2, 25), nn.ReLU(),
#                     nn.Linear(25, 25), nn.ReLU(),
#                     nn.Linear(25, 2))
# layer_sizes = np.array([2, 50, 50, 1])
# PATH_V = './saved_models/rootfind_V.pth'
# PATH_f = './saved_models/rootfind_f.pth'

#convex default
n = 2
fhat = model.fhat(np.array([2, 25, 25, 25, n]), False)
layer_sizes = np.array([n, 25, 25, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)
PATH_V = './saved_models/convex_V_VanderPol_stable.pth'
PATH_f = './saved_models/convex_f_VanderPol_stable.pth'


ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,2)



# PATH_ICNN = './saved_models/rootfind_ICNN.pth'

# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN.pth'
# ICNN.load_state_dict(torch.load(PATH_ICNN))
V.load_state_dict(torch.load(PATH_V))
fhat.load_state_dict(torch.load(PATH_f))
# f = rootfind_model.rootfind_module(fhat,V,is_training = False)
f = model.dynamics_convex(fhat,V,True)


A = torch.tensor([[0.90, 1],[0, 0.90]])
f_true = lambda x : F.linear(x, A, bias = False)

plotting = vis.plot_dynamics(f,V)
plotting_true = vis.plot_dynamics(f_true, V)


x0 = torch.tensor([[[-0.75, 3.15]]], dtype = torch.float)
x0 = torch.tensor([[[4, 2]]], dtype = torch.float)

# x0 = torch.tensor([[[-0.75, 3.15]]], dtype = torch.float)

# A = torch.tensor([[0.90, 1],[0, 0.90]])

VanderPol = generate_data.data_VanderPol()
VanderPol.gen_data(1)

kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 1, "label": "Prediction"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = True, steps = 3000, ax = plt)
# kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "label": "True dynamics"}
# X_true = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = plt)

kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 3, "label": "True dynamics"}
X_true = np.loadtxt("./datasets/data_VanderPol_stable.csv", delimiter=",")
plt.plot(X_true[:, 0], X_true[:, 1],  **kwargs)



plt.legend()
plt.show()
