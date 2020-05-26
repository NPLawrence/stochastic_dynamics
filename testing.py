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
import stochastic_model_V3
import stochastic_model_V2

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
k = 2
n = 2
beta = 0.99
mode = 1
# fhat = model.fhat(np.array([2, 25, 25, 25, n]), False)
# layer_sizes = np.array([n, 25, 25, 1])
# ICNN = L.ICNN(layer_sizes)
# V = L.MakePSD(ICNN,n)
# PATH_V = './saved_models/convex_V_VanderPol_stable.pth'
# PATH_f = './saved_models/convex_f_VanderPol_stable.pth'

#linear experiment
fhat = nn.Sequential(nn.Linear(n, 50), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 2*n*k))
layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)

# f = stochastic_model_V2.MixtureDensityNetwork(n, n, k, V, mode = mode)
# PATH_V = './saved_models/convex_V_stochastic_linear_V3.pth'
PATH_f = './saved_models/convex_f_nonConvex.pth'
PATH_f = './saved_models/rootfind_f_nonConvex.pth'

# V.load_state_dict(torch.load(PATH_V))
# fhat.load_state_dict(torch.load(PATH_f))
# f = stochastic_model_V3.MDN_module(fhat, V, n=n, k=k, is_training = False, show_mu = False)
# f = stochastic_model.MDN_module(fhat, V, n=n, k=k, beta = beta, show_mu = False, is_training = False)
# f = stochastic_model.stochastic_module(fhat = fhat, V = V, n=n, k=k, mode = mode, beta = beta, show_mu = False, is_training = False)


# f = model.dynamics_convex(V, n, beta = 0.99)
f = rootfind_model.rootfind_module(V,n,is_training = True)


# f = stochastic_model_V2.MixtureDensityNetwork(n, n, k, V, mode = mode)

#
# fhat = model.fhat(np.array([n, 25, 25, 25, n]), False)
#
# layer_sizes = np.array([n, 25, 25, 1])
# ICNN = L.ICNN(layer_sizes)
# V = L.MakePSD(ICNN,n)
# # layer_sizes = np.array([2, 50, 50, 50])
# # V = L.Lyapunov_NN(L.PD_weights(layer_sizes))
#
# # PATH_V = './saved_models/convex_V_VanderPol_stable.pth'
# # PATH_f = './saved_models/convex_f_VanderPol_stable.pth'
# PATH_f = './saved_models/convex_f_Lorenz.pth'
# PATH_f = './saved_models/convex_f_linear_twostep.pth'
# f = model.dynamics_convex(V, n, add_state=add_state)


f.load_state_dict(torch.load(PATH_f))




# PATH_ICNN = './saved_models/rootfind_ICNN.pth'

# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN.pth'
# ICNN.load_state_dict(torch.load(PATH_ICNN))
# V.load_state_dict(torch.load(PATH_V))
# fhat.load_state_dict(torch.load(PATH_f))
# f = rootfind_model.rootfind_module(fhat,V,is_training = False)
# f = model.dynamics_convex(fhat,V,True)


A = torch.tensor([[0.90, 1],[0, 0.90]])
f_true = lambda x : F.linear(x, A, bias = False)

plotting = vis.plot_dynamics(f,V,show_mu = False, is_stochastic = False)
plotting_true = vis.plot_dynamics(f_true, V)


x0 = 5*torch.randn((1,1,2))
x0 = torch.tensor([[[6,1]]], dtype = torch.float)
# x0 = torch.tensor([[[1]]], dtype = torch.float)

# x0 = torch.tensor([[[4, 2]]], dtype = torch.float)

# x0 = torch.tensor([[[-0.75, 3.15]]], dtype = torch.float)

# A = torch.tensor([[0.90, 1],[0, 0.90]])

# VanderPol = generate_data.data_VanderPol()
# VanderPol.gen_data(1)

kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 1, "label": "Prediction"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = True, steps = 200, ax = plt)
# kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "label": "True dynamics"}
# X_true = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = plt)

kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 3, "label": "True dynamics"}
# X_true = np.loadtxt("./datasets/data_VanderPol_stable.csv", delimiter=",")
# X_true = np.loadtxt("./datasets/data_linear.csv", delimiter=",")
# plt.plot(X_true[:, 0], X_true[:, 1],  **kwargs)

# X = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = plt)




plt.legend()
plt.show()
