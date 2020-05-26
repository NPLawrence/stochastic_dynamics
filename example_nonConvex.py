#Plots a random instance of a stable model

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
import stochastic_model_V2

import lyapunov_NN as L
import dynamics_plotting as vis
import generate_data

n = 2
fhat = model.fhat(np.array([n, 25, 25, 25, n]), False)

layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)
layer_sizes = np.array([n, 50, 50, 1])
ICNN_rootfind = L.ICNN(layer_sizes)
V_rootfind = L.MakePSD(ICNN_rootfind,n)

PATH_f = './saved_models/convex_f_nonConvex.pth'
PATH_f_rootfind = './saved_models/rootfind_f_nonConvex.pth'
PATH_f_simple = './saved_models/simple_f_nonConvex.pth'


f = model.dynamics_convex(V, n, beta = 0.99)
f_rootfind = rootfind_model.rootfind_module(V_rootfind,n,is_training = False)
f.load_state_dict(torch.load(PATH_f))
f_rootfind.load_state_dict(torch.load(PATH_f_rootfind))

# fhat.load_state_dict(torch.load(PATH_f_simple))

# f = fhat
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,4), sharey=False, constrained_layout=False)
plt.subplots_adjust(wspace = 0.314)
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

plt.rc( 'font', family = "Times New Roman")
plt.rc('mathtext', fontset = 'custom', rm = 'Times New Roman', it = 'Times New Roman:italic', bf = 'Times New Roman:bold')

ax1 = plt.subplot(121)

#Create streamplot
def func(x,y):
    xdot = y
    ydot = -y -np.sin(x) - 2*np.clip(x+y, a_min = -1, a_max = 1)
    return xdot, ydot
x, y = np.meshgrid(np.arange(-10, 10, 0.05), np.arange(-6, 6, 0.05))
xdot, ydot = func(x, y)
ax1.streamplot(x,y,xdot,ydot, linewidth = 1, color = 'tab:blue')


plotting = vis.plot_dynamics(f,V,show_mu = False, is_stochastic = False)
plotting_rootfind = vis.plot_dynamics(f_rootfind,V_rootfind,show_mu = False, is_stochastic = False)

x0 = 5*torch.randn((1,1,2))
# print(x0)
x0 = torch.tensor([[[-3.5,-3.5]]], dtype = torch.float)

kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "alpha": 1, "label": "Predictions"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax1)
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 1, "label": "Predictions \n (root-find)"}
X = plotting_rootfind.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax1)

ax1.legend(framealpha = 1, loc = 'upper left')

kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "alpha": 1, "label": "Prediction"}
x0 = torch.tensor([[[4,4]]], dtype = torch.float)
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax1)
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 1, "label": "Predictions"}
X = plotting_rootfind.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax1)

ax1.set_title('Sample trajectories')

ax2 = plt.subplot(122)

X = plotting.plot_trajectory(x0, kwargs, sample_paths = 0, show_ls = True, steps = 100, ax = ax2)
ax2.set_title('Learned Lyapunov function')

plt.savefig('figures/example_2.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_2.png', transparent=False, dpi=400, bbox_inches='tight',pad_inches=.01)
plt.show()
