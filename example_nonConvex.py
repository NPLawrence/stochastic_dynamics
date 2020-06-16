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

#Current saved vesion uses ReLU (not Softplus) in the implicit method


n = 2
fhat = model.fhat(np.array([n, 25, 25, 25, n]), False)

layer_sizes = np.array([n, 25, 25, 25, 1])
ICNN = L.ICNN(layer_sizes)
V_ICNN = L.MakePSD(ICNN,n)
V = L.Lyapunov_NN(L.PD_weights(np.array([n, 25, 25, 25])))


# layer_sizes = np.array([n, 50, 50, 1])
# ICNN_rootfind = L.ICNN(layer_sizes)
# V_rootfind = L.MakePSD(ICNN_rootfind,n,make_nonConvex = True)
V_rootfind = L.Lyapunov_NN(L.PD_weights(np.array([n, 25, 25, 25]), make_convex = False))


PATH_f = './saved_models/convex_f_nonConvex.pth'
PATH_f_ICNN = './saved_models/convex_f_nonConvex_ICNN.pth'
PATH_f_rootfind = './saved_models/rootfind_f_nonConvexV.pth'
PATH_f_simple = './saved_models/simple_f_nonConvex.pth'



f = model.dynamics_convex(V, n, beta = 0.99)
f_ICNN = model.dynamics_convex(V_ICNN, n, beta = 0.99)
f_rootfind = rootfind_model.rootfind_module(V_rootfind,n,is_training = False)
f.load_state_dict(torch.load(PATH_f))
f_ICNN.load_state_dict(torch.load(PATH_f_ICNN))
f_rootfind.load_state_dict(torch.load(PATH_f_rootfind))

# f = fhat
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,4), sharey=False, constrained_layout=False)
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

ax1 = plt.subplot(131)

#Create streamplot
def func(x,y):
    xdot = y
    ydot = -y -np.sin(x) - 2*np.clip(x+y, a_min = -1, a_max = 1)
    return xdot, ydot
x, y = np.meshgrid(np.arange(-10, 10, 0.05), np.arange(-6, 6, 0.05))
xdot, ydot = func(x, y)
ax1.streamplot(x,y,xdot,ydot, linewidth = 0.5, color = 'tab:blue')


plotting = vis.plot_dynamics(f,V,show_mu = False)
plotting_ICNN = vis.plot_dynamics(f_ICNN,V,show_mu = False)
plotting_rootfind = vis.plot_dynamics(f_rootfind,V_rootfind,show_mu = False)
sample_trajectory = generate_data.data_nonConvex()

steps = 100
x0 = torch.tensor([[[-5.5,-3]]], dtype = torch.float)

kwargs_true = {"color" : "tab:blue", "markersize": 8, "alpha": 1, "linewidth": 3, "label": "True dynamics"}
X_true = sample_trajectory.gen_data(x0=x0, steps=steps)
ax1.plot(X_true[:,0], X_true[:,1], **kwargs_true)


kwargs = {"color" : "tab:red",  "markersize": 5, "alpha": 1, "linewidth": 2, "label": "Implicit model"}
X = plotting_rootfind.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = ax1)

ax1.legend(framealpha = 1, loc = 'upper left', ncol=1)

x0 = torch.tensor([[[5.5,3.5]]], dtype = torch.float)
X_true = sample_trajectory.gen_data(x0=x0, steps=steps)
ax1.plot(X_true[:,0], X_true[:,1], **kwargs_true)

# kwargs = {"color" : "tab:purple",  "markersize": 4, "alpha": 1, "label": "Prediction"}
# X_ICNN = plotting_ICNN.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = ax1)
# kwargs = {"color" : "tab:green", "markersize": 4, "alpha": 1, "label": "Predictions \n (convex-LNN)"}
# X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = ax1)
kwargs = {"color" : "tab:red",  "markersize": 4, "alpha": 1, "label": "Predictions"}
X = plotting_rootfind.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = ax1)

ax1.plot(0, 0, color = "tab:blue", marker = '*', markersize = 10)
ax1.set_ylim(-4,6)
ax1.set_ylabel('$x_2$')
ax1.set_xlabel('$x_1$')
ax1.set_title('Sample trajectories')

ax2 = plt.subplot(132)

kwargs_true = {"color" : "tab:blue", "marker":'.', "markersize": 5, "alpha": 1, "linewidth": 1, "label": "True dynamics"}
X_true = sample_trajectory.gen_data(x0=x0, steps=steps)
plt.plot( X_true, **kwargs_true)

kwargs = {"color" : "tab:red", 'linestyle': '-', "markersize": 3, "alpha": 1, "linewidth": 2, "label": "Implicit"}
X_rootfind = plotting_rootfind.get_trajectory(x0, steps = steps)
plt.plot(np.linspace(0,steps, steps), X_rootfind, **kwargs)

kwargs = {"color" : "tab:purple", 'linestyle': ':', "markersize": 2, "alpha": 1, "linewidth": 2, "label": "ICNN"}
X_ICNN = plotting_ICNN.get_trajectory(x0, steps = steps)
plt.plot(np.linspace(0,steps, steps), X_ICNN, **kwargs)


kwargs = {"color" : "tab:green", 'linestyle': '--', "markersize": 2, "alpha": 1, "linewidth": 2, "label": "convex-LNN"}
X = plotting.get_trajectory(x0, steps = steps)
plt.plot(np.linspace(0,steps, steps), X, **kwargs)





# ax2.legend(framealpha = 1, loc = 'upper right', ncol=2)

handles,labels = ax2.get_legend_handles_labels()

handles = handles[0::2]
labels = labels[0::2]
ax2.set_ylim((-2.5, 10))
ax2.set_xlim((0, steps))
ax2.set_ylabel('$x$')
ax2.set_xlabel('Time step')
ax2.set_title('Sample trajectories')

ax2.annotate('$x_1$', (0, x0.squeeze()[0].numpy()), (10, x0.squeeze()[0].numpy()), arrowprops = dict(arrowstyle='->'))
ax2.annotate('$x_2$', (0, x0.squeeze()[1].numpy()), (10, x0.squeeze()[1].numpy()), arrowprops = dict(arrowstyle='->'))
ax2.legend(handles,labels)


ax3 = plt.subplot(133)

X = plotting_rootfind.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = True, steps = 1, ax = ax3)
ax3.set_ylabel('$x_2$')
ax3.set_xlabel('$x_1$')
ax3.set_title('Learned Lyapunov function')

plt.savefig('figures/example_2.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_2.png', transparent=False, dpi=400, bbox_inches='tight',pad_inches=.01)
plt.show()
