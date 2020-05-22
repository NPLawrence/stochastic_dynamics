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

import lyapunov_NN as L
import dynamics_plotting as vis
import generate_data


k = 1
n = 1
beta = 1
fhat = nn.Sequential(nn.Linear(n, 50), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 2*n*k))
layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)

# ICNN = L.ICNN_2(layer_sizes)
# V_noise = L.MakePSD(L.ICNN_2(layer_sizes),2)

# layer_sizes = np.array([2, 50, 50, 50])
# V = L.Lyapunov_NN(L.PD_weights(layer_sizes))

# PATH_V_noise = './saved_models/simple_V_stochastic_noisyData_ICNN2.pth'
# PATH_f_noise = './saved_models/simple_f_stochastic_noisyData_ICNN2.pth'
# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN_noisyData_ICNN2.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN_noisyData_ICNN2.pth'
PATH_V = './saved_models/convex_V_stochastic_multiMod.pth'
PATH_f = './saved_models/convex_f_stochastic_multiMod.pth'
PATH_f_simple = './saved_models/simple_f_Lorenz.pth'
# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN.pth'
V.load_state_dict(torch.load(PATH_V))
fhat.load_state_dict(torch.load(PATH_f))
f = stochastic_model.MDN_module(fhat, V, n=n, k=k, beta = beta, is_training = False, show_mu = False)
f_mu = stochastic_model.MDN_module(fhat, V, n=n, k=k, beta = beta, is_training = False, show_mu = True)

plotting = vis.plot_dynamics(f,V)
plotting_mu = vis.plot_dynamics(f_mu,V)




x0 = torch.tensor([[[0.1]]], dtype = torch.float)

multiMod = generate_data.data_multiMod()
X_true, X_mean_true = multiMod.gen_data(1, False, x0.squeeze().numpy())


fig = plt.figure(1)
# plt.subplots_adjust(wspace = 0.314)
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

plt.rc( 'font', family = "Times New Roman")
plt.rc('mathtext', fontset = 'custom', rm = 'Times New Roman', it = 'Times New Roman:italic', bf = 'Times New Roman:bold')

# fig = plt.figure(constrained_layout=False)
# spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                          # height_ratios=heights)
# print(spec[1,0])

# ax1 = fig.add_subplot(aspect=3)
# ax1 = plt.subplot(111)

kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "label": "Predicted mean"}
X_mu = plotting_mu.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, show_mu = True, steps = 100, ax = plt)
# X_nominal = plotting_nominal.plot_trajectory(x0, kwargs, sample_paths = 1, show_mu = True, steps = 2000)
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 3, "alpha": 0.30, "label": "Stable prediction"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 2, show_ls = False, steps = 100, ax = plt)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True dynamics"}
plt.plot(np.linspace(0,len(X_true)-1, len(X_true)), X_true, **kwargs)
kwargs = {"color" : "tab:green", "marker": ".", "markersize": 2, "label": "True mean"}
plt.plot(np.linspace(0,len(X_mean_true)-1, len(X_true)), X_mean_true, **kwargs)

plt.xlim([0, len(X_true)-1])
plt.legend()
# handles,labels = ax1.get_legend_handles_labels()
# handles = [handles[1], handles[2], handles[0]]
# labels = [labels[1], labels[2], labels[0]]
# ax1.legend(handles,labels,loc=1)
plt.title('Stochastic process')
#
# ax2 = plt.subplot(122)
# # ax2 = fig.add_subplot(aspect=3)
# plt.yticks([])
# kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 0.30, "label": "Prediction"}
# X = plotting_noise.plot_trajectory(x0, kwargs, sample_paths = 8, show_ls = True, steps = 100, ax = ax2)
# kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "label": "True mean"}
# X_true = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax2)
# kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "label": "Predicted mean"}
# X_mu = plotting_noise_mu.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, show_mu = True, steps = 100, ax = ax2)
# ax2.set_ylim([-1, 6])
# ax2.legend()
# handles,labels = ax2.get_legend_handles_labels()
# handles = [handles[1], handles[2], handles[0]]
# labels = [labels[1], labels[2], labels[0]]
# ax2.legend(handles,labels,loc=1)
# ax2.set_title("Noisy training data")


plt.savefig('figures/example_3_convex.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_3_convex.png', dpi=400, bbox_inches='tight',pad_inches=.01)
# plt.tight_layout()
plt.show()




plt.show()
