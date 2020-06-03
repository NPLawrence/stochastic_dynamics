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


# k = 1
# n = 1
# beta = 1
# fhat = nn.Sequential(nn.Linear(n, 50), nn.ReLU(),
#                     # nn.Linear(50, 50), nn.ReLU(),
#                     nn.Linear(50, 50), nn.ReLU(),
#                     nn.Linear(50, 2*n*k))
# layer_sizes = np.array([n, 50, 50, 1])
# ICNN = L.ICNN(layer_sizes)
# V = L.MakePSD(ICNN,n)

k = 4
n = 1
beta = 1
mode = 1
fhat = nn.Sequential(nn.Linear(n, 50), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 2*n*k))
layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)
# ICNN_simple = L.ICNN(layer_sizes)
# V_simple = L.MakePSD(ICNN_simple,n)
f = stochastic_model.stochastic_module(fhat = fhat, V = V, n=n, k=k, mode = mode, beta = beta, show_mu = True, is_training = False)
# f_simple = stochastic_model_V2.MixtureDensityNetwork(n, n, k)




# ICNN = L.ICNN_2(layer_sizes)
# V_noise = L.MakePSD(L.ICNN_2(layer_sizes),2)

# layer_sizes = np.array([2, 50, 50, 50])
# V = L.Lyapunov_NN(L.PD_weights(layer_sizes))

# PATH_V_noise = './saved_models/simple_V_stochastic_noisyData_ICNN2.pth'
# PATH_f_noise = './saved_models/simple_f_stochastic_noisyData_ICNN2.pth'
# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN_noisyData_ICNN2.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN_noisyData_ICNN2.pth'
# PATH_V = './saved_models/convex_V_stochastic_multiMod.pth'
# PATH_f = './saved_models/convex_f_stochastic_multiMod.pth'
# PATH_V = './saved_models/convex_V_stochastic_multiMod_k2.pth'
PATH_f = './saved_models/convex_f_stochastic_multiMod.pth'
# PATH_f_simple = './saved_models/simple_f_stochastic_multiMod_k3.pth'

# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN.pth'
# V.load_state_dict(torch.load(PATH_V))
# fhat.load_state_dict(torch.load(PATH_f))
# V.load_state_dict(torch.load(PATH_V))
f.load_state_dict(torch.load(PATH_f))
# f_simple.load_state_dict(torch.load(PATH_f_simple))
# f_simple.load_state_dict(torch.load(PATH_f_simple))

# f = stochastic_model.MDN_module(fhat, V, n=n, k=k, beta = beta, is_training = False, show_mu = False)
# f_mu = stochastic_model.MDN_module(fhat, V, n=n, k=k, beta = beta, is_training = False, show_mu = True)

# plotting = vis.plot_dynamics(f,V,is_stochastic = True)
plotting_mu = vis.plot_dynamics(f,V,show_mu = True)

# plotting_simple = vis.plot_dynamics(f_simple,V,is_stochastic = True)
# plotting_simple_mu = vis.plot_dynamics(f_simple,V,show_mu = True)




x0 = torch.tensor([[[2]]], dtype = torch.float)




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

steps = 200
multiMod = generate_data.data_multiMod()
X_true, X_mean_true = multiMod.gen_data(1, steps = steps, train_data = False, x = x0.squeeze().numpy())
# X_true = np.loadtxt("./datasets/data_multiMod.csv", delimiter=",")


plt.subplot(1,2,1)
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 2, "label": "Predicted mean"}
X_mu = plotting_mu.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = plt)
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 0.5, "alpha": 0.40, "label": "Stable prediction"}
# X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = plt)

kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 0.5, "label": "True dynamics"}
# plt.plot(np.linspace(0,len(X_true)-1, len(X_true)), X_true, **kwargs)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True mean"}
# plt.plot(np.linspace(0,len(X_mean_true)-1, len(X_true)), X_mean_true, **kwargs)

plt.xlim([0, steps-1])
plt.legend()
plt.title('Stable model')

plt.subplot(1,2,2)

kwargs = {"color" : "tab:red", "marker": ".", "markersize": 2, "label": "Simple mean"}
# X_mu_simple = plotting_simple_mu.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = plt)
kwargs = {"color" : "tab:red", "marker": ".", "markersize": 0.5, "alpha": 0.40, "label": "Simple model"}
# X_simple = plotting_simple.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, ax = plt)

kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 0.5, "label": "True dynamics"}
plt.plot(np.linspace(0,len(X_true)-1, len(X_true)), X_true, **kwargs)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True mean"}
# plt.plot(np.linspace(0,len(X_mean_true)-1, len(X_true)), X_mean_true, **kwargs)

plt.xlim([0, steps-1])
plt.legend()
# handles,labels = ax1.get_legend_handles_labels()
# handles = [handles[1], handles[2], handles[0]]
# labels = [labels[1], labels[2], labels[0]]
# ax1.legend(handles,labels,loc=1)
plt.title('Simple model')
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


plt.savefig('figures/example_3_stochasticNonlinear.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_3_stochasticNonlinear.png', dpi=400, bbox_inches='tight',pad_inches=.01)
# plt.tight_layout()
plt.show()


plt.show()
