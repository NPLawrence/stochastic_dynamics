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


n = 3
add_state = True

f_simple = model.fhat(np.array([n, 25, 25, 25, n]), True)
fhat = model.fhat(np.array([n, 25, 25, 25, n]), False)
layer_sizes = np.array([n, 25, 25, 1])
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
PATH_V = './saved_models/convex_V_Lorenz_stable.pth'
PATH_f = './saved_models/convex_f_Lorenz_stable.pth'
PATH_f_simple = './saved_models/simple_f_Lorenz.pth'
# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN.pth'
V.load_state_dict(torch.load(PATH_V))
fhat.load_state_dict(torch.load(PATH_f))
f = model.dynamics_convex(fhat,V,add_state)
f_simple.load_state_dict(torch.load(PATH_f_simple))


# V_noise.load_state_dict(torch.load(PATH_V_noise))
# fhat_noise.load_state_dict(torch.load(PATH_f_noise))

# f = stochastic_model.MDN_module(fhat, V, k, is_training = False, show_mu = False)
# f_mu = stochastic_model.MDN_module(fhat, V, k, is_training = False, show_mu = True)
# f_noise = stochastic_model.MDN_module(fhat_noise, V_noise, k, is_training = False, show_mu = False)
# f_noise_mu = stochastic_model.MDN_module(fhat_noise, V_noise, k, is_training = False, show_mu = True)

# x0 = torch.tensor([[[-.75,-.75]]], dtype = torch.float)
# x0 = torch.rand(size = [1,1,2])





x0 = torch.tensor([[[1,1,1]]], dtype = torch.float)

lorenz = generate_data.data_Lorenz()
lorenz.gen_data(1)
# x0 = torch.tensor([[[-0.75, 3.15]]], dtype = torch.float)

# A = torch.tensor([[0.90, 1],[0, 0.90]])
# f_true = lambda x : F.linear(x, A, bias = False)

plotting = vis.plot_dynamics_3D(f,V)
plotting_nominal = vis.plot_dynamics_3D(f_simple,V)
# plotting_mu = vis.plot_dynamics(f_mu,V)
# plotting_noise = vis.plot_dynamics(f_noise,V_noise)
# plotting_noise_mu = vis.plot_dynamics(f_noise_mu,V_noise)

# plotting_simple = vis.plot_dynamics(f_simple,V)

# for i in range(1):

    # x0 = 4*torch.randn([1,2], dtype = torch.float)
# fig = figsize=(10,3)
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,4), sharey=False, constrained_layout=False)
fig = plt.figure(1)
fig.gca(projection='3d')
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

# fig = plt.figure(constrained_layout=False)
# spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                          # height_ratios=heights)
# print(spec[1,0])

# ax1 = fig.add_subplot(aspect=3)
# ax1 = plt.subplot(111)

# kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "label": "Nominal prediction"}
# X_nominal = plotting_nominal.plot_trajectory(x0, kwargs, sample_paths = 1, show_mu = True, steps = 2000)
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 3, "label": "Stable prediction"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, steps = 2000)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True dynamics"}
X_true = np.loadtxt("./datasets/data_Lorenz_stable.csv", delimiter=",")
plt.plot(X_true[:, 0], X_true[:, 1], X_true[:, 2], **kwargs)

# ax1.set_ylim([-1, 6])
plt.legend()
# handles,labels = ax1.get_legend_handles_labels()
# handles = [handles[1], handles[2], handles[0]]
# labels = [labels[1], labels[2], labels[0]]
# ax1.legend(handles,labels,loc=1)
plt.title('Deterministic')
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


plt.savefig('figures/example_2_convex.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_2_convex.png', dpi=400, bbox_inches='tight',pad_inches=.01)
# plt.tight_layout()
plt.show()


fig = plt.figure(2)

kwargs_nominal = {"color" : "tab:red", "marker": ".", "markersize": 3, "label": "Nominal prediction"}
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 3, "label": "Stable prediction"}
kwargs_true = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True dynamics"}

plt.subplot(131)
# plt.plot(X_nominal[:,0],**kwargs_nominal)
plt.plot(X[:,0],**kwargs)
plt.plot(X_true[:, 0],**kwargs_true)

plt.subplot(132)
# plt.plot(X_nominal[:,1],**kwargs_nominal)
plt.plot(X[:,1],**kwargs)
plt.plot(X_true[:, 1],**kwargs_true)

plt.subplot(133)
# plt.plot(X_nominal[:,2],**kwargs_nominal)
plt.plot(X[:,2],**kwargs)
plt.plot(X_true[:, 2],**kwargs_true)

plt.show()
