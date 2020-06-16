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
import stochastic_model_V2

import lyapunov_NN as L
import dynamics_plotting as vis
import generate_data


k = 1
n = 3
beta = 1
mode = 1

Lorenz = generate_data.data_Lorenz()
Lorenz.gen_data(1)

f_simple = model.fhat(np.array([n, 25, 25, 25, n]), add_state = True)
# fhat = model.fhat(np.array([n, 25, 25, 25, n]), False)
layer_sizes = np.array([n, 25, 25, 25, 1])

# layer_sizes = np.array([n, 50, 50, 1])

ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)

# f = model.dynamics_convex(V, n, beta = 0.99, add_state = add_state)

f = model.dynamics_nonincrease(V, n)
# f = stochastic_model_V2.MixtureDensityNetwork(n, n, k, V, mode = mode)



# ICNN = L.ICNN_2(layer_sizes)
# V_noise = L.MakePSD(L.ICNN_2(layer_sizes),2)

# layer_sizes = np.array([2, 50, 50, 50])
# V = L.Lyapunov_NN(L.PD_weights(layer_sizes))

# PATH_V_noise = './saved_models/simple_V_stochastic_noisyData_ICNN2.pth'
# PATH_f_noise = './saved_models/simple_f_stochastic_noisyData_ICNN2.pth'
# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN_noisyData_ICNN2.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN_noisyData_ICNN2.pth'
# PATH_V = './saved_models/convex_V_Lorenz.pth'
# PATH_f = './saved_models/convex_f_Lorenz.pth'
# PATH_f_simple = './saved_models/simple_f_Lorenz.pth'

# PATH_f = './saved_models/convex_f_stochastic_Lorenz_k3.pth'
PATH_f = './saved_models/noninc_f_Lorenz_unstable.pth'
PATH_f_simple = './saved_models/simple_f_Lorenz_unstable.pth'

# PATH_f = './saved_models/convex_f_Lorenz.pth'



# PATH_V_LowN = './saved_models/simple_V_stochastic_LowN.pth'
# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN.pth'
# V.load_state_dict(torch.load(PATH_V))
# fhat.load_state_dict(torch.load(PATH_f))
# f = model.dynamics_convex(fhat,V,add_state)
f.load_state_dict(torch.load(PATH_f))
f_simple.load_state_dict(torch.load(PATH_f_simple))


# f_simple.load_state_dict(torch.load(PATH_f_simple))


# V_noise.load_state_dict(torch.load(PATH_V_noise))
# fhat_noise.load_state_dict(torch.load(PATH_f_noise))

# f = stochastic_model.MDN_module(fhat, V, k, is_training = False, show_mu = False)
# f_mu = stochastic_model.MDN_module(fhat, V, k, is_training = False, show_mu = True)
# f_noise = stochastic_model.MDN_module(fhat_noise, V_noise, k, is_training = False, show_mu = False)
# f_noise_mu = stochastic_model.MDN_module(fhat_noise, V_noise, k, is_training = False, show_mu = True)




x0 = torch.tensor([[[1.2,1.1,0.9]]], dtype = torch.float)

lorenz = generate_data.data_Lorenz()
lorenz.gen_data(1)

plotting = vis.plot_dynamics_3D(f,V,show_mu = False, is_stochastic = False)
plotting_nominal = vis.plot_dynamics_3D(f_simple,V)

# plotting = vis.plot_dynamics_3D(f,V)
# plotting_nominal = vis.plot_dynamics_3D(f_simple,V)

# plotting_mu = vis.plot_dynamics(f_mu,V)
# plotting_noise = vis.plot_dynamics(f_noise,V_noise)
# plotting_noise_mu = vis.plot_dynamics(f_noise_mu,V_noise)

# plotting_simple = vis.plot_dynamics(f_simple,V)

# for i in range(1):

    # x0 = 4*torch.randn([1,2], dtype = torch.float)
# fig = figsize=(10,3)
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,4), sharey=False, constrained_layout=False)
# fig = plt.figure(1)
fig = plt.figure(figsize=plt.figaspect(0.5))
steps = 3000

# fig.gca(projection='3d')
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
# ax1 = plt.subplot(121)
# fig,_  = plt.subplots(1, 4, gridspec_kw={'width_ratios': [3, 1,1,1]}, figsize=(15,5))
# fig.suptitle('Bounded model')
ax1 = fig.add_subplot(121, projection='3d')
# ax1.gca(projection='3d')
kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "label": "Nominal prediction"}
# X_nominal = plotting_nominal.plot_trajectory(x0, kwargs, sample_paths = 1, steps = 1000)
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 3, "label": "Bounded prediction"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, steps = steps)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True dynamics"}
X_true = np.loadtxt("./datasets/data_Lorenz.csv", delimiter=",")
ax1.plot(X_true[:, 0], X_true[:, 1], X_true[:, 2], **kwargs)
ax1.set_title('Bounded increments')

ax2 = fig.add_subplot(122, projection='3d')

kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "label": "Nominal prediction"}
X_nominal = plotting_nominal.plot_trajectory(x0, kwargs, sample_paths = 1, steps = steps)
# kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "label": "True mean"}
# X_true = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax2)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True dynamics"}
X_true = np.loadtxt("./datasets/data_Lorenz_stable.csv", delimiter=",")
ax2.plot(X_true[:, 0], X_true[:, 1], X_true[:, 2], **kwargs)

ax2.set_title('Simple model')

plt.tight_layout()


plt.savefig('figures/example_lorenz.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_lorenz.png', dpi=400, bbox_inches='tight',pad_inches=.01)
# plt.tight_layout()


fig = plt.figure(figsize=plt.figaspect(0.5))

kwargs_nominal = {"color" : "tab:red", "marker": ".", "markersize": 3, "label": "Nominal prediction"}
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 3, "label": "Stable prediction"}
kwargs_true = {"color" : "tab:blue", "marker": ".", "markersize": 2, "label": "True dynamics"}

plt.subplot(231)
plt.plot(X_true[:, 0],**kwargs_true)
plt.plot(X[:,0],**kwargs)
plt.ylabel('$x$')
plt.xlim([0, steps-1])

#
plt.subplot(232)
plt.plot(X_true[:, 1],**kwargs_true)
plt.plot(X[:,1],**kwargs)
plt.title('Bounded increments')
plt.ylabel('$y$')
plt.xlim([0, steps-1])

#
plt.subplot(233)
plt.plot(X_true[:, 2],**kwargs_true)
plt.plot(X[:,2],**kwargs)
plt.ylabel('$z$')
plt.xlim([0, steps-1])


plt.subplot(234)
plt.plot(X_true[:, 0],**kwargs_true)
plt.plot(X_nominal[:,0],**kwargs_nominal)
plt.ylabel('$x$')
plt.xlabel('Time step')
plt.xlim([0, steps-1])

#
plt.subplot(235)
plt.plot(X_true[:, 1],**kwargs_true)
plt.plot(X_nominal[:,1],**kwargs_nominal)
plt.title('Simple model')
plt.ylabel('$y$')
plt.xlabel('Time step')
plt.xlim([0, steps-1])

#
plt.subplot(236)
plt.plot(X_true[:, 2],**kwargs_true)
plt.plot(X_nominal[:,2],**kwargs_nominal)
plt.ylabel('$z$')
plt.xlabel('Time step')
plt.xlim([0, steps-1])

plt.tight_layout()

# ax1.set_ylim([-1, 6])
# ax1.legend()
# handles,labels = ax1.get_legend_handles_labels()
# handles = [handles[1], handles[2], handles[0]]
# labels = [labels[1], labels[2], labels[0]]
# ax1.legend(handles,labels,loc=1)
# plt.title('Deterministic')
#
# ax2 = plt.subplot(122)
# # ax2 = fig.add_subplot(aspect=3)
# plt.yticks([])
# fig = plt.figure(2)

# ax2.set_ylim([-1, 6])
# ax2.legend()
# handles,labels = ax2.get_legend_handles_labels()
# handles = [handles[1], handles[2], handles[0]]
# labels = [labels[1], labels[2], labels[0]]
# ax2.legend(handles,labels,loc=1)
# ax2.legend(loc=1)
# ax2.set_title("Noisy training data")


plt.savefig('figures/example_lorenz_grid.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_lorenz_grid.png', dpi=400, bbox_inches='tight',pad_inches=.01)
# plt.tight_layout()
plt.show()
