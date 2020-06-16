#Plots a random instance of a stable model

import numpy as np
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

k = 2
n = 2
beta = 0.99
fhat = nn.Sequential(nn.Linear(n, 25), L.ReHU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), L.ReHU(),
                    nn.Linear(25, 2*n*k))
fhat_noise = nn.Sequential(nn.Linear(n, 25), L.ReHU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), L.ReHU(),
                    nn.Linear(25, 2*n*k))
fhat_rf_noise = nn.Sequential(nn.Linear(n, 25), L.ReHU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), L.ReHU(),
                    nn.Linear(25, 2*n*k))
layer_sizes = np.array([n, 25, 25, 25, 1])
ICNN = L.ICNN(layer_sizes)
ICNN_noise = L.ICNN(layer_sizes)

ICNN_rf_noise = L.ICNN(layer_sizes)

V = L.MakePSD(ICNN,n)
V_noise = L.MakePSD(ICNN_noise,n)
V_rf_noise = L.MakePSD(ICNN_rf_noise,n)



# PATH_f_LowN = './saved_models/simple_f_stochastic_LowN_noisyData_ICNN2.pth'
PATH_f = './saved_models/convex_f_linear.pth'
PATH_f_noise = './saved_models/convex_f_linear_noise.pth'
PATH_f_rf_noise = './saved_models/rootfind_f_linear_noise.pth'



f = stochastic_model.stochastic_module(fhat = fhat, V = V, n=n, k=k, mode = 1, is_training = False, beta = beta, show_mu = False)


# f = stochastic_model.MDN_module(fhat, V, n=n, k=k, is_training = False, show_mu = False)
f_mu = stochastic_model.stochastic_module(fhat = fhat, V = V, n=n, k=k, mode = 1, is_training = False, beta = beta, show_mu = True)

# f_mu = stochastic_model.MDN_module(fhat, V, n=n, k=k, is_training = False, show_mu = True)
f_noise = stochastic_model.stochastic_module(fhat = fhat_noise, V = V_noise, n=n, k=k, mode = 1, is_training = False, beta = beta, show_mu = False)

# f_noise = stochastic_model.MDN_module(fhat_noise, V_noise, n=n, k=k, is_training = False, show_mu = False)
f_noise_mu = stochastic_model.stochastic_module(fhat = fhat_noise, V = V_noise, n=n, k=k, mode = 1, is_training = False, beta = beta, show_mu = True)

# f_noise_mu = stochastic_model.MDN_module(fhat_noise, V_noise, n=n, k=k, is_training = False, show_mu = True)

f_rf_noise = stochastic_model.stochastic_module(fhat = fhat_rf_noise, V = V_rf_noise, n=n, k=k, mode = 2, is_training = False, beta = beta, show_mu = False)

# f_rf_noise = stochastic_model.stochastic_module(fhat_rf_noise, V_rf_noise, n=n, k=k, is_training = False, show_mu = False)
f_rf_noise_mu = stochastic_model.stochastic_module(fhat = fhat_rf_noise, V = V_rf_noise, n=n, k=k, mode = 2, is_training = False, beta = beta, show_mu = True)

# f_rf_noise_mu = stochastic_model.stochastic_module(fhat_rf_noise, V_rf_noise, n=n, k=k, is_training = False, show_mu = True)

# V.load_state_dict(torch.load(PATH_V))
f.load_state_dict(torch.load(PATH_f))
f_mu.load_state_dict(torch.load(PATH_f))

# V_noise.load_state_dict(torch.load(PATH_V_noise))
f_noise.load_state_dict(torch.load(PATH_f_noise))
f_noise_mu.load_state_dict(torch.load(PATH_f_noise))

# V_rf_noise.load_state_dict(torch.load(PATH_V_rf_noise))
f_rf_noise.load_state_dict(torch.load(PATH_f_rf_noise))
f_rf_noise_mu.load_state_dict(torch.load(PATH_f_rf_noise))



x0 = torch.tensor([[[-0.5, 1.15]]], dtype = torch.float)
x0 = torch.tensor([[[-0.75, 3.15]]], dtype = torch.float)

A = torch.tensor([[0.90, 1],[0, 0.90]])
f_true = lambda x : F.linear(x, A, bias = False)

plotting_true = vis.plot_dynamics(f_true, V)
plotting = vis.plot_dynamics(f,V,reset_model=True)
plotting_mu = vis.plot_dynamics(f_mu,V,reset_model=True)
plotting_noise = vis.plot_dynamics(f_noise,V_noise,reset_model=True)
plotting_noise_mu = vis.plot_dynamics(f_noise_mu,V_noise,reset_model=True)

plotting_rf_noise = vis.plot_dynamics(f_rf_noise,V_rf_noise)
plotting_rf_noise_mu = vis.plot_dynamics(f_rf_noise_mu,V_rf_noise)


# plotting_simple = vis.plot_dynamics(f_simple,V)

# for i in range(1):

    # x0 = 4*torch.randn([1,2], dtype = torch.float)
# fig = figsize=(10,3)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,4), sharey=False, constrained_layout=False)
plt.subplots_adjust(wspace = 0.2718)
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


kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 0.30, "label": "Prediction"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 4, show_ls = False, steps = 100, ax = ax1)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "label": "True dynamics"}
X_true = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax1)
kwargs = {"color" : "tab:red", "marker": "*", "markersize": 5, "label": "Predicted mean"}
X_mu = plotting_mu.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax1)
ax1.set_ylim([-1, 6])
ax1.legend()
handles,labels = ax1.get_legend_handles_labels()
handles = [handles[1], handles[2], handles[0]]
labels = [labels[1], labels[2], labels[0]]
ax1.legend(handles,labels,loc=1)
ax1.set_title('Deterministic training data')

ax2 = plt.subplot(132)
plt.yticks([])
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 0.30, "label": "Prediction"}
X = plotting_noise.plot_trajectory(x0, kwargs, sample_paths = 4, show_ls = False, steps = 100, ax = ax2)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "label": "True mean"}
X_true = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax2)
kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "label": "Predicted mean"}
X_mu = plotting_noise_mu.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax2)
ax2.set_ylim([-1, 6])
ax2.legend()
handles,labels = ax2.get_legend_handles_labels()
handles = [handles[1], handles[2], handles[0]]
labels = [labels[1], labels[2], labels[0]]
ax2.legend(handles,labels,loc=1)
ax2.set_title("Noisy training data")

ax3 = plt.subplot(133)
plt.yticks([])
kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 0.30, "label": "Prediction"}
X = plotting_rf_noise.plot_trajectory(x0, kwargs, sample_paths = 4, show_ls = False, steps = 100, ax = ax3)
kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "label": "True mean"}
X_true = plotting_true.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax3)
kwargs = {"color" : "tab:red", "marker": ".", "markersize": 5, "label": "Predicted mean"}
X_mu = plotting_rf_noise_mu.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = 100, ax = ax3)
ax3.set_ylim([-1, 6])
ax3.legend()
handles,labels = ax2.get_legend_handles_labels()
handles = [handles[1], handles[2], handles[0]]
labels = [labels[1], labels[2], labels[0]]
ax3.legend(handles,labels,loc=1)
ax3.set_title("Root-finding method")


plt.savefig('figures/example_1_rf.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_1_rf.png', transparent=False, dpi=400, bbox_inches='tight',pad_inches=.01)
# plt.tight_layout()
plt.show()
