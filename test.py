# Simple script for validating trained models
#   This just loads a model and visualizes trajectories (dynamics_plotting.py)
#   from the learned model against the true dynamics (true_dynamics.py)

import numpy as np
import matplotlib.pyplot as plt
import torch

import modules.convex_model as convex_model
import modules.rootfind_model as rootfind_model
import modules.stochastic_model as stochastic_model
import modules.lyapunov_NN as L

import dynamics_plotting as vis
import true_dynamics

n = 2 # state dimension

layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)

experiment = 'exp_name' + '.pth'
PATH_f_net = './saved_models/' + experiment

f = rootfind_model.dynamics_model(V, n, is_training = False)
f.load_state_dict(torch.load(PATH_f_net))
plotting = vis.plot_dynamics(f,V)

x0 = torch.randn((1,1,2)) + 3

sample_trajectory = true_dynamics.data_linear()
X_true = sample_trajectory.gen_data(x0)

kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 3, "alpha": 1, "label": "Prediction"}
X = plotting.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = True, steps = 50, ax = plt)

kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 3, "label": "True dynamics"}
plt.plot(X_true[:, 0], X_true[:, 1],  **kwargs)

plt.legend()
plt.show()
