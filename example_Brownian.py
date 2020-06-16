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

k = 6
n = 2
beta = 0.99

eval_performance = True
steps = 100
sample_trajectory = generate_data.data_stochasticNonlinear()


fhat = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), nn.ReLU(),
                    nn.Linear(25, 2*n*k))
fhat_noise = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
                # nn.Linear(50, 50), nn.ReLU(),
                nn.Linear(25, 25), nn.ReLU(),
                nn.Linear(25, 2*n*k))
fhat_simple = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), nn.ReLU(),
                    nn.Linear(25, 2*n*6))
fhat_simple_noise = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), nn.ReLU(),
                    nn.Linear(25, 2*n*6))
layer_sizes = np.array([n, 25, 25, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)
layer_sizes = np.array([n, 25, 25, 1])
ICNN_noise = L.ICNN(layer_sizes)
V_noise = L.MakePSD(ICNN_noise,n)


show_mu = True
f = stochastic_model.stochastic_module(fhat = fhat, V = V, n=n, k=k, mode = 1, beta = beta, show_mu = show_mu, is_training = False)
f_simple = stochastic_model.stochastic_module(fhat = fhat_simple, V = None, n=n, k=6, mode = None, beta = beta, show_mu = show_mu, is_training = False)
f_noise = stochastic_model.stochastic_module(fhat = fhat_noise, V = V_noise, n=n, k=k, mode = 1, beta = beta, show_mu = False, is_training = False)
f_simple_noise = stochastic_model.stochastic_module(fhat = fhat_simple_noise, V = None, n=n, k=6, mode = None, beta = beta, show_mu = False, is_training = False)


PATH_f = './saved_models/convex_f_stochastic_nonLinear2.pth'
PATH_f_simple = './saved_models/simple_f_stochastic_nonLinear2.pth'


f.load_state_dict(torch.load(PATH_f))
f_simple.load_state_dict(torch.load(PATH_f_simple))
f_noise.load_state_dict(torch.load(PATH_f))
f_simple_noise.load_state_dict(torch.load(PATH_f_simple))

##Evaluate stable and simple models over many trajectories

f_eval = stochastic_model.stochastic_module(fhat = fhat, V = V, n=n, k=k, mode = 1, beta = beta, show_mu = show_mu, is_training = False)
fhat_eval = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), nn.ReLU(),
                    nn.Linear(25, 2*n*k))
layer_sizes = np.array([n, 25, 25, 1])
ICNN_eval = L.ICNN(layer_sizes)
V_eval = L.MakePSD(ICNN_eval,n)
fhat_simple_eval = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(25, 25), nn.ReLU(),
                    nn.Linear(25, 2*n*6))

f_eval = stochastic_model.stochastic_module(fhat = fhat_eval, V = V_eval, n=n, k=k, mode = 1, beta = beta, show_mu = False, is_training = True, return_mean = False)
f_simple_eval = stochastic_model.stochastic_module(fhat = fhat_simple_eval, V = None, n=n, k=6, mode = None, beta = beta, show_mu = False, is_training = True, return_mean = False)
f_eval.load_state_dict(torch.load(PATH_f))
f_simple_eval.load_state_dict(torch.load(PATH_f_simple))

num_trajectories = 20

logp_stable = np.zeros((num_trajectories,steps))
logp_simple = np.zeros((num_trajectories,steps))


for i in range(num_trajectories):
        # x0 = 3*torch.tensor([[[torch.rand(1)+0.2, -torch.rand(1)-0.2]]])
        # x0 = 3*torch.randn((1,1,n))
        x0 = 5*torch.tensor([[[torch.rand(1)-0.5, -torch.rand(1)-0.5]]])
        inputs = sample_trajectory.gen_data(x0,steps = steps)
        outputs = inputs[1:]
        logp1 = f_eval(torch.tensor(inputs[0:-1], dtype = torch.float).view((-1,1,n)), torch.tensor(outputs, dtype = torch.float).view((-1,1,n)))
        logp2 = f_simple_eval(torch.tensor(inputs[0:-1], dtype = torch.float).view((-1,1,n)), torch.tensor(outputs, dtype = torch.float).view((-1,1,n)))
        logp_stable[i] = logp1.detach().numpy()
        logp_simple[i] = logp2.detach().numpy()


##Plotting
f_noise.reset()
plotting = vis.plot_dynamics(f,V,show_mu = show_mu)
plotting_simple = vis.plot_dynamics(f_simple,V,show_mu = show_mu)
plotting_noise = vis.plot_dynamics(f_noise,V_noise,show_mu = False)
plotting_simple_noise = vis.plot_dynamics(f_simple_noise,V,show_mu = False)


x0 = 2*torch.tensor([[[torch.rand(1)+0.2, -torch.rand(1)-0.2]]])
print(x0)

# x0 = torch.tensor([[[1.9847, -1.5158]]], dtype = torch.float)
# x0 = torch.tensor([[[2.2470, -2.2133]]], dtype = torch.float)
#
# x0 = torch.tensor([[[2.0416, -0.9034]]], dtype = torch.float)


SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

plt.rc( 'font', family = "Times New Roman")
plt.rc('mathtext', fontset = 'custom', rm = 'Times New Roman', it = 'Times New Roman:italic', bf = 'Times New Roman:bold')


sample_trajectory = generate_data.data_stochasticNonlinear()
X_true = sample_trajectory.gen_data(x0,steps = steps)
X = plotting.get_trajectory(x0, steps = steps)

fig, _ = plt.subplots(1,3, figsize=(14,4), sharey=False, constrained_layout=False)
fig.subplots_adjust(left=0.09, bottom=None, right=0.93, top=None, wspace=0.26, hspace=None)


ax1 = plt.subplot(1,3,1)

kwargs = {"color" : "black", "linestyle": "--", "linewidth": 1.25, "label": "Equilibrium"}
plt.plot(np.linspace(0,steps-1, steps), np.zeros(steps), **kwargs)

kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 0.30, "label": "Prediction"}

for i in range(4):


        X_noise = plotting_noise.get_trajectory(x0, steps)
        if i > 0:
                kwargs["label"] = None
        ax1.plot(np.linspace(0,steps-1, steps), X_noise, **kwargs)
        f_noise.reset()

kwargs = {"color" : "tab:red", "linewidth": 2, "label": "Predicted mean"}
plt.plot(np.linspace(0,steps-1, steps), X, **kwargs)


kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "linewidth": 1, "label": "True dynamics"}
plt.plot(X_true, **kwargs)


handles,labels = ax1.get_legend_handles_labels()
labels = labels[1::2]
labels = labels[::-1]
handles = handles[1::2]
handles = handles[::-1]
ax1.legend(handles,labels)

ax1.set_xlim([0, steps-1])
ax1.set_xlabel('Time step')
ax1.set_ylabel('$x$')
ax1.set_title('Stable model')

ax1.annotate('$x_1$', (0, x0.squeeze()[0].numpy()), (20, x0.squeeze()[0].numpy()), arrowprops = dict(arrowstyle='->'))
ax1.annotate('$x_2$', (0, x0.squeeze()[1].numpy()), (20, x0.squeeze()[1].numpy()), arrowprops = dict(arrowstyle='->'))


ax2 = plt.subplot(1,3,2)

kwargs = {"color" : "black", "linestyle": "--", "linewidth": 1.25, "label": "Equilibrium"}
plt.plot(np.linspace(0,steps-1, steps), np.zeros(steps), **kwargs)

kwargs = {"color" : "tab:purple", "marker": ".", "markersize": 5, "alpha": 0.30, "label": "Prediction"}
Xnoisy = plotting_noise.plot_trajectory(x0, kwargs, sample_paths = 1, show_ls = False, steps = steps, xy_plane = False, ax = ax1)
for i in range(4):

        # f_simple_noise.reset()
        X_simple_noise = plotting_simple_noise.get_trajectory(x0, steps)
        if i > 0:
                kwargs["label"] = None
        ax2.plot(np.linspace(0,steps-1, steps), X_simple_noise, **kwargs)

kwargs = {"color" : "tab:red", "linewidth": 2, "label": "Predicted mean"}
X_simple = plotting_simple.get_trajectory(x0, steps = steps)
ax2.plot(np.linspace(0,steps-1, steps), X_simple, **kwargs)


kwargs = {"color" : "tab:blue", "marker": ".", "markersize": 5, "linewidth": 1, "label": "True dynamics"}
ax2.plot(X_true, **kwargs)


handles,labels = ax2.get_legend_handles_labels()
labels = labels[1::2]
labels = labels[::-1]
handles = handles[1::2]
handles = handles[::-1]
ax2.legend(handles,labels)

ax2.set_xlim([0, steps-1])
ax2.set_xlabel('Time step')
ax2.set_ylabel('$x$')
ax2.annotate('$x_1$', (0, x0.squeeze()[0].numpy()), (20, x0.squeeze()[0].numpy()), arrowprops = dict(arrowstyle='->'))
ax2.annotate('$x_2$', (0, x0.squeeze()[1].numpy()), (20, x0.squeeze()[1].numpy()), arrowprops = dict(arrowstyle='->'))



# plt.legend()
ax2.set_title('Simple model')


ax3 = plt.subplot(1,3,3)

kwargs = {"label":"Stable model"}
ax3.plot(np.mean(logp_stable,0), **kwargs)
kwargs = {"label":"Simple model"}
ax3.plot(np.mean(logp_simple,0), **kwargs)
ax3.legend()

ax3.set_xlim([0, steps-1])
ax3.set_xlabel('Time step')
ax3.set_ylabel('Average NLL')

ax3.set_title('Performance')


plt.savefig('figures/example_3.eps', dpi=400, bbox_inches='tight',pad_inches=.01, metadata='eps')
plt.savefig('figures/example_3.png', dpi=400, bbox_inches='tight',pad_inches=.01)

plt.show()
