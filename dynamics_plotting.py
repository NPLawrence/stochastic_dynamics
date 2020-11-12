import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D
import pylab
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class plot_dynamics(nn.Module):
    def __init__(self, f, V, reset_model = False):
        super().__init__()

        self.f = f
        self.V = V
        self.reset_model = reset_model

    def get_trajectory(self, x0, steps):

            x = x0
            X = torch.empty([steps,x.shape[-1]])
            X[0,:] = x.squeeze()

            for i in range(steps-1):

                with torch.no_grad():

                    x = self.f(x)
                    X[i+1,:] = x

            return X.detach().numpy()


    def plot_trajectory(self, x0, kwargs, sample_paths = 1, show_ls = True, steps = 600, xy_plane = True, ax = plt):

        if show_ls:

            x = np.arange(-20.0, 20.0, 0.25)
            y = np.arange(-20.0, 20.0, 0.25)

            X, Y = np.meshgrid(x, y)
            Z = np.ndarray(X.shape)

            for i in range(0, len(x)):
                for j in range(0, len(x)):
                    z = torch.tensor([[X[i][j],Y[i][j]]], dtype = torch.float)
                    Z[i][j] = (self.V(z))


            # Create contour lines or level curves using matpltlib.pyplot module
            contours = ax.contour(X, Y, Z, linewidths = 1)

            # Display z values on contour lines
            ax.clabel(contours, inline=1, fontsize=10, fmt = '%1.2f')

        if xy_plane:
            for i in range(sample_paths):
                if self.reset_model:
                    self.f.reset()
                X_val = self.get_trajectory(x0, steps)
                if i > 0:
                    kwargs["label"] = None
                ax.plot(X_val[:,0],X_val[:,1], **kwargs)
                if i==0:
                    ax.plot(X_val[-1,0], X_val[-1,1], color = "tab:blue", marker = '*', markersize = 10)
        else:
            for i in range(sample_paths):
                X_val = self.get_trajectory(x0, steps = steps)
                if i > 0:
                    kwargs["label"] = None
                ax.plot(X_val, **kwargs)

        return X_val

    def surface_plot(self, x0, plot_dynamics = True):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = y = np.arange(-1.0, 1.0, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = np.ndarray(X.shape)

        for i in range(0, len(x)):
            for j in range(0, len(x)):
                z = torch.tensor([[X[i][j],Y[i][j]]], dtype = torch.float)
                Z[i][j] = (self.V(z))

        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=.5)


        if plot_dynamics:
            X_val = self.get_trajectory(self.f, x0)
            X_val = torch.tensor(X_val, dtype = torch.float).view((-1,1,2))
            with torch.no_grad():
                V_vals = (self.V(X_val)).squeeze()
            ax.plot3D(X_val[:,:,0],X_val[:,:,1],V_vals, 'r')
            ax.scatter(X_val[:,:,0],X_val[:,:,1],V_vals, color = 'r',)

        ax.grid(False)
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.zaxis.set_ticks([])
        plt.show()



class plot_dynamics_3D(nn.Module):
    def __init__(self, f, V, show_mu = False, is_stochastic = False):
        super().__init__()

        self.f = f
        self.V = V
        self.show_mu = show_mu
        self.is_stochastic = is_stochastic

    def get_trajectory(self, x0, steps):

        if self.show_mu:
            mu = x0

            X = torch.empty([steps,mu.shape[-1]])
            X[0,:] = mu.squeeze()
        else:
            x = x0
            X = torch.empty([steps, x.shape[-1]])
            X[0,:] = x.squeeze()

        for i in range(steps-1):
            if self.show_mu:

                pi, normal = self.f(mu)
                mu = torch.sum(pi.probs.view(-1,1)*normal.loc,1).view(-1,1,x0.shape[-1])
                X[i+1,:] = mu

            else:
                if self.is_stochastic:
                    x = self.f.sample(x)
                else:
                    x = self.f(x)
                X[i+1,:] = x

        return X.detach().numpy()


    def plot_trajectory(self, x0, kwargs, sample_paths = 1, steps = 200):

        for i in range(sample_paths):
            X_val = self.get_trajectory(x0, steps)
            if i > 0:
                kwargs["label"] = None

            plt.plot(X_val[:, 0], X_val[:, 1], X_val[:, 2], **kwargs)

        return X_val
