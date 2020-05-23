import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print('hello world')
from mpl_toolkits.mplot3d import Axes3D
import pylab
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class plot_dynamics(nn.Module):
    def __init__(self, f, V, show_mu = False, is_stochastic = False):
        super().__init__()

        self.f = f
        self.V = V
        self.show_mu = show_mu
        self.is_stochastic = is_stochastic

    def get_trajectory(self, x0, steps):

        if self.show_mu:
            mu = x0
            # y = self.f(mu)
            # X = torch.empty([steps,2,mu.squeeze().size(0)])
            X = torch.empty([steps,mu.shape[-1]])
            X[0,:] = mu.squeeze()
            # X[0,:] = torch.stack([mu.squeeze(), torch.zeros_like(mu.squeeze())])
        else:
            x = x0
            X = torch.empty([steps, x.shape[-1]])
            X[0,:] = x.squeeze()

        for i in range(steps-1):

            with torch.no_grad():

                if self.show_mu:
                    pi, normal = self.f(mu)

                    mu = torch.sum(pi.probs.view(-1,1)*normal.loc,1).view(-1,1,x0.shape[-1])
                    # mu = torch.sum(pi*normal.loc,1)
                    # print(mu)
                    # mu = y[0].unsqueeze(dim = 0)
                    # mu = y[0].view(-1,1,mu.shape[-1])
                    # var = y[1]
                    X[i+1,:] = mu

                else:
                    if self.is_stochastic:
                        x = self.f.sample(x)
                    else:
                        x = self.f(x)
                    X[i+1,:] = x

        return X.detach().numpy()


    def plot_trajectory(self, x0, kwargs, sample_paths = 1, show_ls = True, steps = 600, ax = plt):

        X_val = self.get_trajectory(x0, steps)

        if show_ls:


            x = np.arange(-5, 5.0, 0.1)
            y = np.arange(-5, 5.0, 0.1)
            # x = np.arange(-3, 8, 0.1)
            # y = np.arange(-3, 8, 0.1)
            X, Y = np.meshgrid(x, y)
            Z = np.ndarray(X.shape)

            for i in range(0, len(x)):
                for j in range(0, len(x)):
                    z = torch.tensor([[X[i][j],Y[i][j]]], dtype = torch.float)
                    Z[i][j] = (self.V(z))

            # Set the x axis and y axis limits
            # pylab.xlim([-1,6])
            # pylab.ylim([-1,3])

            # Provide a title for the contour plot



            # Create contour lines or level curves using matpltlib.pyplot module
            contours = ax.contour(X, Y, Z, linewidths = 1)

            # Display z values on contour lines
            ax.clabel(contours, inline=1, fontsize=10, fmt = '%1.0f')

        if self.show_mu:

            if x0.shape[-1]>1:
                ax.plot(X_val[:,0], X_val[:,1],**kwargs)
                ax.plot(X_val[-1,0], X_val[-1,1], color = "tab:blue", marker = '*', markersize = 10)
                # plt.scatter(X_val[:,0,0],X_val[:,0,1], color = color, s = )
                # p = ax.scatter(X, Y, c=c, s=z, cmap='viridis', vmin=0, vmax=1)
                # plt.plot(x,y2, 'o', ms=14, markerfacecolor="None", markeredgecolor='red', markeredgewidth=5)
                # plt.plot(X_val[:,0,0] + np.sqrt(X_val[:,1,0]), X_val[:,0,1] + np.sqrt(X_val[:,1,1]))
                # plt.plot(X_val[:,0,0] - np.sqrt(X_val[:,1,0]), X_val[:,0,1] - np.sqrt(X_val[:,1,1]))
                # plt.fill_between(X_val[0:20,0,0], X_val[0:20,0,0] + X_val[0:20,1,0], X_val[0:20,0,0] - X_val[0:20,1,0])
                # var = X_val[:,0,:] + X_val[:,1,:]
                # print((X_val[:,0,0]))
                # X = X_val[:, 0, :] + X_val[:, 1, :]
                # plt.fill(X_val[:,0,0] + X_val[:,1,0],  X_val[:,0,:] - X_val[:,1,:])
            else:
                ax.plot(np.linspace(0, steps-1, steps), X_val, **kwargs)

        else:
            if x0.shape[-1]>1:
                for i in range(sample_paths):
                    X_val = self.get_trajectory(x0, steps)
                    if i > 0:
                        kwargs["label"] = None
                    ax.plot(X_val[:,0],X_val[:,1], **kwargs)
                    # ax.plot(X_val[-1,0], X_val[-1,1], color = "tab:blue", marker = '*', markersize = 10)
            else:
                for i in range(sample_paths):
                    X_val = self.get_trajectory(x0, steps)
                    if i > 0:
                        kwargs["label"] = None
                    ax.plot(np.linspace(0, steps-1, steps), X_val, **kwargs)


        return X_val
        # plt.show()

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
        # zs = np.array(fun(np.ravel(X), np.ravel(Y)))
        # Z = zs.reshape(X.shape)
        #
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=.5)

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        if plot_dynamics:
            X_val = self.get_trajectory(self.f, x0)
            X_val = torch.tensor(X_val, dtype = torch.float).view((-1,1,2))
            with torch.no_grad():
                V_vals = (self.V(X_val)).squeeze()
            ax.plot3D(X_val[:,:,0],X_val[:,:,1],V_vals, 'r')
            ax.scatter(X_val[:,:,0],X_val[:,:,1],V_vals, color = 'r',)


        # ax.zaxis._axinfo["grid"]['color'] = 'w'
        # print(ax.zaxis._axinfo)
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
            # y = self.f(mu)
            # X = torch.empty([steps,2,mu.squeeze().size(0)])
            X = torch.empty([steps,mu.shape[-1]])
            X[0,:] = mu.squeeze()
            # X[0,:] = torch.stack([mu.squeeze(), torch.zeros_like(mu.squeeze())])
        else:
            x = x0
            X = torch.empty([steps, x.shape[-1]])
            X[0,:] = x.squeeze()

        with torch.no_grad():

            for i in range(steps-1):
                if self.show_mu:


                    pi, normal = self.f(mu)

                    # print(pi.probs.view(-1,1)*normal.loc)
                    # print(torch.sum(pi.probs.view(-1,1)*normal.loc,1))
                    mu = torch.sum(pi.probs.view(-1,1)*normal.loc,1).view(-1,1,x0.shape[-1])
                    # mu = torch.sum(pi*normal.loc,1)
                    # print(mu)
                    # mu = y[0].unsqueeze(dim = 0)
                    # mu = y[0].view(-1,1,mu.shape[-1])
                    # var = y[1]
                    X[i+1,:] = mu

                else:
                    if self.is_stochastic:
                        x = self.f.sample(x)
                    else:
                        x = self.f(x)
                    X[i+1,:] = x

        return X.detach().numpy()


    def plot_trajectory(self, x0, kwargs, sample_paths = 1, steps = 200):

        # rho = 28.0
        # sigma = 10.0
        # beta = 8.0 / 3.0
        # fig = plt.figure()
        # fig.gca(projection='3d')

        for i in range(sample_paths):
            X_val = self.get_trajectory(x0, steps)
            if i > 0:
                kwargs["label"] = None

            # plt.draw()
            plt.plot(X_val[:, 0], X_val[:, 1], X_val[:, 2], **kwargs)

        return X_val
