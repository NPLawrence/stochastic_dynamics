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
    #Generates stochastically stable system via root-find
    def __init__(self, f, V):
        super().__init__()

        self.f = f
        self.V = V

    def get_trajectory(self, x0, steps, show_mu = False):

        if show_mu:
            mu = x0
            y = self.f(mu)
            X = torch.empty([steps,2,mu.squeeze().size(0)])
            X[0,:] = torch.stack([mu.squeeze(), torch.zeros_like(mu.squeeze())])
        else:
            x = x0
            X = torch.empty([steps,x.squeeze().size(0)])
            X[0,:] = x.squeeze()

        for i in range(steps-1):

            with torch.no_grad():

                if show_mu:
                    y = self.f(mu)
                    mu = y[0].unsqueeze(dim = 0)
                    var = y[1]
                    X[i+1,:] = y

                else:
                    x = self.f(x)
                    X[i+1,:] = x

        return X.detach().numpy()


    def plot_trajectory(self, x0, kwargs, sample_paths = 1, show_ls = True, show_mu = False, steps = 600, ax = plt):

        X_val = self.get_trajectory(x0, steps, show_mu)

        if show_ls:

            x = np.arange(-2, 18.0, 0.05)
            y = np.arange(-2, 18.0, 0.05)
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

        if show_mu:

            ax.plot(X_val[:,0,0], X_val[:,0,1],**kwargs)
            ax.plot(X_val[-1,0,0], X_val[-1,1,0], color = "tab:blue", marker = '*', markersize = 10)
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
            for i in range(sample_paths):
                X_val = self.get_trajectory(x0, steps, show_mu)
                if i > 0:
                    kwargs["label"] = None
                ax.plot(X_val[:,0],X_val[:,1], **kwargs)
                # ax.plot(X_val[-1,0], X_val[-1,1], color = "tab:blue", marker = '*', markersize = 10)


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
