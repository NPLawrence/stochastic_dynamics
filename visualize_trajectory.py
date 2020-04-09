import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import pylab
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import simple_model as model

class plot_dynamics(nn.Module):
    #Generates stochastically stable system via root-find
    def __init__(self, f, V, x0):
        super().__init__()

        self.f = f
        self.V = V
        self.x0 = x0


    def get_trajectory(self, f, steps = 50):

        X = torch.empty([steps,self.x0.size(1)])
        x = self.x0
        X[0,:] = x

        for i in range(steps-1):

            x = self.f(x)
            X[i+1,:] = x

        return X.detach().numpy()


    def plot_trajectory(self):

        X = self.get_trajectory(self.f)

        # List of points in x axis
        XPoints     = []

        # List of points in y axis
        YPoints     = []

        # X and Y points are from -6 to +6 varying in steps of 2
        for val in np.linspace(-4, 4, 100):
            XPoints.append(val)
            YPoints.append(val)

        # Z values as a matrix
        ZPoints     = np.ndarray((len(XPoints),len(YPoints)))

        # Populate Z Values (a 7x7 matrix) - For a circle x^2+y^2=z
        for i in range(0, len(XPoints)):
            for j in range(0, len(YPoints)):
                # z = np.array([x,y])
                x = XPoints[i]
                y = YPoints[j]
                z = torch.tensor([[x,y]], dtype = torch.float)
                ZPoints[i][j] = (self.V(z))

        # Set the x axis and y axis limits
        pylab.xlim([-4,4])
        pylab.ylim([-4,4])

        # Provide a title for the contour plot
        plt.title('Contour plot')

        # Set x axis label for the contour plot
        plt.xlabel('X')

        # Set y axis label for the contour plot
        plt.ylabel('Y')

        # Create contour lines or level curves using matpltlib.pyplot module
        contours = plt.contour(XPoints, YPoints, ZPoints)

        # Display z values on contour lines
        plt.clabel(contours, inline=1, fontsize=10)

        plt.plot(X[:,0],X[:,1], 'r')
        plt.plot(X[-1,0], X[-1,1], "b*")

        # Display the contour plot
        # plt.show()
