import numpy as np
import matplotlib.pyplot as plt

import pylab

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import simple_model as model

layer_sizes = np.array([2, 100, 1])

fhat = model.fhat(np.array([2, 50, 50, 2]))
V = model.MakePSD(model.ICNN(layer_sizes),2)
# f = model.dynamics_simple(fhat,V)
# f = model.dynamics_nonincrease(fhat,V)
# f = model.dynamics_rootfind(fhat,V)
f = model.dynamics_stochastic(fhat,V)


def get_trajectory(f, x0, steps):

    X = torch.empty([steps,x0.size(1)])
    x = x0
    X[0,:] = x

    for i in range(steps-1):

        x = f(x)
        X[i+1,:] = x

    return X.detach().numpy()

X = get_trajectory(f, 3.5*torch.randn([1,2], dtype = torch.float), 200)


# List of points in x axis
XPoints     = []

# List of points in y axis
YPoints     = []

# X and Y points are from -6 to +6 varying in steps of 2
for val in np.linspace(-5, 5, 60):
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
        ZPoints[i][j] = (V(z))

# Set the x axis and y axis limits
pylab.xlim([-5,5])
pylab.ylim([-5,5])

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
plt.show()
