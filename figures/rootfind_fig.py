import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import pylab
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

A = torch.tensor([[1, 0], [0, 1]], dtype = torch.float)
x_k = torch.tensor([5,2], dtype = torch.float)
f_k = torch.tensor([6,4], dtype = torch.float)

# List of points in x axis
XPoints     = []

# List of points in y axis
YPoints     = []

# X and Y points are from -6 to +6 varying in steps of 2
for val in np.linspace(-0.5, 10, 100):
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
        z = torch.tensor([x,y], dtype = torch.float)
        # print(z.shape)

        ZPoints[i][j] = (F.linear(F.linear(A,z),z))

# Set the x axis and y axis limits
pylab.xlim([-0.25,8.5])
pylab.ylim([-0.25,8.5])


# Set x axis label for the contour plot

# Set y axis label for the contour plot

# Create contour lines or level curves using matpltlib.pyplot module
contours = plt.contour(XPoints, YPoints, ZPoints, cmap = 'viridis')

# Display z values on contour lines
# plt.clabel(contours, inline=1, fontsize=10)


# fmt = {}
# strs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
# for l, s in zip(contours.levels, strs):
#     fmt[l] = s
#
# # Label every other level using strings
#
# plt.clabel(contours, contours.levels[], inline=True, fmt=fmt, fontsize=10)
#
# # contours.collections[::2].remove()
# for col in contours.collections:
#     print(col)

plt.scatter(0,0, c='k')
# plt.scatter(x_k[0], x_k[1])
# plt.scatter(f_k[0], f_k[1])

plt.xticks([])
plt.yticks([])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
