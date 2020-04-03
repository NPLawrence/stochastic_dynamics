import numpy as np
import matplotlib.pyplot as plt

import pylab

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import simple_model as model

layer_sizes = np.array([2, 100, 1])

V = model.ICNN(layer_sizes)

print(V(torch.randn(1,2)))


# # List of points in x axis
# XPoints     = []
#
# # List of points in y axis
# YPoints     = []
#
# # X and Y points are from -6 to +6 varying in steps of 2
# for val in range(-6, 8, 2):
#     XPoints.append(val)
#     YPoints.append(val)
#
# # Z values as a matrix
# ZPoints     = np.ndarray((7,7))
#
# # Populate Z Values (a 7x7 matrix) - For a circle x^2+y^2=z
# for x in range(0, len(XPoints)):
#     for y in range(0, len(YPoints)):
#         # z = np.array([x,y])
#         z = torch.tensor([x,y])
#         ZPoints[x][y] = (V.forward(z))
#
# # Print x,y and z values
# print(XPoints)
# print(YPoints)
# print(ZPoints)
#
# # Set the x axis and y axis limits
# pylab.xlim([-10,10])
# pylab.ylim([-10,10])
#
# # Provide a title for the contour plot
# plt.title('Contour plot')
#
# # Set x axis label for the contour plot
# plt.xlabel('X')
#
# # Set y axis label for the contour plot
# plt.ylabel('Y')
#
# # Create contour lines or level curves using matpltlib.pyplot module
# contours = plt.contour(XPoints, YPoints, ZPoints)
#
# # Display z values on contour lines
# plt.clabel(contours, inline=1, fontsize=10)
#
# # Display the contour plot
# plt.show()
