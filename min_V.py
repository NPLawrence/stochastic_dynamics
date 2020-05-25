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
import lyapunov_NN as L


# optimizer = optim.SGD(f_net.parameters(), lr=learning_rate)
# optimizer = optim.RMSprop(f_net.parameters(), lr=learning_rate)




n = 3

layer_sizes = np.array([n, 25, 25, 1])
# layer_sizes = np.array([n, 50, 50, 1])

ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)

# f = model.dynamics_convex(V, n, beta = 0.99, add_state = add_state)
PATH_V = './saved_models/convex_V_Lorenz.pth'
V.load_state_dict(torch.load(PATH_V))

V_func = lambda x : V(x)
x = torch.randn((1,n))

optimizer = optim.SGD([x], lr=0.01)
for i in range(10000):
    optimizer.zero_grad()
    y = V_func(x)
    y.backward(retain_graph=True)
    optimizer.step()

print(x,y)
