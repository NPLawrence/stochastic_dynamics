import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#ICNN-based Lyapunov NN : Taken from Manek + Kolter, with some modifications
#   https://github.com/locuslab/stable_dynamics

class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d = 1):
        super().__init__()
        self.a = 1/(2*d)
        self.b = d/2

    def forward(self, x):

        return torch.max(torch.clamp(torch.sign(x)*self.a*x**2,min=0,max=self.b),x-self.b)


class MakePSD(nn.Module):
    def __init__(self, f, n, eps=0.01, d=1.0):
        super().__init__()
        self.f = f

        self.eps = eps
        self.d = d
        self.n = n
        self.rehu = ReHU(self.d)


    def forward(self, x):


        zero = self.f(torch.zeros((1,1,x.shape[-1])))
        smoothed_output = self.rehu(self.f(x) - zero)

        quadratic_under = self.eps*(torch.norm(x, dim = -1, keepdim = True)**2)

        return smoothed_output + quadratic_under


class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=ReHU()):
        super().__init__()


        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])

        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act(z)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)

        return F.linear(x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]


#Lyapunov NN proposed by Richards et al
#   https://github.com/befelix/safe_learning/blob/master/examples/utilities.py

#For simplicty, this version assumes all hidden dimensions are the same
class Lyapunov_NN(nn.Module):
    def __init__(self, f, quadratic_under = True, epsilon = 0.01):
        super().__init__()

        self.f = f
        self.quadratic_under = quadratic_under
        self.eps = epsilon

    def forward(self, x):

        if self.quadratic_under:
            return (torch.norm(self.f(x), dim = -1, keepdim = True)**2) + self.eps*(torch.norm(x, dim = -1, keepdim = True)**2)
        else:
            return F.linear(self.f(x), self.f(x))

#Feedforward NN with positive definite weights
class PD_weights(nn.Module):
    def __init__(self, layer_sizes, activation=F.relu, epsilon = 1e-6, make_convex = True):
        super().__init__()

        self.make_convex = make_convex

        self.G = nn.ParameterList([nn.Parameter(torch.Tensor(np.int(np.ceil((layer_sizes[i]+1)/2)), layer_sizes[i]))
                                   for i in range(0,len(layer_sizes)-1)])
        self.G_2 = nn.Parameter(torch.Tensor(layer_sizes[1] - layer_sizes[0], layer_sizes[0]))


        if self.make_convex:
            self.G_conv = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                       for l in layer_sizes[1:]])

        self.I_input = nn.Parameter(torch.eye(layer_sizes[0]), requires_grad = False)
        self.I = nn.Parameter(torch.eye(layer_sizes[1]), requires_grad = False)
        self.I_end = nn.Parameter(torch.eye(layer_sizes[-1]), requires_grad = False)
        self.act = activation
        self.eps = epsilon
        self.rehu = ReHU()
        self.reset_parameters()

    def reset_parameters(self):

        for G in self.G:
            nn.init.kaiming_uniform_(G, a=5**0.5)
        if self.make_convex:
            for G in self.G_conv:
                nn.init.kaiming_uniform_(G, a=5**0.5)
        nn.init.kaiming_uniform_(self.G_2, a=5**0.5)

    def forward(self, x):

        if self.make_convex:

            W1 = torch.mm(torch.transpose(self.G[0],0,1),self.G[0]) + self.eps*self.I_input
            W = torch.cat((W1, self.G_2),0)
            z = F.linear(x, W)
            z = self.rehu(z)

            for G,G_conv in zip(self.G[1:-1], self.G_conv[:-1]):
                W_conv = F.linear(x, G_conv)
                W = torch.mm(torch.transpose(self.rehu(G),0,1),self.rehu(G)) + self.eps*self.I
                z = W_conv + F.linear(z, W)
                z = self.rehu(z)

            W_conv = F.linear(x, self.rehu(self.G_conv[-1]))
            W = torch.mm(torch.transpose(self.rehu(self.G[-1]),0,1), self.rehu(self.G[-1])) + self.eps*self.I
            return W_conv + F.linear(z, W)

        else:

            W1 = torch.mm(torch.transpose(self.G[0],0,1),self.G[0]) + self.eps*self.I_input
            W = torch.cat((W1, self.G_2),0)
            z = F.linear(x, W)
            z = self.act(z)

            for G in self.G[1:-1]:

                W = torch.mm(torch.transpose(G,0,1),G) + self.eps*self.I
                z = F.linear(z, W)
                z = self.act(z)

            W = torch.mm(torch.transpose(self.G[-1],0,1), self.G[-1]) + self.eps*self.I

            return F.linear(z, W)
