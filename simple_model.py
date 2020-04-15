import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


#Some code samples taken from implementation by Manek + Kolter

#A simple proof of concept for convex Lyapunov functions
#   -no training, just defining and visualizing a stable deterministic system

#Three steps:
#   1. Create class for NN model for dynamics fhat
#   2. Create class NN model for Lyapunov function
#   3. Combine these models in a new class to create stable model




class fhat(nn.Module):
    #This is our 'nominal' model (before modifying/correcting the dynamics)
    def __init__(self, layer_sizes):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])
        self.reset_parameters()
        # logger.info(f"Initialized ICNN with {self.act} activation")

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)

        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):

        for W,b in zip(self.W, self.bias):
            z = F.linear(x, W, b)
            z = torch.tanh(z)
            # z = F.relu(z)

        return z

class dynamics_simple(nn.Module):
    #Modifies fhat via a simple scaling rule, exploiting convexity
    def __init__(self, fhat, V):
        super().__init__()

        # fhat = nn.Sequential(nn.Linear(2, 50), nn.ReLU(),
        #                     nn.Linear(50, 50), nn.ReLU(),
        #                     nn.Linear(50, 50), nn.ReLU(),
        #                     nn.Linear(50, 2))

        self.fhat = fhat
        self.V = V

    def forward(self, x):

        with torch.no_grad():
            beta = 1
            fx = self.fhat(x)*((beta*self.V(x) - F.relu(beta*self.V(x) - self.V(self.fhat(x)))) / self.V(self.fhat(x)))
            # fx = self.fhat(x)

        return fx

class dynamics_nonincrease(nn.Module):
    #Modifies fhat by ensuring 'non-expansiveness'
    #i.e. it never moves in a direction that is acute with the gradient of V at x_k
    def __init__(self, fhat, V):
        super().__init__()



        self.fhat = fhat
        self.V = V

    def forward(self, x):

        x = x.requires_grad_(True)
        fhatx = self.fhat(x)
        Vx = self.V(x)

        # gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True)[0]
        gV = torch.autograd.grad(Vx, x, create_graph=True, only_inputs=True, grad_outputs=torch.ones_like(Vx))[0]

        # fx = fhatx - F.relu((gV*(fhatx - x)).sum(dim = -1))*gV/(gV**2).sum(dim=1)[:,None]

        fx = fhatx - F.relu((gV*(fhatx - x)).sum(dim = -1, keepdim = True))*gV/(torch.norm(gV, dim = -1, keepdim = True)**2)

        # rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]

        return fx

class dynamics_rootfind(nn.Module):
    #Modifies fhat using the root-find approach
    def __init__(self, fhat, V):
        super().__init__()

        self.tol = 0.01

        self.fhat = fhat
        self.V = V

    def forward(self, x):

        # x.requires_grad_(True)
        fhatx = self.fhat(x)
        Vx = self.V(x)
        # G = Vx.backward()
        # gV = x.grad
        # print(fx.dtype)
        # print(F.relu((gV*(fx - x)).sum(dim = 1)))
        # gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True)[0]

        # fx = fhatx - F.relu((gV*(fhatx - x)).sum(dim = 1))*gV/(gV**2).sum(dim=1)[:,None]
        # rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]

        beta = 0.99

        alpha = torch.tensor([1], dtype = torch.float, requires_grad = True)
        target = beta*Vx
        g = self.V(fhatx*alpha) - target

        while (self.V(fhatx*alpha) - target) > self.tol:

            y = fhatx*alpha
            gV = torch.autograd.grad([a for a in self.V(y)], [y], create_graph=True, only_inputs=True)[0]
            alpha = alpha - (self.V(fhatx*alpha) - target)/(gV*fhatx).sum(dim = 1)

            # h = torch.autograd.grad([a for a in self.V(fhatx*alpha)], [alpha], create_graph=True, only_inputs=True)[0]
            # print((gV*fhatx).sum(dim = 1))
            # print(h)

        return fhatx*alpha

class dynamics_stochastic(nn.Module):
    #Generates stochastically stable system via root-find
    def __init__(self, fhat, V):
        super().__init__()

        self.tol = 0.001

        self.fhat = fhat
        self.V = V

    def forward(self, x):

        fhatx = self.fhat(x)
        Vx = self.V(x)
        beta = 0.999

        alpha = torch.tensor([1], dtype = torch.float, requires_grad = True)
        target = torch.distributions.Normal(beta*Vx, 0.1).sample()
        while target < 0:
            target = torch.distributions.Normal(beta*Vx, 0.1).sample()

        f_rand = torch.distributions.Normal(fhatx, 0.1).sample()
        g = self.V(f_rand*alpha) - target

        while (self.V(f_rand*alpha) - target) > self.tol:

            y = f_rand*alpha
            gV = torch.autograd.grad([a for a in self.V(y)], [y], create_graph=True, only_inputs=True)[0]
            alpha = alpha - (self.V(f_rand*alpha) - target)/(gV*f_rand).sum(dim = 1)

            # h = torch.autograd.grad([a for a in self.V(fhatx*alpha)], [alpha], create_graph=True, only_inputs=True)[0]
            # print((gV*fhatx).sum(dim = 1))
            # print(h)

        return f_rand*alpha




class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        # print(torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b))
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)

class MakePSD(nn.Module):
    def __init__(self, f, n, eps=0.01, d=1.0):
        super().__init__()
        self.f = f
        self.zero = torch.nn.Parameter(f(torch.zeros((1,1,n))), requires_grad=False)
        self.eps = eps
        self.d = d
        self.rehu = ReHU(self.d)

    def forward(self, x):

        smoothed_output = self.rehu(self.f(x) - self.zero)

        quadratic_under = self.eps*(torch.norm(x, dim = -1, keepdim = True)**2)

        return smoothed_output + quadratic_under

class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=F.relu_):
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
