import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#A simple proof of concept for convex Lyapunov functions
#   -no training, just defining and visualizing a stable deterministic system

#Three steps:
#   1. Create class for NN model for dynamics fhat
#   2. Create class NN model for Lyapunov function
#   3. Combine these models in a new class to create stable model

class fhat(nn.Module):
    #This is our 'nominal' model (before modifying/correcting the dynamics)
    def __init__(self, layer_sizes, add_state = False):
        super().__init__()

        self.add_state = add_state
        layers = []
        for i in range(len(layer_sizes)-2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.fhat = nn.Sequential(*layers)
        # self.fhat = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
        #                     nn.Linear(25, 25), nn.ReLU(),
        #                     nn.Linear(25, 25), nn.ReLU(),
        #                     nn.Linear(25, n))

    # def reset_parameters(self):
    #     # copying from PyTorch Linear
    #     for W in self.W:
    #         nn.init.kaiming_uniform_(W, a=5**0.5)
    #
    #     for i,b in enumerate(self.bias):
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
    #         bound = 1 / (fan_in**0.5)
    #         nn.init.uniform_(b, -bound, bound)

    def forward(self, x):

        if self.add_state:
            z = x + self.fhat(x)
        else:
            z = self.fhat(x)

        return z

class dynamics_convex(nn.Module):
    #Modifies fhat via a simple scaling rule, exploiting convexity
    def __init__(self, V, n, beta = 0.99, add_state = False, fhat = None):
        super().__init__()

        # fhat = nn.Sequential(nn.Linear(2, 50), nn.ReLU(),
        #                     nn.Linear(50, 50), nn.ReLU(),
        #                     nn.Linear(50, 50), nn.ReLU(),
        #                     nn.Linear(50, 2))
        if fhat is None:
            self.fhat = fhat(np.array([n, 25, 25, 25, n]), False)
        else:
            self.fhat = fhat
        self.V = V
        self.beta = beta
        self.add_state = add_state

    def forward(self, x):
        # with torch.no_grad():

        if self.add_state:
            fx = x + self.fhat(x)*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(self.fhat(x)))) / self.V(self.fhat(x)))
            # print(self.V(fx - x))
        else:

            fx = self.fhat(x)*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(self.fhat(x)))) / self.V(self.fhat(x)))
        return fx

# class dynamics_convex(nn.Module):
#     #Modifies fhat via a simple scaling rule, exploiting convexity
#     def __init__(self, V, n, beta = 0.99, add_state = False, is_training = False, is_initial = True):
#         super().__init__()
#
#         # fhat = nn.Sequential(nn.Linear(2, 50), nn.ReLU(),
#         #                     nn.Linear(50, 50), nn.ReLU(),
#         #                     nn.Linear(50, 50), nn.ReLU(),
#         #                     nn.Linear(50, 2))
#
#         self.fhat = fhat(np.array([n, 25, 25, 25, n]), False)
#         # self.x_inf = nn.Sequential(nn.Identity(n,1))
#         self.p = torch.nn.init.normal_(nn.Parameter(torch.zeros(1,n)), mean=0.0, std=1)
#         self.x_inf = nn.Identity()
#         self.V = V
#         self.beta = beta
#         self.add_state = add_state
#         self.is_training = is_training
#         self.is_initial = is_initial
#         # self.target = 10000*torch.ones((1,1,n)) #used for add_state method
#                                                #    just make sure this value is sufficiently large
#
#     # def forward(self, x):
#     #     # with torch.no_grad():
#     #     if self.add_state:
#     #         if self.is_training:
#     #
#     #
#     #             state = x - self.p
#     #             target = self.beta*self.V(state)
#     #             fx = self.fhat(state)*((target - F.relu(target - self.V(self.fhat(state)))) / self.V(self.fhat(state)))
#     #             self.p = self.x_inf(self.p)
#     #             fx = fx + self.p
#     #
#     #         else:
#     #             if self.is_initial:
#     #                 # self.target = self.beta*self.V(x)
#     #                 self.dif = 10000*torch.ones((1,1,3))
#     #                 self.target = self.beta*self.V(self.dif)
#     #                 self.is_initial = False
#     #
#     #             fx = x + self.fhat(x)*((self.target - F.relu(self.target - self.V(self.fhat(x)))) / self.V(self.fhat(x)))
#     #             # print(self.V(self.fhat(x)))
#     #             self.dif = fx - x
#     #             self.target = self.beta*self.V(self.dif)
#     #     else:
#     #         fx = self.fhat(x)*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(self.fhat(x)))) / self.V(self.fhat(x)))
#     #     # print(self.p)
#     #     return fx
#
#     def forward(self, x):
#         # with torch.no_grad():
#         if self.add_state:
#             if self.is_training:
#                 input = x[:,:,:x.shape[-1]//2]
#                 # input = x[:,:,x.shape[-1]//2:]
#                 # target1 = self.beta*self.V(input)
#                 dif1 = self.fhat(input)
#                 target = self.beta*self.V(dif1)
#                 output1 = dif1 + input
#
#                 dif2 = self.fhat(output1)*((target - F.relu(target - self.V(self.fhat(output1)))) / self.V(self.fhat(output1)))
#                 fx = dif2 + output1
#
#
#
#                 # input = x[:]
#                 # state = input - target_val
#                 # target = self.beta*self.V(state)
#                 # fx = input + self.fhat(state)*((target - F.relu(target - self.V(self.fhat(state)))) / self.V(self.fhat(state)))
#                 # target_val = x[:,:,:x.shape[-1]//2]
#                 # input = x[:,:,x.shape[-1]//2:]
#                 #
#                 # x0 = target_val[:,:,0]
#                 # x1 = target_val[:,:,1]
#                 # # input = x[:]
#                 # state = input - target_val
#                 # target = self.beta*self.V(state)
#                 # fx = input + self.fhat(state)*((target - F.relu(target - self.V(self.fhat(state)))) / self.V(self.fhat(state)))
#
#             else:
#                 if self.is_initial:
#                     # self.target = self.beta*self.V(x)
#                     self.state = self.fhat(x)
#                     fx = self.state + x
#                     self.target = self.beta*self.V(self.state)
#                     self.is_initial = False
#                 else:
#
#                     self.state = self.fhat(self.state)*((self.target - F.relu(self.target - self.V(self.fhat(self.state)))) / self.V(self.fhat(self.state)))
#                     # print(self.dif)
#                     fx = self.state +  x
#                     self.target = self.beta*self.V(self.state)
#         else:
#             fx = self.fhat(x)*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(self.fhat(x)))) / self.V(self.fhat(x)))
#
#         return fx

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
        fhatx.requires_grad_(True)
        Vx = self.V(x)
        # print(Vx.backward(torch.ones_like(Vx)))
        # gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True)[0]
        with torch.enable_grad():
            gV = torch.autograd.grad(Vx, x, create_graph=True, only_inputs=True, grad_outputs=torch.ones_like(Vx))[0]
        # fx = fhatx - F.relu((gV*(fhatx - x)).sum(dim = -1))*gV/(gV**2).sum(dim=1)[:,None]

        # print(gV.requires_grad)
        fx = self.fhat(x) - F.relu((gV*(self.fhat(x) - x)).sum(dim = -1, keepdim = True))*gV/(torch.norm(gV, dim = -1, keepdim = True)**2)

        # rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]
        return fx

class dynamics_rootfind(nn.Module):
    #Modifies fhat using the root-find approach
    #Not trainable! See rootfind_model.py
    def __init__(self, fhat, V):
        super().__init__()

        self.tol = 0.01

        self.fhat = fhat
        self.V = V

    def forward(self, x):

        # get_root = rootfind_module.rootfind_alg.get_root

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
        # g = self.V(fhatx*alpha) - target

        # root = get_root(self.V,self.fhat,target,x)
        # rootfind = rootfind_module.rootfind_train.apply

        # V,fhat,target,x
        # x_root = rootfind(self.V,self.fhat,target,x)

        # x_root = rootfind_module(self.V,self.fhat,target,x)
        while (self.V(fhatx*alpha) - target) > self.tol:


            y = fhatx*alpha
            # print(y.requires_grad)
            # h = torch.autograd.grad(self.V(y), alpha, create_graph=True, only_inputs=True)[0]
            gV = torch.autograd.grad(self.V(y), y, create_graph=True, only_inputs=True)[0]
            # print((gV*fhatx).sum(dim = -1))
            # print(h, ((gV*fhatx).sum(dim = -1)))
            alpha = alpha - (self.V(fhatx*alpha) - target)/((gV*fhatx).sum(dim = -1))


            # h = torch.autograd.grad([a for a in self.V(fhatx*alpha)], [alpha], create_graph=True, only_inputs=True)[0]
            # print((gV*fhatx).sum(dim = 1))
            # print(h)
        # print(alpha)
        x_root = fhatx*alpha

        return x_root

    # @staticmethod
    # def backward(ctx, grad_output):
    #
    #     input, = ctx.saved_tensors
    #     grad_input = grad_output.clone()




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
