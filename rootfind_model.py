import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class newton_iter(nn.Module):
    #This is the function x - f(x)/f'(x) in Newton's method in terms of fhat and V wrt alpha
    def __init__(self, fhat, V):
        super().__init__()

        self.fhat = fhat
        self.V = V

    def forward(self, fhatx, target, alpha):

        alpha = alpha.requires_grad_(True)
        fhatx.requires_grad_(True)
        target.requires_grad_(True)
        Vfx = self.V(fhatx*alpha)

        with torch.enable_grad():
            dV_da = torch.autograd.grad(Vfx, alpha, create_graph = True, grad_outputs=torch.ones_like(Vfx))[0]

        F = alpha - (self.V(fhatx*alpha) - target)/dV_da

        return F


class rootfind_module(nn.Module):
    #This is where we bring together:
    #   1. The Newton iteration function (newton_iter)
    #   2. The custom autograd function for backprop though Newton's method (rootfind_train)
    def __init__(self,fhat, V, beta = 0.99):
        super().__init__()

        self.V = V
        self.fhat = fhat
        self.beta = beta
        self.F = newton_iter(self.fhat, self.V)

    def forward(self,x):

        fhatx = self.fhat(x)
        target = self.beta*self.V(x)

        #If the nominal model decreases in V, we train fhat with the 'usual' backprop
        #    V does not get updated
        #   Otherwise, apply rootfind_train

        if self.V(fhatx) <= target:
            x_root = fhatx
        else:
            rootfind = rootfind_train.apply
            x_root = rootfind(self.V, self.F, fhatx, target, x)

        return x_root

class rootfind_train(torch.autograd.Function):

    @staticmethod
    def forward(ctx,V,F,fhatx,target,x):
        ctx.V = V
        ctx.F = F

        alpha = torch.tensor([1], dtype = torch.float, requires_grad = True)
        x_root = (fhatx*alpha)

        #Since V(fhatx*1) > target, we stop iterating when we get sufficiently
        #   close to within the level set
        while (V(fhatx*alpha) - target) > 0.001:

            with torch.enable_grad():
                alpha = F(fhatx,target,alpha)

            x_root = fhatx*alpha

        ctx.alpha = alpha


        #Since fhatx and target are inputs to the rootfinding algorithm, we backprop
        #   through themand consequently through their parameters
        #   and consequently through their parameters
        ctx.save_for_backward(fhatx, target, x_root)

        return x_root

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.clone()

        fhatx, target, x_root = ctx.saved_tensors

        alpha = ctx.alpha
        V = ctx.V
        F = ctx.F

        with torch.enable_grad():

            # alpha = F(alpha) F depends on fhat, V
            # (1-p_F/p_alpha)*p_alpha/p_fhatx = p_F/p_fhatx
            # (1-p_F/p_alpha)*p_alpha/p_target = p_F/p_target

            Fx = F(fhatx, target, alpha)
            A_f = torch.autograd.grad(Fx, fhatx, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            A_t = torch.autograd.grad(Fx, target, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            b = torch.autograd.grad(Fx, alpha, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            a = torch.autograd.grad(fhatx, fhatx, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(fhatx))[0]

            # p_alpha*fhatx / p_fhatx

        dF_df = A_f/(1-b)
        dF_dt = A_t/(1-b)

        # loss wrt alpha*fhatx

        grad_rootfind_f = grad_input*(a*alpha + fhatx*dF_df)
        grad_rootfind_t = grad_input*fhatx*dF_dt

        return None, None, grad_rootfind_f, grad_rootfind_t, None

class rootfind_alg(torch.autograd.Function):
#A basic implementation of Newton's method

    @staticmethod
    def forward():
        gV = torch.zeros_like(x)
        while (V(fhatx*alpha) - target) > 0.01:

            # print('hello')

            # print(y.requires_grad)

            with torch.enable_grad():
                y = fhatx*alpha
                y.requires_grad_(True)
                gV = torch.autograd.grad(V(y), y, create_graph = False, allow_unused = True, only_inputs=True)[0]
            # print(gV)
            # print(alpha)
            # gV = torch.autograd.grad([a for a in V(y)], [y], create_graph=True, only_inputs=True)[0]
            alpha = alpha - (V(fhatx*alpha) - target)/((gV*fhatx).sum(dim = -1))
            x_root = fhatx*alpha


    @staticmethod
    def get_root(V,f,target,x):

        fhatx = f(x)
        alpha = torch.tensor([1], dtype = torch.float, requires_grad = True)

        while (V(fhatx*alpha) - target) > 0.001:

            y = fhatx*alpha
            gV = torch.autograd.grad([a for a in V(y)], [alpha], create_graph=True, only_inputs=True)[0]
            # gV = torch.autograd.grad([a for a in V(y)], [y], create_graph=True, only_inputs=True)[0]
            alpha = (alpha - (V(fhatx*alpha) - target)/(gV*fhatx).sum(dim = 1))

        x_root = alpha

        return x_root
