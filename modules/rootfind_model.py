import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

# Stabilizing strategy for non-convex Lyapunov functions (implicit method)

class newton_iter(nn.Module):
    """
    Newton iteration function. One step of Newton's method for finding gamma i.e.
        This is the function x - f(x)/f'(x) in Newton's method in terms of fhat and V wrt gamma.

    fhat : nominal model
    V : Lyapunov neural network
    """
    def __init__(self, fhat, V):
        super().__init__()

        self.fhat = fhat
        self.V = V

    def forward(self, fhatx, target, gamma, backprop = False):

        Vfx = self.V(fhatx*gamma) - target
        Vfx.clone().detach().requires_grad_()
        with torch.enable_grad():
            dV_da = torch.autograd.grad(Vfx, gamma, create_graph = backprop, grad_outputs=torch.ones_like(Vfx))[0]

        F = gamma - (self.V(fhatx*gamma) - target)/dV_da

        return F

class dynamics_model(nn.Module):
    """
    This is where we bring together:
        1. The Newton iteration function (newton_iter)
        2. The custom autograd function for backprop though Newton's method (rootfind_train)

    V : Lyapunov neural network
    n : state dimension
    is_training : binary variable. When True, this triggers a data management process.
    beta : number in (0,1] in the stability criterion V(x') <= beta V(x)
    f : optional user-defined nominal model.
    """
    def __init__(self, V, n, is_training = False, beta = 0.99, f = None):
        super().__init__()

        if f is None:
            self.fhat =  nn.Sequential(nn.Linear(n, 25), nn.Softplus(),
                                        nn.Linear(25, 25), nn.Softplus(),
                                        nn.Linear(25, n))
        else:
            self.fhat = f

        self.V = V
        self.beta = beta
        self.is_training = is_training

        self.F = newton_iter(self.fhat, self.V)

    def forward(self, x):

        if self.is_training:
            y = torch.empty_like(x)
            x_usual, x_rootfind, m = self.split_rootfind(x)
            rootfind = rootfind_train.apply
            fhatx = self.fhat(x_rootfind)
            target = self.beta*self.V(x_rootfind)
            x_root = rootfind(self.V, self.F, fhatx, target, x_rootfind)
            y[torch.where(m)] = self.fhat(x_usual)
            y[torch.where(~m)] = x_root
            return y
        else:
            fhatx = self.fhat(x)
            target = self.beta*self.V(x)
            rootfind = rootfind_train.apply
            x_root = rootfind(self.V, self.F, fhatx, target, x)
            return x_root

    def split_rootfind(self, inputs):

        fhatx = self.fhat(inputs)
        target = self.beta*self.V(inputs)
        m = (self.V(fhatx) <= target).squeeze()

        x_usual = inputs[torch.where(m)]
        x_rootfind = inputs[torch.where(~m)]

        return x_usual, x_rootfind, m


class rootfind_train(torch.autograd.Function):
    """
    Performs forward and backward pass of implicit dynamics model by incorporating an
        implementation of Newton's method combined with bisection method.
        Newton's method is not guaranteed to converge in the case of nonconvex Lyapunov function,
        but the bisection method is. We use the bisection method for backup iterations
        when Newton's method iteration moves outside the current bisection method interval.

    V : Lyapunov neural network
    F : newton_iter
    fhatx : nominal model evaluated at x
    target : value of V to be less than or equal to e.g. beta*V(x).
        In order to train V it is important that target depends on V.
    x : current state
    """
    @staticmethod
    def forward(ctx,V,F,fhatx,target,x):

        ctx.V = V
        ctx.F = F

        tol = 0.0001

        gamma_temp = torch.ones(size = (x.shape[0], 1, 1), requires_grad = True)

        # Since V(fhatx*1) > target, we stop iterating when we get sufficiently
        #   close to the level set
        m = (ctx.V(fhatx) - target > 0.0).squeeze()
        end_1 = torch.zeros_like(gamma_temp, requires_grad = False)
        end_2 = torch.ones_like(gamma_temp, requires_grad = False)
        iter = 0

        while m.nonzero().shape[0] > 0 and iter < 1000:

            a = gamma_temp[torch.where(m)].requires_grad_(True)
            fx = fhatx[torch.where(m)].requires_grad_(True)
            t = target[torch.where(m)].requires_grad_(True)
            with torch.enable_grad():
                a = ctx.F(fx,t,a) #take Newton step
            gamma_temp[torch.where(m)] = a
            #bisection method
            m1_bisec = (gamma_temp<end_1).squeeze()
            m2_bisec = (gamma_temp>end_2).squeeze()
            m_bisec = ((m1_bisec + m2_bisec) > 0)
            if m_bisec.nonzero().shape[0] > 0: #check if bisection is necessary
                a_bisec = end_1[torch.where(m_bisec)] + (end_2[torch.where(m_bisec)] - end_1[torch.where(m_bisec)])/2
                fx_bisec = fhatx[torch.where(m_bisec)]
                t_bisec = target[torch.where(m_bisec)]
                end1_temp = end_1[torch.where(m_bisec)]
                end2_temp = end_2[torch.where(m_bisec)]

                m_end2 = (np.sign(ctx.V(fx_bisec*a_bisec) - t_bisec)*np.sign(ctx.V(fx_bisec*end1_temp) - t_bisec) < 0).squeeze()
                m_end1 = (np.sign(ctx.V(fx_bisec*a_bisec) - t_bisec)*np.sign(ctx.V(fx_bisec*end2_temp) - t_bisec) < 0).squeeze()

                end_1[torch.where(m_end2)] = end_1[torch.where(m_end2)]
                end_2[torch.where(m_end2)] = a_bisec[torch.where(m_end2)]

                end_1[torch.where(m_end1)] = a_bisec[torch.where(m_end1)]
                end_2[torch.where(m_end1)] = end_2[torch.where(m_end1)]

                gamma_temp[torch.where(m_bisec)] = a_bisec.requires_grad_(True)


            m = (torch.abs(ctx.V(fhatx*gamma_temp) - target) > tol).squeeze()
            iter += 1

        gamma = gamma_temp.clone().detach().requires_grad_(True)
        x_root = (fhatx*gamma)

        ctx.gamma = gamma

        #Since fhatx and target are inputs to the rootfinding algorithm, we backprop
        #   through them and consequently through their parameters
        ctx.save_for_backward(fhatx, target, x_root)

        return x_root

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.clone()

        fhatx, target, x_root = ctx.saved_tensors

        gamma = ctx.gamma
        V = ctx.V
        F = ctx.F

        with torch.enable_grad():

            Fx = F(fhatx, target, gamma, backprop = True) # TODO: We assume we are close to the root when differentiating -- ensure iterate is `stable`
            dF_df = torch.autograd.grad(Fx, fhatx, create_graph=True, grad_outputs=torch.ones_like(Fx))[0]
            dF_dt = torch.autograd.grad(Fx, target, create_graph=False, grad_outputs=torch.ones_like(Fx))[0]

        grad_rootfind_f = Fx*grad_input + torch.bmm(grad_input, torch.bmm(torch.transpose(fhatx,1,2),dF_df))
        grad_rootfind_t = torch.bmm(grad_input, torch.transpose(fhatx,1,2))*dF_dt

        return None, None, grad_rootfind_f, grad_rootfind_t, None
