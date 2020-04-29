import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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

        Vfx = self.V(fhatx*alpha)
        Vfx.clone().detach().requires_grad_()
        with torch.enable_grad():
            dV_da = torch.autograd.grad(Vfx, alpha, create_graph = False, grad_outputs=torch.ones_like(Vfx))[0]

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

        rootfind = rootfind_train.apply
        x_root = rootfind(self.V, self.F, fhatx, target, x)

        return x_root

    def split_rootfind(self, inputs, labels):

        fhatx = self.fhat(inputs)
        target = self.beta*self.V(inputs)
        m = (self.V(fhatx) <= target).squeeze()
        x_usual = inputs[torch.where(m)]
        labels_usual = labels[torch.where(m)]
        x_rootfind = inputs[torch.where(~m)]
        labels_rootfind = labels[torch.where(~m)]

        return x_usual, labels_usual, x_rootfind, labels_rootfind

class rootfind_train(torch.autograd.Function):

    @staticmethod
    def forward(ctx,V,F,fhatx,target,x):
        #Implementation of Newton's method combined with bisection method
        #   Newton's method is not guaranteed to converge in the case of nonconvex Lyapunov function,
        #   but the bisection method is
        ctx.V = V
        ctx.F = F

        alpha_temp = torch.ones(size = (x.shape[0], 1, 1), requires_grad = True)

        # Since V(fhatx*1) > target, we stop iterating when we get sufficiently
        #   close to within the level set
        m = (ctx.V(fhatx*alpha_temp) - target > 0.0).squeeze()
        end_1 = torch.zeros_like(alpha_temp, requires_grad = False)
        end_2 = torch.ones_like(alpha_temp, requires_grad = False)
        iter = 0

        while m.nonzero().shape[0] > 0 and iter < 100:

            a = alpha_temp[torch.where(m)].requires_grad_(True)
            fx = fhatx[torch.where(m)].requires_grad_(True)
            t = target[torch.where(m)].requires_grad_(True)
            with torch.enable_grad():
                a = ctx.F(fx,t,a) #take Newton step

            alpha_temp[torch.where(m)] = a

            #bisection method
            m1_bisec = (alpha_temp<end_1).squeeze()
            m2_bisec = (alpha_temp>end_2).squeeze()
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

                alpha_temp[torch.where(m_bisec)] = a_bisec.requires_grad_(True)

            m = (torch.abs(ctx.V(fhatx*alpha_temp) - target) > 0.0001).squeeze()

            iter += 1

        alpha = alpha_temp.clone().detach().requires_grad_(True)
        x_root = (fhatx*alpha)

        ctx.alpha = alpha

        #Since fhatx and target are inputs to the rootfinding algorithm, we backprop
        #   through them and consequently through their parameters
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
            #Calculations for implicity differentiation on the fixed point assuming alpha = F(alpha)
            Fx = F(fhatx, target, alpha)
            x_root = Fx*fhatx

            A_f = torch.autograd.grad(x_root, fhatx, create_graph=False, retain_graph = True, grad_outputs=grad_input)[0]
            A_t = torch.autograd.grad(x_root, target, create_graph=False, retain_graph = True, grad_outputs=grad_input)[0]

            # A_f = torch.autograd.grad(Fx, fhatx, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            # A_t = torch.autograd.grad(Fx, target, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            # b = torch.autograd.grad(Fx, alpha, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            # a = torch.autograd.grad(fhatx, fhatx, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(fhatx))[0]
        # a = torch.ones_like(fhatx)

        # dF_df = A_f/(1-b)
        # dF_dt = A_t/(1-b)

        #Theoretically we get b = 0 (above) at a fixed point whenever alpha is not a critical point
        #since alpha is only a critical point at 0, we can say b = 0
        dF_df = A_f
        dF_dt = A_t

        # grad_rootfind_f = grad_input*(alpha + fhatx*B)
        # grad_rootfind_t = grad_input*fhatx*dF_dt

        #we only need to differentiate w.r.t fhatx, target
        return None, None, dF_df, dF_dt, None
