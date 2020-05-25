import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


class newton_iter(nn.Module):
    #This is the function x - f(x)/f'(x) in Newton's method in terms of fhat and V wrt alpha
    def __init__(self, fhat, V):
        super().__init__()

        self.fhat = fhat
        self.V = V

    def forward(self, fhatx, target, alpha, is_backprop):

        Vfx = self.V(fhatx*alpha) - target
        Vfx.clone().detach().requires_grad_()
        with torch.enable_grad():
            dV_da = torch.autograd.grad(Vfx, alpha, create_graph = False, retain_graph = True, grad_outputs=torch.ones_like(Vfx))[0]
            if is_backprop:
                dV_df = torch.autograd.grad(Vfx, fhatx, create_graph = False, retain_graph = True, grad_outputs=torch.ones_like(Vfx))[0]
            # dV_dt = torch.autograd.grad(Vfx, target, create_graph = False, retain_graph = True, grad_outputs=torch.ones_like(Vfx))[0]
        F = alpha - (self.V(fhatx*alpha) - target)/dV_da
        if is_backprop:
            return F, dV_da, dV_df
        else:
            return F

class rootfind_module(nn.Module):
    #This is where we bring together:
    #   1. The Newton iteration function (newton_iter)
    #   2. The custom autograd function for backprop though Newton's method (rootfind_train)
    def __init__(self, fhat, V, is_training = False, beta = 0.99):
        super().__init__()

        self.V = V
        self.fhat = fhat
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
        # labels_usual = labels[torch.where(m)]
        x_rootfind = inputs[torch.where(~m)]
        # labels_rootfind = labels[torch.where(~m)]

        # return x_usual, labels_usual, x_rootfind, labels_rootfind
        return x_usual, x_rootfind, m


class rootfind_train(torch.autograd.Function):

    @staticmethod
    def forward(ctx,V,F,fhatx,target,x):
        #Implementation of Newton's method combined with bisection method
        #   Newton's method is not guaranteed to converge in the case of nonconvex Lyapunov function,
        #   but the bisection method is
        ctx.V = V
        ctx.F = F

        # tol = torch.clamp(target*((1-0.99)/0.99), max = 0.001)
        tol = 0.001

        alpha_temp = torch.ones(size = (x.shape[0], 1, 1), requires_grad = True)

        # Since V(fhatx*1) > target, we stop iterating when we get sufficiently
        #   close to within the level set
        m = (ctx.V(fhatx) - target > 0.0).squeeze()
        end_1 = torch.zeros_like(alpha_temp, requires_grad = False)
        end_2 = torch.ones_like(alpha_temp, requires_grad = False)
        iter = 0

        # print(torch.nonzero(m, as_tuple = True)[0].shape[0])
        while m.nonzero().shape[0] > 0 and iter < 100:

            a = alpha_temp[torch.where(m)].requires_grad_(True)
            fx = fhatx[torch.where(m)].requires_grad_(True)
            t = target[torch.where(m)].requires_grad_(True)
            with torch.enable_grad():
                a = ctx.F(fx,t,a,False) #take Newton step

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

            m = (torch.abs(ctx.V(fhatx*alpha_temp) - target) > tol).squeeze()

            iter += 1

        alpha = alpha_temp.clone().detach().requires_grad_(True)
        x_root = (fhatx*alpha)

        ctx.alpha = alpha

        #Since fhatx and target are inputs to the rootfinding algorithm, we backprop
        #   through them and consequently through their parameters
        ctx.save_for_backward(fhatx, target, x_root)

        return x_root


    # @staticmethod
    # def backward(ctx, grad_output):
    #
    #     grad_input = grad_output.clone()
    #
    #     fhatx, target, x_root = ctx.saved_tensors
    #
    #     alpha = ctx.alpha
    #     V = ctx.V
    #     F = ctx.F
    #
    #
    #     with torch.enable_grad():
    #
    #         Fx, dF_da, dF_df = F(fhatx, target, alpha, True)
    #         # A_f = torch.bmm(torch.transpose(fhatx,1,2), (-1/dF_da)*dF_df)
    #         A_f = torch.bmm(torch.transpose(fhatx,1,2), dF_df/dF_da)
    #         # A_t = torch.bmm(torch.transpose(fhatx,1,2), 1/dF_da)
    #         A_t = torch.transpose(fhatx,1,2)/dF_da
    #
    #     dF_df = alpha*grad_input - torch.bmm(grad_input, A_f)
    #     dF_dt = torch.bmm(grad_input, A_t)
    #
    #
    #     #we only need to differentiate w.r.t fhatx, target
    #     return None, None, dF_df, dF_dt, None

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.clone()

        fhatx, target, x_root = ctx.saved_tensors

        alpha = ctx.alpha
        V = ctx.V
        F = ctx.F


        with torch.enable_grad():

            Fx = F(fhatx, target, alpha, False)
            A_f = torch.autograd.grad(Fx, fhatx, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            A_t = torch.autograd.grad(Fx, target, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            # b = torch.autograd.grad(Fx, alpha, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(Fx))[0]
            # a = torch.autograd.grad(fhatx, fhatx, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(fhatx))[0]

            # p_alpha*fhatx / p_fhatx

        dF_df = A_f
        dF_dt = A_t

        # loss wrt alpha*fhatx
        grad_rootfind_f = alpha*grad_input + torch.bmm(grad_input, torch.bmm(torch.transpose(fhatx,1,2),dF_df))
        grad_rootfind_t = torch.bmm(grad_input, torch.transpose(fhatx,1,2))*dF_dt

        return None, None, grad_rootfind_f, grad_rootfind_t, None
