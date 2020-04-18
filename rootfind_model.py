import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class rootfind_module(nn.Module):

    def __init__(self,fhat, V):
        super().__init__()

        self.V = V
        self.fhat = fhat

    def forward(self,x):


        target = .99*self.V(x)
        alpha = torch.tensor([1], dtype = torch.float, requires_grad = True)
        fhatx = self.fhat(x)
        # print(fhatx.requires_grad)
        rootfind = rootfind_train.apply
        x_root = rootfind(self.V, fhatx, target, x)
        # output_f = alpha_root*fhatx
        # raise NotImplemented
        return x_root


class rootfind_train(torch.autograd.Function):

    @staticmethod
    def forward(ctx,V,fhatx,target,x):
        #just return the input -- the root is computed elsewhere
        # ctx.save_for_backward(x)
        ctx.V = V
        ctx.fhatx = fhatx
        # ctx.fhatx = (fhat(x)).clone().detach()
        ctx.target = target
        # ctx.alpha = alpha
        # ctx.fhatx = fhatx


        # print(alpha.requires_grad)
        # fhatx.requires_grad_(True)

        alpha = torch.tensor([1], dtype = torch.float, requires_grad = True)
        # fhatx = fhat(x)
        # x_root = (fhatx*alpha).clone().detach()
        # print(alpha)
        # print(fhatx, x)


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

        # print(alpha)
        alpha_root = alpha.clone().detach().requires_grad_(True)
        # print(alpha_root)

        # print(alpha_root)
        # print(alpha_root)

        # x_root = alpha_root*fhat(x)
        x_root = (alpha_root*fhatx).clone().detach().requires_grad_(True)

        ctx.alpha_root = alpha_root
        # print(alpha_root.requires_grad)

        ctx.save_for_backward(x, x_root)


        return x_root


    @staticmethod
    def backward(ctx, grad_output):
        # print('hello')
        # print(ctx.saved_tensors)
        x, x_root= ctx.saved_tensors
        x = x.clone().detach().requires_grad_()
        x_root = x_root.clone().detach().requires_grad_()
        grad_input = grad_output.clone()



        alpha_root = ctx.alpha_root.clone()

        # alpha, = ctx.saved_tensors

        V = ctx.V
        fhatx = ctx.fhatx
        # fhatx = fhat(x)
        target = ctx.target


        # w = V(x_root)
        # print(grad_input)

        # with torch.enable_grad():
        #     newton_func = rootfind_alg.g(V,target,x_root)

        # print(V(x_root*fhat(x)), x_root)
        with torch.enable_grad():

            # fhatx.requires_grad_(True)
            # x_root.requires_grad_(True)
            # target.requires_grad_(True)
            # gV = torch.autograd.grad(V(x_root), x_root)[0]
            # gV.backward(torch.ones_like(gV))
            # dummy = torch.autograd.grad(gV, x_root, grad_outputs = torch.ones_like(gV))
            # A = torch.autograd.grad((fhatx*(V(x_root) - target)/((gV*fhatx).sum(dim = -1))), x_root,create_graph = False, retain_graph = True, allow_unused = True, only_inputs=True, grad_outputs = torch.ones_like(x_root))[0]
            # T = (x_root - fhatx*(V(x_root) - target)/((gV*fhatx).sum(dim = -1))).requires_grad_(True)

            # A = torch.autograd.grad((x_root - fhatx*(V(x_root) - target)/((torch.autograd.grad(V(x_root), x_root, create_graph = False, allow_unused = True, only_inputs=True)[0]*fhatx).sum(dim = -1))).requires_grad_(True), x_root, grad_outputs=torch.ones_like((x_root - fhatx*(V(x_root) - target)/(((torch.autograd.grad(V(x_root), x_root, create_graph = False, allow_unused = True, only_inputs=True)[0]).requires_grade_(True)*fhatx).sum(dim = -1))).requires_grad_(True)), create_graph=True, retain_graph = True, only_inputs=True)[0]
            A = torch.autograd.grad(V(x_root) - target, x_root, create_graph=True, retain_graph = True, only_inputs=False, grad_outputs=torch.ones_like(V(x_root)))[0]
          # print((V(x_root*fhat(x)) - target).requires_grad)
          # print(V)
          # fhatx = fhat(x)
          # b = V(alpha_root*fhatx) - target
          # b.backward()
            c = torch.autograd.grad(V(alpha_root*fhatx) - target, fhatx, create_graph=True, retain_graph = True, only_inputs=False, grad_outputs=torch.ones_like(V(x_root)))[0]
            b = torch.autograd.grad(V(alpha_root*fhatx) - target, target, create_graph=True, retain_graph = True, only_inputs=False, grad_outputs=torch.ones_like(V(x_root)))[0]
          # b = V(x).backward(torch.ones_like(V(x)))
          # b = torch.autograd.grad((V(x_root*x) - target), x, create_graph=True, only_inputs=True)[0]
          # print(b)
        # for name, weight in V.named_parameters():
        #
        #     weight.grad = weight.grad/A
            # print(name, weight.grad)
        # print(A)
        # print(b)

        dx_dw = b.clone().detach()/A.clone().detach()
        dx_dv = c.clone().detach()/A.clone().detach()

        # grad_rootfind = [weight.grad/A for weight in V.named_parameters()]
        grad_rootfind = grad_input*dx_dw #apply chain rule
        # grad_root = grad_rootfind.clone()

        # print(grad_rootfind)
        return None, grad_input*dx_dw, grad_input*dx_dw, None

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

    # @staticmethod
    # def g(V, target, x):
    #     return V(x) - target
