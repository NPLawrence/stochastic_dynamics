import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class rootfind_module(nn.Module):

    def __init__(self,V,fhat,target,x):
        super().__init__()

        self.V = V
        self.f = f

    def forward(self,target,x):

        rootfind = rootfind_train.apply
        output = rootfind(target,x)
        # raise NotImplemented
        return output


    class rootfind_train(torch.autograd.Function):

        @staticmethod
        def forward(ctx,V,fhat,target,x):
            #just return the input -- the root is computed elsewhere

            ctx.V = V
            ctx.fhat = fhat
            ctx.target = target

            fhatx = fhat(x)
            alpha = torch.tensor([1], dtype = torch.float, requires_grad = True)

            while (V(fhatx*alpha) - target) > 0.001:

                y = fhatx*alpha
                gV = torch.autograd.grad([a for a in V(y)], [alpha], only_inputs=True)[0]
                # gV = torch.autograd.grad([a for a in V(y)], [y], create_graph=True, only_inputs=True)[0]
                alpha = (alpha - (V(fhatx*alpha) - target)/(gV*fhatx).sum(dim = 1))

            x_root = alpha

            ctx.save_for_backward(x_root, x)

            return x_root


        @staticmethod
        def backward(ctx, grad_output):

            x_root, x = ctx.saved_tensors
            V = ctx.V
            fhat = ctx.fhat
            target = ctx.target
            grad_input = grad_output.clone()


            # with torch.enable_grad():
            #     newton_func = rootfind_alg.g(V,target,x_root)

            # print(V(x_root*fhat(x)), x_root)
            with torch.enable_grad():
              A = torch.autograd.grad(V(x_root*fhat(x)) - target, x_root, create_graph=True, only_inputs=True)[0]
              # print((V(x_root*fhat(x)) - target).requires_grad)
              # print(V)
              b = torch.autograd.grad(V(x_root*fhat(x)) - target, x, create_graph=True, only_inputs=True, grad_outputs=torch.ones_like(V(x_root*fhat(x))))[0]
              # b = V(x).backward(torch.ones_like(V(x)))
              # b = torch.autograd.grad((V(x_root*x) - target), x, create_graph=True, only_inputs=True)[0]
              # print(b)
            # for name, weight in V.named_parameters():
            #
            #     weight.grad = weight.grad/A
                # print(name, weight.grad)
            dx_dw = b/A

            # grad_rootfind = [weight.grad/A for weight in V.named_parameters()]
            grad_rootfind = grad_input*dx_dw #apply chain rule
            

            return grad_rootfind

    class rootfind_alg(torch.autograd.Function):
    #A basic implementation of Newton's method
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
