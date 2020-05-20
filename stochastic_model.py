import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

import convex_model
import rootfind_model

#Snippets taken from: https://github.com/tonyduan/mdn/blob/master/mdn/models.py
class MDN_module(nn.Module):
    #This is where we bring together:
    #   1. Stochastic targets for V
    #   2. The rootfinding-based training/inference method

    def __init__(self, fhat, V, beta = 0.99, k = 1, is_training = True, show_mu = False):
        super().__init__()

        self.fhat = fhat
        self.V = V
        self.beta = beta
        self.k = k
        self.is_training = is_training
        self.show_mu = show_mu

        # self.F = rootfind_model.newton_iter(self.fhat, self.V)

    def forward(self, x, y = None):

        fhatx = self.fhat(x)

        mu, var = torch.split(fhatx, fhatx.shape[-1] // 2, dim=-1)
        mu = torch.stack(mu.split(mu.shape[1] // self.k, 1)).view(-1,1,2)
        var = torch.exp(torch.stack(var.split(var.shape[1] // self.k, 1))).view(-1,1,2)
        mu_stable = mu*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(mu))) / self.V(mu))
        # model_dist = Normal(mu_stable, var)
        # print(var)
        model_dist = MultivariateNormal(mu_stable, torch.diag_embed(var))
        fx = (model_dist.rsample())

        if self.is_training:
            logp_y = -(model_dist.log_prob(y).squeeze()).sum()

            return fx, logp_y
        else:
            # return mu_stable.view(-1,1,2)
            if self.show_mu:
                return torch.stack([mu_stable, var]).squeeze()
            else:
                return fx

class stochastic_module(nn.Module):
    #This is where we bring together:
    #   1. Stochastic targets for V
    #   2. The rootfinding-based training/inference method

    def __init__(self, fhat, V, beta = 0.99, k = 1, is_training = True, show_mu = False):
        super().__init__()

        self.fhat = fhat
        self.V = V
        self.beta = beta
        self.k = k
        self.is_training = is_training
        self.show_mu = show_mu

        self.mu_dynamics = MDN_dynamics(self.fhat, self.k,True)
        self.mu_rootfind = rootfind_model.rootfind_module(self.mu_dynamics, self.V, is_training, beta)

        self.var_dynamics = MDN_dynamics(self.fhat, self.k,False)

    def forward(self, x, y = None):
        #Perform a noisy forward pass in fhat and V
        #   stochasticity in fhat adds uncertainty to the rootfind initialization
        #   in V we make sure its mean is decreasing

        mu_stable = self.mu_rootfind(x)

        var = self.var_dynamics(x)
        # var = self.var_dynamics(x)


        model_dist = MultivariateNormal(mu_stable, torch.diag_embed(var))
        fx = (model_dist.rsample())

        if self.is_training:
            logp_y = -(model_dist.log_prob(y).squeeze()).sum()
            loss = logp_y
            return fx, loss
        else:
            # return mu_stable.view(-1,1,2)
            if self.show_mu:
                return torch.stack([mu_stable, var]).squeeze()
            else:
                return fx

class MDN_dynamics(nn.Module):
    def __init__(self,fhat,k, get_mu = True):
        super().__init__()
        self.fhat = fhat
        self.k = k
        self.get_mu = get_mu

    def forward(self, x):
        fhatx = self.fhat(x)
        mu, var = torch.split(fhatx, fhatx.shape[-1] // 2, dim=-1)
        if self.get_mu:
            output = torch.stack(mu.split(mu.shape[1] // self.k, 1)).view(-1,1,2)
        else:
            output = torch.exp(torch.stack(var.split(var.shape[1] // self.k, 1))).view(-1,1,2)

        return output
