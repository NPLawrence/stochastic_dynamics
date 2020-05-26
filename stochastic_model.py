import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Normal, OneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

import convex_model
import rootfind_model

#Snippets taken from: https://github.com/tonyduan/mdn/blob/master/mdn/models.py
class MDN_module(nn.Module):
    #This is where we bring together:
    #   1. Stochastic targets for V
    #   2. The rootfinding-based training/inference method

    def __init__(self, fhat, V, n = None, beta = 0.99, k = 1, is_training = True, show_mu = False):
        super().__init__()

        self.fhat = fhat
        self.V = V
        self.beta = beta
        self.k = k
        self.n = n
        self.is_training = is_training
        self.show_mu = show_mu

        # self.F = rootfind_model.newton_iter(self.fhat, self.V)

    def forward(self, x, y = None):

        fhatx = self.fhat(x)

        mu, var = torch.split(fhatx, fhatx.shape[-1] // 2, dim=-1)
        mu = torch.stack(mu.split(mu.shape[1] // self.k, 1)).view(-1,1,self.n)
        var = torch.exp(torch.stack(var.split(var.shape[1] // self.k, 1))).view(-1,1,self.n)
        mu_stable = mu*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(mu))) / self.V(mu))
        # model_dist = Normal(mu_stable, var)
        # print(var)
        model_dist = MultivariateNormal(mu_stable, torch.diag_embed(var))
        fx = (model_dist.rsample())

        if self.is_training:
            logp_y = -(model_dist.log_prob(y).squeeze()).mean()

            return logp_y
        else:
            # return mu_stable.view(-1,1,2)
            if self.show_mu:
                return mu_stable
            else:
                return fx

class stochastic_module(nn.Module):
    #This is where we bring together:
    #   1. Stochastic targets for V
    #   2. The rootfinding-based training/inference method

    def __init__(self, fhat, V, n = None, beta = 0.99, k = 1, mode = None, is_training = True, show_mu = False):
        super().__init__()

        self.fhat = fhat
        self.V = V
        self.beta = beta
        self.k = k
        self.n = n
        self.is_training = is_training
        self.show_mu = show_mu

        self.pi = pi_Network(self.n, self.k)

        if mode is None:
            self.mu_dynamics = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)
        elif mode == 1:
            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.mu_dynamics = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, fhat = self.mu)
            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)
        else:
            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.mu_dynamics = rootfind_model.rootfind_module(self.mu, self.V, is_training, beta)
            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)

    def forward(self, x, y = None):
        #Perform a noisy forward pass in fhat and V
        #   stochasticity in fhat adds uncertainty to the rootfind initialization
        #   in V we make sure its mean is decreasing

        pi = self.pi(x)

        mu_stable = self.mu_dynamics(x)
        var = self.var_dynamics(x)
        model_dist = MultivariateNormal(mu_stable, covariance_matrix = torch.diag_embed(var)) #Suboptimal a.s. but found to be more convenient/stable implementation
        # print(pi.sample().shape)
        if self.is_training:
            logp_y = (model_dist.log_prob(y).squeeze())
            # print((torch.log(pi.probs).squeeze() + logp_y).shape)
            loss = torch.logsumexp(torch.log(pi.probs).squeeze() + logp_y, dim=-1)

            return -torch.mean(loss)
        else:
            if self.show_mu:
                mean = torch.sum(pi.probs.view(self.k,1)*mu_stable,1)
                return mean
            else:
                return torch.sum(pi.sample().view(self.k,1)*model_dist.sample(),1)

class MDN_dynamics(nn.Module):
    def __init__(self,fhat,n,k, get_mu = True):
        super().__init__()
        self.fhat = fhat
        self.k = k
        self.n = n
        self.get_mu = get_mu

    def forward(self, x):
        fhatx = self.fhat(x)
        mu, var = torch.split(fhatx, fhatx.shape[-1] // 2, dim=-1)
        if self.get_mu:
            output = torch.stack(mu.split(mu.shape[-1] // self.k, 1)).view(-1,self.k,self.n)
        else:
            # output = torch.exp(torch.stack(var.split(var.shape[-1] // self.k, 1))).view(-1,self.k,self.n)
            output = torch.clamp(torch.exp(torch.stack(var.split(var.shape[-1] // self.k, 1))).view(-1,self.k,self.n), min = 1e-10)

        return output
#
# class pi_Network(nn.Module):
#
#     def __init__(self, n, k):
#         super().__init__()
#
#         self.network = nn.Sequential(
#             nn.Linear(n, 50), nn.ReLU(),
#             nn.Linear(50,50), nn.ReLU(),
#             nn.Linear(50, k))
#
#     def forward(self, x):
#
#         params = self.network(x)
#
#         return F.softmax(params, dim = -1)
class pi_Network(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)
