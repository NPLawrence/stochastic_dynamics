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

from modules.lyapunov_NN import ReHU

import modules.convex_model
import modules.rootfind_model

#Stabilizing strategy for stochastic systems (convex or nonconvex Lyapunov functions)

#Snippets taken from: https://github.com/tonyduan/mdn/blob/master/mdn/models.py

# class MDN_module(nn.Module):
#     #This is where we bring together:
#     #   1. Stochastic targets for V
#     #   2. The rootfinding-based training/inference method
#
#     def __init__(self, fhat, V, n = None, beta = 0.99, k = 1, is_training = True, show_mu = False):
#         super().__init__()
#
#         self.fhat = fhat
#         self.V = V
#         self.beta = beta
#         self.k = k
#         self.n = n
#         self.is_training = is_training
#         self.show_mu = show_mu
#
#     def forward(self, x, y = None):
#
#         fhatx = self.fhat(x)
#
#         mu, var = torch.split(fhatx, fhatx.shape[-1] // 2, dim=-1)
#         mu = torch.stack(mu.split(mu.shape[1] // self.k, 1)).view(-1,1,self.n)
#         var = torch.exp(torch.stack(var.split(var.shape[1] // self.k, 1))).view(-1,1,self.n)
#         mu_stable = mu*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(mu))) / self.V(mu))
#
#         model_dist = MultivariateNormal(mu_stable, torch.diag_embed(var))
#         fx = (model_dist.rsample())
#
#         if self.is_training:
#             logp_y = -(model_dist.log_prob(y).squeeze()).mean()
#
#             return logp_y
#         else:
#             if self.show_mu:
#                 return mu_stable
#             else:
#                 return fx

class stochastic_module(nn.Module):
    """
    This is where we bring together:
        1. Stochastic model (MDN)
        2. Stable models (convex_model or rootfind_model)

    V : Lyapunov neural network
    n : state dimension
    f : nominal model
    beta : number in (0,1] in the stability criterion V(x') <= beta V(x)
    k : number of mixtures in the MDN
    mode : None, 1 or some other integer
        None is a vanilla MDN
        1 is convexity-based MDN
        otherwise, use implicit method
    is_training : binary variable. Gets passed to other modules; also used in forward pass
        to indicate if we need negative log-likelihood or a sample from MDN.
    show_mu : binary variable. If is_training == False then this will either output the
        conditional mean or a sample from the MDN.
    """
    def __init__(self, V, n, f = None, beta = 0.99, k = 1, mode = None, is_training = False, show_mu = False):
        super().__init__()

        if f is None:
            self.fhat = nn.Sequential(nn.Linear(n, 25), nn.Softplus(),
                                nn.Linear(25, 25), nn.Softplus(),
                                nn.Linear(25, 25), nn.Softplus(),
                                nn.Linear(25, 2*n*k))
        else:
            self.fhat = f

        self.V = V
        self.beta = beta
        self.k = k
        self.n = n
        self.is_training = is_training
        self.show_mu = show_mu

        self.pi = pi_Network(self.n, self.k)

        if mode is None:
            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)
            self.gamma = lambda x: 1.0
            self.mean_dynamics = lambda x: 1.0
        elif mode == 1:
            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.mean_dynamics = mean_dynamics(self.mu, self.pi, self.k, self.n)
            self.gamma = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, f = self.mean_dynamics)

            self.var = MDN_dynamics(self.fhat, self.n, self.k, False)
            self.var_dynamics = variance_dynamics(self.pi, self.gamma, self.var, self.V, self.k)

        else:

            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.mean_dynamics = mean_dynamics(self.mu, self.pi, self.k, self.n)
            self.gamma = rootfind_model.rootfind_module(self.V, self.n, is_training=is_training, beta=beta, f = self.mean_dynamics)

            self.var = MDN_dynamics(self.fhat, self.n, self.k,False)
            self.var_dynamics = variance_dynamics(self.pi, self.gamma, self.var, self.V, self.k)

    def forward(self, x, y = None):

        pi = self.pi(x)

        mu_stable = self.gamma(x)*(self.mu(x)/self.mean_dynamics(x))
        var = torch.clamp(self.var_dynamics(x),min = 1e-8)
        model_dist = MultivariateNormal(mu_stable, covariance_matrix = torch.diag_embed(var))

        if self.is_training:
            logp_y = (model_dist.log_prob(y).squeeze())
            loss = torch.logsumexp(torch.log(pi.probs).squeeze() + logp_y, dim=-1)
            return -torch.mean(loss)

        else:
            if self.show_mu:
                mean = torch.sum(pi.probs.view(-1,self.k,1)*mu_stable,1)
                return mean
            else:
                return torch.sum(pi.sample().view(-1,self.k,1)*model_dist.sample(),1)

    def reset(self):
        self.gamma.reset()

class MDN_dynamics(nn.Module):
    """
    This module takes a MDN and outputs the mean and variance parameters

    fhat : The user-provided MDN model
    n : state dimension
    k : number of mixtures
    get_mu : binary variable indicating whether to output the mean parameters
    """
    def __init__(self, fhat, n, k, get_mu = True):
        super().__init__()
        self.fhat = fhat
        self.n = n
        self.k = k
        self.get_mu = get_mu

    def forward(self, x):
        fhatx = self.fhat(x)
        mu, var = torch.split(fhatx, fhatx.shape[-1] // 2, dim=-1)
        if self.get_mu:
            output = torch.stack(mu.split(mu.shape[-1] // self.k, 1)).view(-1,self.k,self.n)
        else:
            #   torch.exp is also possible here, but using ELU + 1 tends to be more `stable'
            output = torch.clamp(F.elu(torch.stack(var.split(var.shape[-1] // self.k, 1))).view(-1,self.k,self.n) + 1, 1e-8, max = 100)
        return output

class pi_Network(nn.Module):
    """
    This is where we define the mixture coefficients network

    in_dim : input dimension (e.g. n)
    out_dim : output dimension (e.g. k)
    hidden_dim : hidden dimension
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 25
        self.network = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)

class mean_dynamics(nn.Module):
    """
    Conditional mean dynamics derived from the means parameters and mixture coefficients
        of a MDN

    MDN_means : network for the mean parameters (e.g. MDN_dynamics)
    pi_Network : network for the mixture coefficients (e.g. pi_Network)
    k : number of mixtures
    n : state dimension
    """
    def __init__(self, MDN_means, pi_Network, k, n):
        super().__init__()

        self.MDN_means = MDN_means
        self.pi = pi_Network
        self.k = k
        self.n = n

    def forward(self, x):

        mu = self.MDN_means(x)
        mean = torch.sum(self.pi(x).probs.view(-1,self.k,1)*mu,1).view(-1,1,self.n)
        return mean

class variance_dynamics(nn.Module):
    """
    Conditional covariance dynamics derived from the variance parameters of a MDN.
        Here the user can define the way in which the covariance goes to zero.

    pi_Network : network for the mixture coefficients (e.g. pi_Network)
    mean_dynamics : conditional mean (e.g. mean_dynamics module)
    MDN_vars : network for the variance parameters (e.g. MDN_dynamics)
    V : Lyapunov neural network
    k : number of mixtures
    """
    def __init__(self, pi_Network, mean_dynamics, MDN_vars, V, k):
        super().__init__()

        self.pi = pi_Network
        self.mean_dynamics = mean_dynamics
        self.MDN_vars = MDN_vars
        self.V = V
        self.k = k

    def forward(self, x):

        # Define a function of mean_dynamics that goes to zero as mean_dynamics --> 0
        #   e.g. norm, V
        scaleVar = torch.norm(self.mean_dynamics(x),dim = -1)

        output = scaleVar.unsqueeze(-1)*torch.clamp(self.MDN_vars(x), min = 1e-8, max = 100)

        return output
