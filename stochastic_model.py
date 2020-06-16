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

from lyapunov_NN import ReHU

import convex_model
import rootfind_model

#Stabilizing strategy for stochastic systems (convex or nonconvex Lyapunov functions)


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

    def forward(self, x, y = None):

        fhatx = self.fhat(x)

        mu, var = torch.split(fhatx, fhatx.shape[-1] // 2, dim=-1)
        mu = torch.stack(mu.split(mu.shape[1] // self.k, 1)).view(-1,1,self.n)
        var = torch.exp(torch.stack(var.split(var.shape[1] // self.k, 1))).view(-1,1,self.n)
        mu_stable = mu*((self.beta*self.V(x) - F.relu(self.beta*self.V(x) - self.V(mu))) / self.V(mu))

        model_dist = MultivariateNormal(mu_stable, torch.diag_embed(var))
        fx = (model_dist.rsample())

        if self.is_training:
            logp_y = -(model_dist.log_prob(y).squeeze()).mean()

            return logp_y
        else:
            if self.show_mu:
                return mu_stable
            else:
                return fx

class stochastic_module(nn.Module):
    #This is where we bring together:
    #   1. Stochastic model (MDN)
    #   2. The rootfinding-based or convex-scaling methods

    def __init__(self, fhat, V=None, n = None, beta = 0.99, k = 1, mode = None, is_training = True, return_meanNLL = True, show_mu = False):
        super().__init__()

        self.fhat = fhat

        self.V = V
        self.beta = beta
        self.k = k
        self.n = n
        self.is_training = is_training
        self.return_meanNLL = return_meanNLL
        self.show_mu = show_mu

        self.pi = pi_Network(self.n, self.k)

        if mode is None:
            self.mu_dynamics = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)
        elif mode == 1:
            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            # self.mu_dynamics = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, f = self.mu)
            self.mean_dynamics = mean_dynamics(self.mu, self.pi, self.k, self.n)
            self.gamma = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, f = self.mean_dynamics)
            # self.gamma_mu = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, return_gamma = True, f = self.mean_dynamics)
            # self.mu_dynamics = stable_dynamics(self.mu, self.gamma_mu)

            self.var = MDN_dynamics(self.fhat, self.n, self.k, False)
            # self.var_dynamics = variance_dynamics_2(self.pi, self.gamma, self.var, self.V, self.k)

            # self.var = MDN_dynamics(self.fhat, self.n, self.k,False)
            # self.variance_dynamics = variance_dynamics(self.mu_dynamics, self.var, self.pi, self.k, self.n)
            # self.gamma_var = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, return_gamma = True, f = self.variance_dynamics)
            # self.var_dynamics = stable_dynamics(self.var, self.gamma_var)

        else:
            # self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            # self.mu_dynamics = rootfind_model.rootfind_module(self.mu, self.V, is_training, beta)
            # self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)

            # self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            # self.mean_dynamics = mean_dynamics(self.mu, self.pi, self.k, self.n)
            # self.gamma_mu = rootfind_model.rootfind_module_stochastic(self.V, n, beta  = beta, is_training = is_training, f = self.mean_dynamics)
            # self.mu_dynamics = stable_dynamics(self.mu, self.gamma_mu)
            # self.mu_dynamics = rootfind_model.rootfind_module(self.mu, self.V, is_training, beta)

            # self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)

            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.mean_dynamics = mean_dynamics(self.mu, self.pi, self.k, self.n)
            self.gamma = rootfind_model.rootfind_module(self.V, self.n, is_training=is_training, beta=beta, f = self.mean_dynamics)

            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)


    def forward(self, x, y = None):

        pi = self.pi(x)

        mu_stable = self.gamma(x)*(self.mu(x)/self.mean_dynamics(x))
        var = torch.clamp(self.var_dynamics(x),min = 1e-8)
        model_dist = MultivariateNormal(mu_stable, covariance_matrix = torch.diag_embed(var))

        if self.is_training:
            logp_y = (model_dist.log_prob(y).squeeze())
            loss = torch.logsumexp(torch.log(pi.probs).squeeze() + logp_y, dim=-1)
            if self.return_meanNLL:
                return -torch.mean(loss)
            else:
                return -loss

        else:
            if self.show_mu:
                mean = torch.sum(pi.probs.view(-1,self.k,1)*mu_stable,1)
                return mean
            else:
                return torch.sum(pi.sample().view(-1,self.k,1)*model_dist.sample(),1)

    def reset(self):
        self.gamma.reset()
        # self.gamma_var.reset()

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
            output = torch.clamp(torch.exp(torch.stack(var.split(var.shape[-1] // self.k, 1))).view(-1,self.k,self.n), 1e-8, max = 100)
        return output

class pi_Network(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(nn.Linear(in_dim, 10), nn.Softplus(),
                nn.Linear(10, 10), nn.Tanh(),
                nn.Linear(10, out_dim))

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)

class mean_dynamics(nn.Module):

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

    def __init__(self, MDN_means, MDN_vars, pi_Network, k, n):
        super().__init__()

        self.MDN_means = MDN_means
        self.MDN_vars = MDN_vars
        self.pi = pi_Network
        self.k = k
        self.n = n

    def forward(self, x):

        pi = self.pi(x).probs.view(-1,self.k,1)
        sum_var = torch.sum(pi*self.MDN_vars(x), 1)
        sum_squaremean = torch.sum(pi*self.MDN_means(x)**2, 1)
        sum_meansquare = torch.sum(pi*self.MDN_means(x), 1)**2

        return sum_var + sum_squaremean - sum_meansquare

class variance_dynamics_2(nn.Module):

    def __init__(self, pi, mean_dynamics, MDN_vars, V, k):
        super().__init__()

        self.pi = pi
        self.mean_dynamics = mean_dynamics
        self.MDN_vars = MDN_vars
        self.V = V
        self.k = k

    def forward(self, x):

        max = torch.norm(self.mean_dynamics(x),dim = -1)**2

        output = max.unsqueeze(-1)*torch.clamp(self.MDN_vars(x), min = 1e-8, max = 100)

        return output

class stable_dynamics(nn.Module):

    def __init__(self, MDN_params, gamma_val):
        super().__init__()

        self.MDN_params = MDN_params
        self.gamma_val = gamma_val


    def forward(self, x):

        return self.gamma_val(x)*self.MDN_params(x)
