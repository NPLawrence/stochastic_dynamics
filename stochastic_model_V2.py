

import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

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

    def __init__(self, fhat, V, n = None, beta = 0.99, k = 1, is_training = True, show_mu = False):
        super().__init__()

        self.fhat = fhat
        self.V = V
        self.beta = beta
        self.k = k
        self.n = n
        self.is_training = is_training
        self.show_mu = show_mu

        self.mu_dynamics = MDN_dynamics(self.fhat, self.n, self.k, True)
        self.mu_rootfind = rootfind_model.rootfind_module(self.mu_dynamics, self.V, is_training, beta)
        self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)

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
            output = torch.stack(mu.split(mu.shape[1] // self.k, 1)).view(-1,1,self.n)
        else:
            output = torch.exp(torch.stack(var.split(var.shape[1] // self.k, 1))).view(-1,1,self.n)

        return output

#see the following for original source:
#   https://github.com/tonyduan/mdn/blob/master/mdn/models.py
#   (modified for own use)
class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    mode: int; None for standard MDN, 1 for convex-based stability, 2 for root-finding
    """

    def __init__(self, dim_in, dim_out, n_components, V = None, mode = None, is_training = True, show_mu = False):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components, V = V, mode = mode)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        loss = loss.mean()
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None, V = None, mode = None):
        super().__init__()
        self.out_dim = out_dim
        self.n_components = n_components
        self.V = V
        self.mode = mode
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

        if mode == 1:
            self.mu = MDN_dynamics(self.network, self.in_dim, self.n_components, True)
            self.mu_dynamics = convex_model.dynamics_convex(self.mu, self.V)
            self.var_dynamics = MDN_dynamics(self.network, self.in_dim, self.n_components, False)

        elif mode == 2:
            self.mu = MDN_dynamics(self.network, self.in_dim, self.n_components, True)
            self.mu_dynamics = rootfind_model.rootfind_module(self.mu, self.V, is_training, beta)
            self.var_dynamics = MDN_dynamics(self.network, self.in_dim, self.n_components, False)


    def forward(self, x):

        if self.mode is None:
            params = self.network(x)
            mean, sd = torch.split(params, params.shape[-1] // 2, dim=-1)
            mean = torch.stack(mean.split(mean.shape[-1] // self.n_components, 1)).view(-1,self.n_components,self.out_dim)
            sd = torch.stack(sd.split(sd.shape[-1] // self.n_components, 1)).view(-1,self.n_components,self.out_dim)
            return Normal(mean, torch.exp(sd))
        else:
            mean = self.mu_dynamics(x)
            var = self.var_dynamics(x)
            return Normal(mean.transpose(0, 1), var.transpose(0, 1))


class CategoricalNetwork(nn.Module):

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
        return OneHotCategorical(logits=params.squeeze())
