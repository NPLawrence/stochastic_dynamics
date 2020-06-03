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

    def __init__(self, fhat, V=None, n = None, beta = 0.99, k = 1, mode = None, is_training = True, return_mean = True, show_mu = False):
        super().__init__()

        self.fhat = fhat

        self.V = V
        self.beta = beta
        self.k = k
        self.n = n
        self.is_training = is_training
        self.return_mean = return_mean
        self.show_mu = show_mu

        self.pi = pi_Network(self.n, self.k)

        if mode is None:
            self.mu_dynamics = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)
        elif mode == 1:
            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            # self.mu_dynamics = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, f = self.mu)
            self.mean_dynamics = mean_dynamics(self.mu, self.pi, self.k, self.n)
            self.gamma_mu = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, return_gamma = True, f = self.mean_dynamics)
            self.mu_dynamics = stable_dynamics(self.mu, self.gamma_mu)

            # self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)

            self.var = MDN_dynamics(self.fhat, self.n, self.k,False)
            self.var_dynamics = variance_dynamics_2(self.pi, self.mu_dynamics, self.var, self.V, self.k)

            # self.var = MDN_dynamics(self.fhat, self.n, self.k,False)
            # self.variance_dynamics = variance_dynamics(self.mu_dynamics, self.var, self.pi, self.k, self.n)
            # self.gamma_var = convex_model.dynamics_convex(V = self.V, n = n, beta  = beta, is_stochastic_train = is_training, return_gamma = True, f = self.variance_dynamics)
            # self.var_dynamics = stable_dynamics(self.var, self.gamma_var)

        else:
            # self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            # self.mu_dynamics = rootfind_model.rootfind_module(self.mu, self.V, is_training, beta)
            # self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)

            self.mu = MDN_dynamics(self.fhat, self.n, self.k, True)
            self.mean_dynamics = mean_dynamics(self.mu, self.pi, self.k, self.n)
            self.gamma_mu = rootfind_model.rootfind_module_stochastic(self.V, n, beta  = beta, is_training = is_training, f = self.mean_dynamics)
            self.mu_dynamics = stable_dynamics(self.mu, self.gamma_mu)
            # self.mu_dynamics = rootfind_model.rootfind_module(self.mu, self.V, is_training, beta)
            self.var_dynamics = MDN_dynamics(self.fhat, self.n, self.k,False)

    def forward(self, x, y = None):

        pi = self.pi(x)

        # mu = self.mu(x)
        # gamma = self.gamma_mu(x)
        # mu_stable = mu*gamma
        mu_stable = self.mu_dynamics(x)
        # mu_stable = self.mu_dynamics(x)
        var = torch.clamp(self.var_dynamics(x),1e-8)
        model_dist = MultivariateNormal(mu_stable, covariance_matrix = torch.diag_embed(var)) #Suboptimal w.p.1 but found to be more convenient/stable implementation

        if self.is_training:
            logp_y = (model_dist.log_prob(y).squeeze())
            loss = torch.logsumexp(torch.log(pi.probs).squeeze() + logp_y, dim=-1)
            # if np.isnan(loss.detach().numpy()).any():
            #     print(np.argwhere(np.isnan(x)))
            #     print(np.argwhere(np.isnan(var.detach().numpy())), np.argwhere(np.isnan(mu_stable.detach().numpy())))
            if self.return_mean:
                return -torch.mean(loss)
            else:
                return -loss

        else:
            if self.show_mu:
                mean = torch.sum(pi.probs.view(self.k,1)*mu_stable,1)
                # print(mean)
                # print(self.V(mean))
                # print(mean)

                return mean
            else:
                return torch.sum(pi.sample().view(self.k,1)*model_dist.sample(),1)

    def reset(self):
        self.gamma_mu.reset()
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
        # print(var)
        # if np.isnan(var.detach().numpy()).any():
        #     print(np.isnan(x), 'x')
        if self.get_mu:
            output = torch.stack(mu.split(mu.shape[-1] // self.k, 1)).view(-1,self.k,self.n)
            # print(output.shape)
            # print(output)
        else:
            # output = torch.exp(torch.stack(var.split(var.shape[-1] // self.k, 1))).view(-1,self.k,self.n)
            output = torch.clamp(torch.exp(torch.stack(var.split(var.shape[-1] // self.k, 1))).view(-1,self.k,self.n), 1e-8, max = 1000)
            # output = torch.clamp(F.elu(torch.stack(var.split(var.shape[-1] // self.k, 1))).view(-1,self.k,self.n)+1, min = 1e-8, max = 1000)
            # print(output)
        return output

class pi_Network(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        # self.network = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, out_dim)
        self.network = nn.Sequential(nn.Linear(in_dim, 25), nn.ReLU(),
                # nn.Linear(50, 50), nn.ReLU(),
                nn.Linear(25, 25), nn.Tanh(),
                nn.Linear(25, out_dim))

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
        # print(self.pi(x).probs.view(-1,self.k,1))
        # print(self.MDN_means(x).shape)
        mean = torch.sum(self.pi(x).probs.view(-1,self.k,1)*self.MDN_means(x),1).view(-1,1,self.n)
        # print(mean.shape, 'hello')

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
        # print(self.pi(x).probs.view(-1,self.k,1))
        # print(self.MDN_means(x).shape)
        pi = self.pi(x).probs.view(-1,self.k,1)
        sum_var = torch.sum(pi*self.MDN_vars(x), 1)
        sum_squaremean = torch.sum(pi*self.MDN_means(x)**2, 1)
        sum_meansquare = torch.sum(pi*self.MDN_means(x), 1)**2
        # print(sum_var + sum_squaremean - sum_meansquare)
        return sum_var + sum_squaremean - sum_meansquare

class variance_dynamics_2(nn.Module):

    def __init__(self, pi, mu_dynamics, MDN_vars, V, k):
        super().__init__()

        self.pi = pi
        self.mu_dynamics = mu_dynamics
        self.MDN_vars = MDN_vars
        self.V = V
        self.k = k
        self.rehu = ReHU()

    def forward(self, x):
        # print(self.V(self.mu_dynamics(x)))
        # print((torch.clamp(torch.norm(torch.sum(self.pi(x).probs.view(-1,self.k,1)*self.mu_dynamics(x))),1e-8).view(-1,1,1)**2))
        # print(self.V(self.mu_dynamics(x)))
        # print(torch.norm(torch.sum(self.pi(x).probs.view(-1,self.k,1)*self.mu_dynamics(x)).view(-1,1,1)))
        # return (torch.clamp(torch.norm(torch.sum(self.pi(x).probs.view(-1,self.k,1)*self.mu_dynamics(x))),1e-8).view(-1,1,1)**2)*self.MDN_vars(x)
        # return torch.sqrt(torch.clamp(self.V(self.mu_dynamics(x)), 1e-8)).view(-1,self.k,1)*self.MDN_vars(x)
        # print(torch.mean(self.rehu(torch.norm(self.mu_dynamics(x),dim = -1))))
        # print(torch.norm(self.mu_dynamics(x),dim = -1))
        # mean = torch.sum(self.pi(x).probs.view(-1,self.k,1)*self.mu_dynamics(x),1)
        # scale = 10*F.hardtanh(torch.mean(torch.abs(mean)), min_val = 1e-8, max_val = 0.1).view(-1,1,1)
        scale = 10*F.hardtanh(torch.norm(torch.mean(self.mu_dynamics(x),1)), min_val = 1e-8, max_val = 0.1).view(-1,1,1)
        # print(scale)
        # maxval = (100.0*self.V(mean)).detach().numpy()
        # print(maxval)
        # output = (self.V(mean)).view(-1,1,1)*F.hardtanh(self.mu_dynamics(x), min_val=1e-8, max_val=maxval)
        #
        # # return torch.clamp(self.MDN_vars(x), min = 1e-8, max = maxval)
        # return output

        # return torch.clamp(torch.norm(mean), 1e-8).view(-1,1,1)*self.MDN_vars(x)
        # return torch.clamp(self.rehu(torch.mean(torch.norm(self.mu_dynamics(x),-1))),1e-8).view(-1,1,1)*self.MDN_vars(x)
        # return torch.clamp(torch.mean(self.rehu(torch.norm(self.mu_dynamics(x),dim = -1))), 1e-8).view(-1,1,1)*self.MDN_vars(x)
        # return torch.clamp(torch.sqrt(self.V(mean)), 1e-8).view(-1,1,1)*self.MDN_vars(x)
        # return torch.clamp(torch.mean(torch.abs(mean)), 1e-8).view(-1,1,1)*self.MDN_vars(x)
        return scale*self.MDN_vars(x)

class stable_dynamics(nn.Module):

    def __init__(self, MDN_params, gamma_val):
        super().__init__()

        self.MDN_params = MDN_params
        self.gamma_val = gamma_val


    def forward(self, x):
        # print(self.pi(x).probs.view(-1,self.k,1))
        # print(self.MDN_means(x).shape)

        # print(self.MDN_means(x).shape)
        # print(self.gamma_val(x))
        # print(self.gamma_val(x))
        return self.gamma_val(x)*self.MDN_params(x)
