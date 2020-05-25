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

import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

# import convex_model
# import rootfind_model
# loc = torch.zeros(3)
# scale = torch.ones(3)
# mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
# [torch.Size(()), torch.Size((3,))]


# a = torch.randn(5,2)
# b = torch.rand(5,2)
#
# print(b)
# print(torch.diag_embed(b))
# mvn = MultivariateNormal(a, torch.diag_embed(b))
# print(mvn.variance)
#
# comp = Normal(a, b)
# print(comp.variance)


# import torch
n = 2
d = 5
diagonal = torch.rand(d) + 1.
mu = torch.rand(n, d)
p1 = D.Independent(torch.distributions.Normal(mu, diagonal.reshape(1,1, d)),1)
p2 = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(diagonal).reshape(1, d, d))
x = torch.rand((n,d))
print(p1.mean)
print(p2.mean)
print(p1.log_prob(x).sum() - p2.log_prob(x).sum())
