import torch
import torch.nn as nn
import torch.nn.functional as F

# Stabilizing strategy for convex Lyapunov functions

class fhat(nn.Module):
    """
    This is our 'nominal' model before modifying the dynamics

    layer_sizes : 1D array of shape (m,) where m-2 is the number of layers. First and last
        entires are input and output dimensions, respectively.
    add_state : binary variable. Can be ignored; only used in an example in the appendix.
    """
    def __init__(self, layer_sizes, add_state = False):
        super().__init__()

        self.add_state = add_state
        layers = []
        for i in range(len(layer_sizes)-2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.fhat = nn.Sequential(*layers)

    def forward(self, x):

        if self.add_state:
            z = x + self.fhat(x)
        else:
            z = self.fhat(x)
        return z

class dynamics_convex(nn.Module):
    """
    Stable dynamics model based on convex Lyapunov function

    V : Lyapunov neural network
    n : state dimension
    beta : number in (0,1] in the stability criterion V(x') <= beta V(x)
    is_stochastic_train : binary variable indicating if a stochastic model is being trained.
        For training the stochastic model we may want to return just the gamma term or
        keep track of the previous 'state' i.e. means.
    return_gamma : binary variable. Indicates whether to return gamma(x)*fhat(x) or gamma(x)
    f : optional user-defined nominal model.
    """
    def __init__(self, V, n, beta = 0.99, is_stochastic_train = False, return_gamma = False, f = None):
        super().__init__()

        if f is None:
            self.fhat = fhat(np.array([n, 25, 25, 25, n]), False)
        else:
            self.fhat = f
        self.V = V
        self.beta = beta
        self.is_stochastic_train = is_stochastic_train
        self.is_init = True
        self.return_gamma = return_gamma
        self.n = n

    def forward(self, x):

        if self.is_stochastic_train:
        # This is for training
            target = self.beta*self.V(x)
            current = self.V(self.fhat(x))

            fx = self.fhat(x)*((target - F.relu(target - current)) / current)
            if self.return_gamma:
                return ((target - F.relu(target - current)) / current)
            else:
                return fx
        else:
        # This is for testing -- particularly stochastic models in order to track the mean/var
            if self.is_init:
                target = self.beta*self.V(x)
                self.is_init = False

            else:
                target = self.beta*self.V(self.fx)

            current = self.V(self.fhat(x))

            fx = self.fhat(x)*((target - F.relu(target - current)) / current)
            self.fx = fx
            if self.return_gamma:
                return ((target - F.relu(target - current)) / current)
            else:
                return self.fx

    def reset(self):
        self.is_init = True



class dynamics_nonincrease(nn.Module):
    """
    Modifies fhat by ensuring 'non-expansiveness'. See the appendix for explanation/example.
        i.e. it never moves in a direction that is acute with the gradient of V at x_t

    V : Lyapunov neural network
    n : state dimension
    f : optional user-defined nominal model.
    """

    def __init__(self, V, n, f = None):
        super().__init__()

        if f is None:
            self.fhat = fhat(np.array([n, 25, 25, 25, n]), False)
        else:
            self.fhat = f

        self.V = V

    def forward(self, x):

        x = x.requires_grad_(True)
        fhatx = self.fhat(x)
        Vx = self.V(x)
        with torch.enable_grad():
            gV = torch.autograd.grad(Vx, x, retain_graph = True, only_inputs=True, grad_outputs=torch.ones_like(Vx))[0]

        fx = x + self.fhat(x) - F.relu((gV*(self.fhat(x) - x)).sum(dim = -1, keepdim = True))*gV/(torch.norm(gV, dim = -1, keepdim = True)**2)

        return fx
