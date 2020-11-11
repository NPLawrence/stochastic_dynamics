#This is where we generate data for experiments

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.distributions import Beta

mainPath = Path('./datasets')
mainPath.mkdir(exist_ok=True)

class data_linear():
    def __init__(self, two_step = False, add_noise = False):

        A = np.array([[0.90, 1],[0, 0.90]])
        A = A.transpose()
        data = []

        X = np.linspace(-5,5,num=14)

        for x1 in X:
            for x2 in X:

                x = np.array([[x1,x2]])
                if two_step:
                    if add_noise:
                        x_step = np.dot(x,A) + 0.05*x*np.random.normal(0,1)
                    else:
                        x_step = np.dot(x,A)

                for i in range(30):

                    if two_step:
                        if add_noise:
                            x_new = np.dot(x_step,A) + 0.05*x*np.random.normal(0,1)
                        else:
                            x_new = np.dot(x_step,A)

                        data.append(np.array((x,x_step,x_new)).reshape((1,6)).squeeze())
                        x = x_step
                        x_step = x_new
                    else:
                        if add_noise:
                            x_new = np.dot(x,A) + 0.05*x*np.random.normal(0,1)
                        else:
                            x_new = np.dot(x,A)
                        data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
                        x = x_new

        if two_step:
            if add_noise:
                np.savetxt(mainPath/'data_linear_twostep_noise.csv', data, delimiter=",")
            else:
                np.savetxt(mainPath/'data_linear_twostep.csv', data, delimiter=",")
        else:
            if add_noise:
                np.savetxt(mainPath/'data_linear_noise.csv', data, delimiter=",")
            else:
                np.savetxt(mainPath/'data_linear.csv', data, delimiter=",")

class data_nonConvex():
    def __init__(self):

        self.h = 0.1

    def f(self, state):
        x, y = np.squeeze(state)
        return np.array([[y, -y - np.sin(x) - 2*np.clip(x+y,a_min = -1, a_max = 1)]])

    def gen_data(self, x0 = None, steps = None):
        data = []
        if x0 is None:
            # X = np.linspace(-5,5,num=15)
            X = np.linspace(-6,6,num=15)
            for x1 in X:
                for x2 in X:
                    x = np.array([[x1,x2]])
                    for i in range(40):

                        k1 = self.f(x)
                        k2 = self.f(x + self.h*(k1/2))
                        k3 = self.f(x + self.h*(k2/2))
                        k4 = self.f(x + self.h*k3)
                        x_new = x + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)
                        data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
                        x = x_new

            np.savetxt(mainPath/'data_nonConvex.csv', data, delimiter=",")

        else:
            if steps is None:
                steps = 50
            else:
                steps = steps

            x = np.array(x0.view(1,-1).numpy())
            data.append(np.array((x)).reshape((1,2)).squeeze())
            for i in range(steps):
                k1 = self.f(x)
                k2 = self.f(x + self.h*(k1/2))
                k3 = self.f(x + self.h*(k2/2))
                k4 = self.f(x + self.h*k3)
                x_new = x + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)
                data.append(np.array((x_new)).reshape((1,2)).squeeze())
                x = x_new
            return np.array([data]).squeeze()


class data_stochasticNonlinear():
    def __init__(self):

        self.h = 0.05

    def f(self, state):
        x, y = np.squeeze(state)
        f1 = -x*(1/(np.sqrt(np.linalg.norm(state,2)))) - x + y
        g1 = np.sin(x)
        f2 = -y*(1/(np.sqrt(np.linalg.norm(state,2)))) - (10/3)*y + x
        g2 = y
        a = np.array([[f1, f2]])
        b = np.array([[g1, g2]])
        return a, b

    def gen_data(self, x0=None, steps = None):

        h = self.h
        data = []

        if x0 is None:

            X = np.linspace(-5,5,num=18)

            for x1 in X:
                for x2 in X:
                    x = np.array([[x1,x2]])
                    for i in range(5):

                        Z_t, S_t = np.random.normal(0,1), np.random.choice([-1,1])
                        W_t = np.sqrt(h)*Z_t

                        a1, b1 = self.f(x)
                        k1 = h*a1 + (W_t - np.sqrt(h)*S_t)*b1

                        a2, b2 = self.f(x + k1)
                        k2 = h*a2 + (W_t + np.sqrt(h)*S_t)*b2

                        x_new = x + (1/2)*(k1 + k2)
                        if np.isnan(np.array((x,x_new))).any():
                            print(x)
                            break
                        else:
                            data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
                            x = x_new

            np.savetxt(mainPath/'data_stochasticNonlinear.csv', data, delimiter=",")

        else:
            if steps is None:
                steps = 100
            else:
                steps = steps
            x = np.array(x0.view(1,-1).numpy())
            data.append(np.array((x)).reshape((1,2)).squeeze())
            for i in range(steps):

                Z_t, S_t = np.random.normal(0,1), np.random.choice([-1,1])
                W_t = np.sqrt(h)*Z_t

                a1, b1 = self.f(x)
                k1 = h*a1 + (W_t - np.sqrt(h)*S_t)*b1

                a2, b2 = self.f(x + k1)
                k2 = h*a2 + (W_t + np.sqrt(h)*S_t)*b2

                x_new = x + (1/2)*(k1 + k2)
                data.append(np.array((x_new)).reshape((1,2)).squeeze())
                x = x_new

            return np.array([data]).squeeze()


class data_Lorenz():
    def __init__(self, two_step = False):

        self.rho = 28.0
        # self.rho = 14
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.h = 0.01

        self.two_step = two_step

    def f(self, state):
        x, y, z = np.squeeze(state)
        return np.array([[self.sigma*(y - x), x*(self.rho - z) - y, x*y - self.beta*z]])

    def gen_data(self, trajectories=1):
        steps = 3000
        data = []
        x = np.array([[1.2,1.1,0.9]])
        if self.two_step:
                k1 = self.f(x)
                k2 = self.f(x + self.h*(k1/2))
                k3 = self.f(x + self.h*(k2/2))
                k4 = self.f(x + self.h*k3)
                x_step = x + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)

        for i in range(steps):

            if self.two_step:
                k1 = self.f(x_step)
                k2 = self.f(x_step + self.h*(k1/2))
                k3 = self.f(x_step + self.h*(k2/2))
                k4 = self.f(x_step + self.h*k3)
                x_new = x_step + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)
                data.append(np.array((x,x_step,x_new)).reshape((1,9)).squeeze())
                x = x_step
                x_step = x_new

            else:

                k1 = self.f(x)
                k2 = self.f(x + self.h*(k1/2))
                k3 = self.f(x + self.h*(k2/2))
                k4 = self.f(x + self.h*k3)
                x_new = x + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)
                data.append(np.array((x,x_new)).reshape((1,6)).squeeze())
                x = x_new

        if self.two_step:
            np.savetxt(mainPath/'data_Lorenz_stable_twostep.csv', data, delimiter=",")
        else:
            np.savetxt(mainPath/'data_Lorenz.csv', data, delimiter=",")

class data_VanderPol():
    def __init__(self, two_step = False):

        self.mu = 1.0
        self.h = 0.1

    def f(self, state):
        x, y = np.squeeze(state)
        return np.array([[self.mu*(x - (1/3)*x**3 - y), (1/self.mu)*x]])

    def gen_data(self, trajectories=1):
        data = []
        x = np.array([[4,2]])
        for i in range(400):

            k1 = self.f(x)
            k2 = self.f(x + self.h*(k1/2))
            k3 = self.f(x + self.h*(k2/2))
            k4 = self.f(x + self.h*k3)
            x_new = x + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)
            data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
            x = x_new

        np.savetxt(mainPath/'data_VanderPol_stable.csv', data, delimiter=",")



#see https://github.com/bhuvanakundumani/pytorch_Dataloader
class oversampdata(Dataset):

    def __init__(self, data, add_state = False, n=None):

        if add_state:
            self.inp_data = torch.FloatTensor(data.values.astype('float')[:,:2*n]).reshape((-1,1,2*n))
            self.out_data = torch.FloatTensor(data.values.astype('float')[:,2*n:]).reshape((-1,1,n))
        else:
            self.inp_data = torch.FloatTensor(data.values.astype('float')[:,:data.shape[1]//2]).reshape((-1,1,data.shape[1]//2))
            self.out_data = torch.FloatTensor(data.values.astype('float')[:,data.shape[1]//2:]).reshape((-1,1,data.shape[1]//2))

        # print(self.inp_data.shape)
    def __len__(self):
        return len(self.inp_data)

    def __getitem__(self, index):
        #target = self.out_data[ind]
        #data_val = self.data[index] [:-1]
        return self.inp_data[index], self.out_data[index]
