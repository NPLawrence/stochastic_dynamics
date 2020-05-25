#This is where we generate data for training/validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control.matlab
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class data_linear():
    def __init__(self, two_step = False, add_noise = False):

        A = np.array([[0.90, 1],[0, 0.90]])
        A = A.transpose()
        data = []

        X = np.linspace(-5,5,num=10)
        # X = np.array([-3, 3])

        for x1 in X:
            for x2 in X:

                x = np.array([[x1,x2]])
                if two_step:
                    if add_noise:
                        x_step = np.dot(x,A) + 0.05*x*np.random.normal(0,1)
                    else:
                        x_step = np.dot(x,A)

                for i in range(50):

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
                np.savetxt("./datasets/data_linear_twostep_noise.csv", data, delimiter=",")
            else:
                np.savetxt("./datasets/data_linear_twostep.csv", data, delimiter=",")
        else:
            if add_noise:
                np.savetxt("./datasets/data_linear_noise.csv", data, delimiter=",")
            else:
                np.savetxt("./datasets/data_linear.csv", data, delimiter=",")

class data_Lorenz():
    def __init__(self, two_step = False):

        # self.rho = 28.0
        self.rho = 14
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.h = 0.01

        self.two_step = two_step

    def f(self, state):
        x, y, z = np.squeeze(state)
        return np.array([[self.sigma*(y - x), x*(self.rho - z) - y, x*y - self.beta*z]])

    def gen_data(self, trajectories=1):
        steps = 2000
        data = []
        x = np.array([[1,1,1]])
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
            np.savetxt("./datasets/data_Lorenz_stable_twostep.csv", data, delimiter=",")
        else:
            np.savetxt("./datasets/data_Lorenz_stable.csv", data, delimiter=",")

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

        np.savetxt("./datasets/data_VanderPol_stable.csv", data, delimiter=",")

class data_multiMod():
    def __init__(self, two_step = False):

        self.alpha = 0.5
        self.beta = 25.0
        self.gamma = 8.0

        self.h = 0.1

    def f(self, x, i):
        return self.alpha*x + self.beta*x/(1 + x**2) + self.gamma*np.cos(1.2*(i-1)) + np.random.normal(0,0.1)

    def f_mean(self, x, i):
        return self.alpha*x + self.beta*x/(1 + x**2) + self.gamma*np.cos(1.2*(i-1))

    def gen_data(self, trajectories = 1, steps = 200, train_data = True, x = None):
        data = []
        if train_data:
            for j in range(1):
                x = np.random.normal(0,0.1)
                for i in range(steps):
                    x_new = self.f(x,i)
                    data.append(np.array((x,x_new)).reshape((1,2)).squeeze())
                    x = x_new

            np.savetxt("./datasets/data_multiMod.csv", data, delimiter=",")

        else:
            data.append(np.array((x)).reshape((1,1)).squeeze())
            x_mean = x
            data_mean = []
            data_mean.append(np.array((x_mean)).reshape((1,1)).squeeze())
            for i in range(steps):
                x_new = self.f(x,i)
                x_mean_new = self.f_mean(x_mean,i)
                data.append(np.array((x_new)).reshape((1,1)).squeeze())
                data_mean.append(np.array((x_mean_new)).reshape((1,1)).squeeze())
                x = x_new
                x_mean = x_mean_new
            return data, data_mean



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
