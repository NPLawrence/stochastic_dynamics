#This is where we generate data for training/validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control.matlab
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class data_linear():
    def __init__(self):

        A = np.array([[0.90, 1],[0, 0.90]])
        A = A.transpose()
        data = []

        X = np.linspace(-5,5,num=10)
        # X = np.array([-3, 3])

        for x1 in X:
            for x2 in X:

                # x = np.random.uniform(-3,3,size = (1,2))
                x = np.array([[x1,x2]])
                # x[0,0] = np.abs(x[0,0])
                # while np.dot(x,x.transpose()) < 0.5:
                #     x = np.random.uniform(-3,3,size = (1,2))


                # for i in range(10):
                while np.dot(x,x.transpose()) > 0.1:

                    x_new = np.dot(x,A)
                    data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
                    x = x_new

        np.savetxt("./datasets/data_linear.csv", data, delimiter=",")

        # data_linear = pd.DataFrame(data, columns=["x_k", "x_{k+1}"])
        #
        # data_linear.to_csv("./datasets/data_linear.csv")

class data_linear_noise():
    def __init__(self):

        A = np.array([[0.90, 1],[0, 0.90]])
        A = A.transpose()
        data = []

        X = np.linspace(-5,5,num=10)
        # X = np.linspace(-5,5,num=3)

        for x1 in X:
            for x2 in X:

                # x = np.random.uniform(-3,3,size = (1,2))
                x0 = np.array([[x1,x2]])
                # x[0,0] = np.abs(x[0,0])
                # while np.dot(x,x.transpose()) < 0.5:
                #     x = np.random.uniform(-3,3,size = (1,2))
                for j in range(1):
                    x = x0
                    for i in range(50):
                # while np.dot(x,x.transpose()) > 0.001:

                        x_new = np.dot(x,A) + 0.05*x*np.random.normal(0,1)
                        data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
                        x = x_new

        np.savetxt("./datasets/data_linear_noisy.csv", data, delimiter=",")

class data_Lorenz():
    def __init__(self):

        # self.rho = 28.0
        self.rho = 14.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0

        self.h = 0.01

    def f(self, state):
        x, y, z = np.squeeze(state)
        return np.array([[self.sigma*(y - x), x*(self.rho - z) - y, x*y - self.beta*z]])

    def gen_data(self, trajectories=1):
        data = []
        x = np.array([[1,1,1]])
        for i in range(2000):

            k1 = self.f(x)
            k2 = self.f(x + self.h*(k1/2))
            k3 = self.f(x + self.h*(k2/2))
            k4 = self.f(x + self.h*k3)
            x_new = x + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)
            data.append(np.array((x,x_new)).reshape((1,6)).squeeze())
            x = x_new
        print('hello')
        np.savetxt("./datasets/data_Lorenz_stable.csv", data, delimiter=",")

class data_VanderPol():
    def __init__(self):

        self.mu = 1

        self.h = 0.01

    def f(self, state):
        x, y = np.squeeze(state)
        return np.array([[self.mu*(x - (1/3)*x^3 - y), (1/self.mu)*x]])

    def gen_data(self, trajectories=1):
        data = []
        x = np.array([[1,1]])
        for i in range(100):
            print('hello')

            k1 = self.f(x)
            k2 = self.f(x + self.h*(k1/2))
            k3 = self.f(x + self.h*(k2/2))
            k4 = self.f(x + self.h*k3)
            x_new = x + (self.h/6)*(k1 + 2*k2+ 2*k3 + k4)
            data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
            x = x_new

        np.savetxt("./datasets/data_VanderPol_stable.csv", data, delimiter=",")


#see https://github.com/bhuvanakundumani/pytorch_Dataloader
class oversampdata(Dataset):

    def __init__(self, data):

        self.inp_data = torch.FloatTensor(data.values.astype('float')[:,:data.shape[1]//2]).reshape((-1,1,data.shape[1]//2))
        self.out_data = torch.FloatTensor(data.values.astype('float')[:,data.shape[1]//2:]).reshape((-1,1,data.shape[1]//2))

        # print(self.inp_data.shape)
    def __len__(self):
        return len(self.inp_data)

    def __getitem__(self, index):
        #target = self.out_data[ind]
        #data_val = self.data[index] [:-1]
        return self.inp_data[index], self.out_data[index]
