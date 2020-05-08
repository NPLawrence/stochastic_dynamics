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

        X = np.linspace(-5,5,num=15)
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

        X = np.linspace(-5,5,num=15)
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
