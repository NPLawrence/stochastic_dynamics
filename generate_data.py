#This is where we generate data for training/validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control.matlab
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

num_rollout = 100

class data_linear():
    def __init__(self):

        A = np.array([[1/2, 1],[0, 1/2]])
        A = A.transpose()
        data = []

        for i in range(num_rollout):

            x = np.random.uniform(2,10,size = (1,2))

            while np.dot(x,x.transpose()) > 1e-2:

                x_new = np.dot(x,A)
                data.append(np.array((x,x_new)).reshape((1,4)).squeeze())
                x = x_new

        np.savetxt("./datasets/data_linear.csv", data, delimiter=",")

        # data_linear = pd.DataFrame(data, columns=["x_k", "x_{k+1}"])
        #
        # data_linear.to_csv("./datasets/data_linear.csv")


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
