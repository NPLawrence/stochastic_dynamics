#A simple training procedure for deterministic models

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


import simple_model as model
import generate_data as gen_data

# gen_data.data_linear()

layer_sizes = np.array([2, 100, 1])

fhat = model.fhat(np.array([2, 50, 50, 2]))
V = model.MakePSD(model.ICNN(layer_sizes),2)
# f = model.dynamics_simple(fhat,V)
f_net = model.dynamics_nonincrease(fhat,V)
# f = model.dynamics_rootfind(fhat,V)
params = list(f_net.parameters())

criterion = nn.MSELoss()

epochs = 100

data = pd.read_csv("./datasets/data_linear.csv")

# print(data.values)
# data = data.columns.tolist()
# data = torch.FloatTensor(data.values.astype('float'))
# input, output = torch.FloatTensor(data['x_k'].values.astype('float')), torch.FloatTensor(data['x_{k+1}'].values.astype('float'))

# input = torch.FloatTensor(input.values.astype('float'))
# data = pd.read_csv("./datasets/data_linear.pkl")
# print(data.columns.values)
# tmp = data
# result = torch.from_numpy(tmp)
# x = data.values[0]
# y = data.values[1]
# print(x.dtype)
# result = torch.tensor(data.values)
# print(result)
# input = data[:,0]
# output = data[:,1]
data_input = data.values[:,:2]
data_output = data.values[:,2:]
Trn_input,  Val_inp, Trn_target,Val_target = train_test_split(data_input, data_output, test_size=0.2,random_state=123)
# Train_data has our training dataset and Valid_data has our validation dataset.
Train_data = pd.concat([pd.DataFrame(Trn_input), pd.DataFrame(Trn_target)], axis=1)
Valid_data = pd.concat([pd.DataFrame(Val_inp), pd.DataFrame(Val_target)], axis=1)
# training and validation dataset
train_dataset = gen_data.oversampdata(Train_data)
valid_dataset = gen_data.oversampdata(Valid_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)


# gen_data.oversampdata(data)
# input_train, input_test, output_train, output_test =\
#     train_test_split(input, output, test_size=0.20, random_state=42)

# print(input_train[0])
# input_train = torch.FloatTensor(input_train.astype('float'))
# x = torch.from_numpy(input_train[0]).requires_grad_(True)
# x= Variable(torch.from_numpy(input_train[0]).float(), requires_grad=False)
# y = Variable(torch.from_numpy(output_train[0]).float(), requires_grad=False)
# x = input_train[0]
#
# x,y = train_dataset.__getitem__(2)
# out = f_net(x)
#
# loss = criterion(x,y)
#
# f_net.zero_grad()
# loss.backward()
optimizer = optim.SGD(f_net.parameters(), lr=0.001)
# optimizer.step()
f_net.train()

for epoch in range(2):

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = f_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

print('Finished Training')
