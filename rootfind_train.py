#training for root-finding approach

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

# from rootfind_autograd import rootfind_module

import simple_model
import rootfind_model as model
import lyapunov_NN as L

import generate_data as gen_data

torch.set_grad_enabled(True)

gen_data.data_linear()

epochs = 15
batch_size = 64
learning_rate = 0.001


# fhat = simple_model.fhat(np.array([2, 50, 50, 2]))
fhat = nn.Sequential(nn.Linear(2, 50), nn.Tanh(),
                    nn.Linear(50, 50), nn.Tanh(),
                    nn.Linear(50, 2))
layer_sizes = np.array([2, 50, 1])

V = L.MakePSD(L.ICNN(layer_sizes),2)
# input = torch.randn(1,2, requires_grad=True)
# output = V(torch.randn(1,2))
# # print(torch.autograd.grad(V(input),input))
# output.backward(torch.ones_like(output))

# for name, weight in V.named_parameters():
#     print(name, weight.grad)
f_net = model.rootfind_module(fhat,V)

# f_net = fhat

data = pd.read_csv("./datasets/data_linear.csv")

data_input = data.values[:,:2]
data_output = data.values[:,2:]
Trn_input,  Val_inp, Trn_target,Val_target = train_test_split(data_input, data_output, test_size=0.2,random_state=123)
# Train_data has our training dataset and Valid_data has our validation dataset.
Train_data = pd.concat([pd.DataFrame(Trn_input), pd.DataFrame(Trn_target)], axis=1)
Valid_data = pd.concat([pd.DataFrame(Val_inp), pd.DataFrame(Val_target)], axis=1)
# training and validation dataset
train_dataset = gen_data.oversampdata(Train_data)
valid_dataset = gen_data.oversampdata(Valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

writer = SummaryWriter('runs/linear_experiment_1')

criterion_usual = nn.MSELoss()
criterion_rootfind = nn.MSELoss()
criterion = criterion_usual

#The optimization is the key step
# rootfind = rootfind_module.rootfind_train.apply
optimizer = optim.Adam(f_net.parameters(), lr=learning_rate)

images, labels = next(iter(train_loader))
writer.add_graph(f_net, images)

# f_net.train()

for epoch in range(epochs):

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        inputs_usual, labels_usual, inputs_rootfind, labels_rootfind = f_net.split_rootfind(inputs, labels)
        optimizer.zero_grad()
        if inputs_usual.shape[0] == 0:
            outputs_rootfind = f_net(inputs_rootfind)
            # print('0', outputs_rootfind.shape, labels_rootfind.shape)
            loss = criterion(outputs_rootfind, labels_rootfind)
        elif inputs_rootfind.shape[0] == 0:
            outputs_usual = fhat(inputs_usual)
            # print('1', outputs_usual.shape, labels_usual.shape)
            loss = criterion(outputs_usual, labels_usual)
        else:
            outputs_rootfind = f_net(inputs_rootfind)
            loss_rootfind = criterion(outputs_rootfind, labels_rootfind)
            outputs_usual = fhat(inputs_usual)
            loss_usual = criterion(outputs_usual, labels_usual)
            loss = loss_rootfind + loss_usual

        # outputs = fhat(inputs)
        # loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scalar('Loss', running_loss, epoch)
    for name, weight in f_net.named_parameters():
        writer.add_histogram(name, weight, epoch)

    # for name, weight in V.f.named_parameters():
    #
    #     print(name, weight.grad)




        # print(f'{name}')
        # writer.add_histogram(f'{name}.grad', weight.grad, epoch)
    print("Epoch: ", epoch, "Running loss: ", running_loss)


print('Finished Training')
writer.close()
