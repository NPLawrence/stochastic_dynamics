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

import rootfind_model as model
import lyapunov_NN as L

import generate_data as gen_data

torch.set_grad_enabled(True)

gen_data.data_linear()

epochs = 200
batch_size = 512
learning_rate = 0.001
n = 2


# fhat = simple_model.fhat(np.array([2, 50, 50, 2]))


fhat = nn.Sequential(nn.Linear(n, 25), nn.ReLU(),
                    nn.Linear(25, 25), nn.ReLU(),
                    nn.Linear(25, n))
layer_sizes = np.array([n, 25, 25, 1])

ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)

# layer_sizes = np.array([2, 50, 50, 50])
# V = L.Lyapunov_NN(L.PD_weights(layer_sizes))
# input = torch.randn(1,2, requires_grad=True)
# output = V(torch.randn(1,2))
# # print(torch.autograd.grad(V(input),input))
# output.backward(torch.ones_like(output))

# for name, weight in V.named_parameters():
#     print(name, weight.grad)
f_net = model.rootfind_module(V,n,is_training = True)

# PATH_ICNN = './saved_models/rootfind_ICNN.pth'
PATH_V = './saved_models/rootfind_V1_TEST.pth'
PATH_f = './saved_models/rootfind_f1_TEST.pth'

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

writer = SummaryWriter('runs/linear_experiment_rootfind1_TEST')

criterion = nn.MSELoss()


#The optimization is the key step
# rootfind = rootfind_module.rootfind_train.apply
# optimizer = optim.SGD(f_net.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.RMSprop(f_net.parameters(), lr=learning_rate)
optimizer = optim.Adam(f_net.parameters(), lr=learning_rate)

# f_net.train()

for epoch in range(epochs):

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        # inputs_usual, labels_usual, inputs_rootfind, labels_rootfind = f_net.split_rootfind(inputs, labels)
        optimizer.zero_grad()
        outputs = f_net(inputs)
        loss_print = criterion(outputs, labels)
        # V_loss = torch.mean(V(labels) - V(inputs))
        V_loss = torch.mean(V(labels))
        loss = loss_print
        # if inputs_usual.shape[0] == 0:
        #     outputs_rootfind = f_net(inputs_rootfind)
        #     # print('0', outputs_rootfind.shape, labels_rootfind.shape)
        #     V_loss = torch.mean(V(labels_rootfind) - V(inputs_rootfind))
        #     loss_print = criterion(outputs_rootfind, labels_rootfind)
        #     loss = loss_print + V_loss
        #     # loss = loss_print
        #
        # elif inputs_rootfind.shape[0] == 0:
        #     outputs_usual = fhat(inputs_usual)
        #     V_loss = torch.mean(V(labels_usual) - V(inputs_usual))
        #     # print('1', outputs_usual.shape, labels_usual.shape)
        #     loss_print = criterion(outputs_usual, labels_usual)
        #     loss = loss_print + V_loss
        #     # loss = loss_print
        # else:
        #     outputs_rootfind = f_net(inputs_rootfind)
        #     loss_rootfind = criterion(outputs_rootfind, labels_rootfind)
        #     outputs_usual = fhat(inputs_usual)
        #     loss_usual = criterion(outputs_usual, labels_usual)
        #
        #     V_loss = torch.mean(V(labels) - V(inputs))
        #     loss_print = loss_rootfind + loss_usual
        #     loss = loss_print + V_loss
            # loss = loss_print

        # print(i, epoch)
        # outputs = fhat(inputs)
        # loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss_print.item()

    # for name, weight in f_net.named_parameters():
    #
    #     print(name, weight)

    writer.add_scalar('Loss', running_loss, epoch)
    for name, weight in f_net.named_parameters():
        writer.add_histogram(name, weight, epoch)
        # print(f'{name}.grad', weight.grad)




        # print(f'{name}')
        # writer.add_histogram(f'{name}.grad', weight.grad, epoch)
    print("Epoch: ", epoch, "Running loss: ", running_loss)
# for name, weight in f_net.named_parameters():
#     print(name, weight)

print('Finished Training')

# inputs, outputs = next(iter(train_loader))
# inputs_usual, labels_usual, inputs_rootfind, labels_rootfind = f_net.split_rootfind(inputs, outputs)
# writer.add_graph(f_net, inputs_rootfind)
writer.close()

# torch.save(ICNN.state_dict(), PATH_ICNN)
torch.save(V.state_dict(), PATH_V)
torch.save(fhat.state_dict(), PATH_f)
