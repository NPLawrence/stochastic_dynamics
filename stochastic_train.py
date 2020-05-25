#A simple training procedure for deterministic models

#useful video https://www.youtube.com/watch?v=pSexXMdruFM&t=650s
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

torch.set_grad_enabled(True)

import stochastic_model
import stochastic_model_V3

import lyapunov_NN as L

import generate_data

generate_data.data_linear(add_noise = False)
# multiMod = generate_data.data_multiMod()
# multiMod.gen_data()

epochs = 200
batch_size = 512
learning_rate = 0.005


# fhat = model.fhat(np.array([2, 50, 50, 2]))
k = 2
n = 2
beta = 0.99
mode = 1
fhat = nn.Sequential(nn.Linear(n, 50), nn.ReLU(),
                    # nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 2*n*k))
layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)
# f_net = model.dynamics_simple(fhat,V)
# PATH_ICNN = './saved_models/simple_ICNN_stochastic.pth'
# PATH_V = './saved_models/simple_V_stochastic_noisyData.pth'
# PATH_f = './saved_models/simple_f_stochastic_noisyData.pth'
# PATH_V = './saved_models/simple_V_stochastic.pth'
PATH_f = './saved_models/convex_f_stochastic.pth'
# PATH_V = './saved_models/rootfind_V_stochastic.pth'
# PATH_f = './saved_models/rootfind_f_stochastic.pth'
# PATH_V = './saved_models/convex_V_stochastic_multiMod.pth'
# PATH_f = './saved_models/convex_f_stochastic_multiMod.pth'
# PATH_V = './saved_models/convex_V_stochastic_linear_V3.pth'
# PATH_f = './saved_models/convex_f_stochastic_linear_V3.pth'
# torch.save(f_net.state_dict(), PATH)

# f_net = model.dynamics_simple(fhat,V)
# f_net = model.dynamics_nonincrease(fhat,V)
# f_net = stochastic_model.stochastic_module(fhat, V, k)
f_net = stochastic_model.stochastic_module(fhat = fhat, V = V, n=n, k=k, mode = mode, beta = beta)

# f_net = fhat

data = pd.read_csv("./datasets/data_linear.csv")
# data = pd.read_csv("./datasets/data_linear_noise.csv")
# data = pd.read_csv("./datasets/data_multiMod.csv")


data_input = data.values[:,:2]
data_output = data.values[:,2:]
Trn_input,  Val_inp, Trn_target,Val_target = train_test_split(data_input, data_output, test_size=0.2,random_state=123)
# Train_data has our training dataset and Valid_data has our validation dataset.
Train_data = pd.concat([pd.DataFrame(Trn_input), pd.DataFrame(Trn_target)], axis=1)
Valid_data = pd.concat([pd.DataFrame(Val_inp), pd.DataFrame(Val_target)], axis=1)
# training and validation dataset
train_dataset = generate_data.oversampdata(Train_data)
valid_dataset = generate_data.oversampdata(Valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# writer = SummaryWriter('runs/convex_multiMod_experiment')
# writer = SummaryWriter('runs/stochastic_experiment_rootfind')
writer = SummaryWriter('runs/stochastic_experiment_linear')
# writer = SummaryWriter('runs/stochastic_experiment_linear_noisyData')


# get some random training images
# dataiter = iter(train_loader)
# input, output = dataiter.next()
# grid = torchvision.utils.make_grid(input)
# writer.add_image('sample_data', grid)
# writer.add_graph(f_net, input[0])


# criterion = nn.MSELoss()

optimizer = optim.Adam(f_net.parameters(), lr=learning_rate)

f_net.train()

for epoch in range(epochs):

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        optimizer.zero_grad()
        logp_labels = f_net(inputs, labels)
        loss = logp_labels
        # _,logp_target = f_net.target_distribution(V(labels))
        # loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scalar('Loss', running_loss, epoch)
    # for name, weight in f_net.named_parameters():
    #     # writer.add_histogram(name, weight, epoch)
    #     print(f'{name}', weight.grad)
        # writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    if epoch % 10 == 0:
        print("Epoch: ", epoch, "Running loss: ", running_loss)
# images, labels = next(iter(train_loader))
# writer.add_graph(f_net, images)
print('Finished Training')
writer.close()

# torch.save(ICNN.state_dict(), PATH_ICNN)
# torch.save(V.state_dict(), PATH_V)
torch.save(f_net.state_dict(), PATH_f)
