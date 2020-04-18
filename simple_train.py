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
import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

torch.set_grad_enabled(True)

import simple_model as model
import lyapunov_NN as L

import generate_data as gen_data

# gen_data.data_linear()

epochs = 100
batch_size = 128
learning_rate = 0.001

layer_sizes = np.array([2, 100, 1])

# fhat = model.fhat(np.array([2, 50, 50, 2]))
fhat = nn.Sequential(nn.Linear(2, 50), nn.Tanh(),
                    nn.Linear(50, 50), nn.Tanh(),
                    nn.Linear(50, 50), nn.Tanh(),
                    nn.Linear(50, 2))
V = L.MakePSD(L.ICNN(layer_sizes),2)
f_net = model.dynamics_simple(fhat,V)
# f_net = model.dynamics_nonincrease(fhat,V)
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

writer = SummaryWriter('runs/linear_experiment_2')
# get some random training images
# dataiter = iter(train_loader)
# input, output = dataiter.next()
# grid = torchvision.utils.make_grid(input)
# writer.add_image('sample_data', grid)
# writer.add_graph(f_net, input[0])


criterion = nn.MSELoss()

optimizer = optim.SGD(f_net.parameters(), lr=learning_rate)



f_net.train()

for epoch in range(epochs):

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):


        inputs, labels = data
        optimizer.zero_grad()
        outputs = f_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scalar('Loss', running_loss, epoch)
    for name, weight in f_net.named_parameters():
        writer.add_histogram(name, weight, epoch)
        # print(f'{name}')
        # writer.add_histogram(f'{name}.grad', weight.grad, epoch)

images, labels = next(iter(train_loader))
writer.add_graph(f_net, images)
print('Finished Training')
writer.close()
