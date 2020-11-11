# A generic script for training locally

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import modules.convex_model as convex_model
import modules.rootfind_model as rootfind_model
import modules.stochastic_model as stochastic_model
import modules.lyapunov_NN as L

import generate_data as gen_data

# Create Lyapunov NN, create stable model (convex_model, rootfind_model, stochastic_model)
#   Optional: define nominal model fhat (the above modules automatically do this)
n = 2 # state dimension
is_stochastic = True
layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)
if is_stochastic:
    f_net = stochastic_model.dynamics_model(V,n,mode=1,is_training = True)
else:
    f_net = rootfind_model.dynamics_model(V,n,is_training = True)

# Specify epochs, batch_size, learning_rate, loss function, optimizer, state dimension
epochs = 10
batch_size = 512
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(f_net.parameters(), lr=learning_rate)
# lr_lambda = lambda epoch: 1/np.log(epoch+2) # Optional learning rate scheduler
# lr_scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda, last_epoch=-1)

# Set paths, generate/load/split data
experiment = 'exp_name' + '.pth'
PATH_f_net = './saved_models/' + experiment # TODO: PyTorch's torch.save isn't compatible w/pathlib
gen_data.data_linear()
data = pd.read_csv(Path('./datasets/data_linear.csv'))
data_input = data.values[:,:2]
data_output = data.values[:,2:]
Trn_input,  Val_inp, Trn_target,Val_target = train_test_split(data_input, data_output, test_size=0.2,random_state=123)
#   Train_data has our training dataset and Valid_data has our validation dataset.
Train_data = pd.concat([pd.DataFrame(Trn_input), pd.DataFrame(Trn_target)], axis=1)
Valid_data = pd.concat([pd.DataFrame(Val_inp), pd.DataFrame(Val_target)], axis=1)
#   training and validation dataset
train_dataset = gen_data.oversampdata(Train_data)
valid_dataset = gen_data.oversampdata(Valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

summaryPath = Path('./runs/'+ experiment)
summaryPath.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(summaryPath)

f_net.train()
for epoch in range(epochs):

    running_loss = 0.0

    for i, data in enumerate(train_loader):

        inputs, labels = data
        optimizer.zero_grad()
        if is_stochastic:
            loss = f_net(inputs, labels)
        else:
            outputs = f_net(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # lr_scheduler.step()
    writer.add_scalar('Loss', running_loss, epoch)
    for name, weight in f_net.named_parameters():
        writer.add_histogram(name, weight, epoch)

    if epoch % 10 == 0:
        print("Epoch: ", epoch, "Running loss: ", running_loss)

print('Finished Training')

writer.close()
torch.save(f_net.state_dict(), PATH_f_net)
