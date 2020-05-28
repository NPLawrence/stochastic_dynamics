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

import convex_model as model
import rootfind_model

import lyapunov_NN as L

import generate_data

# generate_data.data_linear(two_step = True, add_noise = False)
# nonconvex = generate_data.data_nonConvex()
# nonconvex.gen_data()
Lorenz = generate_data.data_Lorenz()
Lorenz.gen_data(1)

epochs = 500
batch_size = 512
learning_rate = 0.005
n = 3
add_state = False

fhat = model.fhat(np.array([n, 25, 25, 25, n]), add_state = True)

layer_sizes = np.array([n, 50, 50, 1])
ICNN = L.ICNN(layer_sizes)
V = L.MakePSD(ICNN,n)
# layer_sizes = np.array([2, 50, 50, 50])
# V = L.Lyapunov_NN(L.PD_weights(layer_sizes))

# PATH_V = './saved_models/convex_V_VanderPol_stable.pth'
# PATH_f = './saved_models/convex_f_VanderPol_stable.pth'
# PATH_f = './saved_models/convex_f_linear_twostep.pth'
# PATH_f = './saved_models/convex_f_nonConvex.pth'
PATH_f = './saved_models/convex_f_Lorenz_unstable.pth'

# PATH_f = './saved_models/rootfind_f_nonConvex.pth'

# PATH_V = './saved_models/convex_V_Lorenz.pth'

# PATH_V = './saved_models/convex_V_Lorenz_stable.pth'
# PATH_f = './saved_models/convex_f_Lorenz_stable.pth'
# PATH_f = './saved_models/simple_f_Lorenz.pth'
# PATH_f = './saved_models/simple_fICNN2.pth'

# f_net = model.dynamics_convex(V, n, beta = 0.99)
f_net = model.dynamics_nonincrease(V, n)
# f_net = fhat

# f_net = rootfind_model.rootfind_module(V,n,is_training = True)

# f_net = model.dynamics_nonincrease(fhat,V)
# f_net = fhat

data = pd.read_csv("./datasets/data_Lorenz.csv")
# data = pd.read_csv("./datasets/data_nonConvex.csv")

# data = pd.read_csv("./datasets/data_Lorenz_stable_twostep.csv")
# data = pd.read_csv("./datasets/data_linear_twostep.csv")

if add_state:
    data_input = data.values[:,:n*2]
    data_output = data.values[:,n*2:]
else:
    data_input = data.values[:,:n]
    data_output = data.values[:,n:]
Trn_input,  Val_inp, Trn_target,Val_target = train_test_split(data_input, data_output, test_size=0.2,random_state=123)
# Train_data has our training dataset and Valid_data has our validation dataset.
Train_data = pd.concat([pd.DataFrame(Trn_input), pd.DataFrame(Trn_target)], axis=1)
Valid_data = pd.concat([pd.DataFrame(Val_inp), pd.DataFrame(Val_target)], axis=1)
# training and validation dataset
train_dataset = generate_data.oversampdata(Train_data,add_state = False,n = n)
valid_dataset = generate_data.oversampdata(Valid_data,add_state = False,n = n)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# writer = SummaryWriter('runs/simple_experiment_Lorenz_unstable')

writer = SummaryWriter('runs/noninc_experiment_Lorenz_unstable')

# writer = SummaryWriter('runs/rootfind_experiment_nonConvex')

# writer = SummaryWriter('runs/convex_experiment_linear_twostep')
# writer = SummaryWriter('runs/convex_experiment_VanderPol_stable')
# writer = SummaryWriter('runs/convex_experiment_Lorenz_stable')
# writer = SummaryWriter('runs/simple_experiment_Lorenz')


criterion = nn.MSELoss()

# optimizer = optim.SGD(f_net.parameters(), lr=learning_rate)
# optimizer = optim.RMSprop(f_net.parameters(), lr=learning_rate)
optimizer = optim.Adam(f_net.parameters(), lr=learning_rate)

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
        # print(f'{name}.grad', weight.grad)
    if epoch % 10 == 0:
        print("Epoch: ", epoch, "Running loss: ", running_loss)
# input, labels = next(iter(train_loader))
# writer.add_graph(f_net, input)
print('Finished Training')
writer.close()

# torch.save(ICNN.state_dict(), PATH_ICNN)
# torch.save(V.state_dict(), PATH_V)
torch.save(f_net.state_dict(), PATH_f)
