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

from rootfind_autograd import rootfind_module




import simple_model as model
import generate_data as gen_data

torch.set_grad_enabled(True)

# gen_data.data_linear()

epochs = 1
batch_size = 1
learning_rate = 0.001


# fhat = model.fhat(np.array([2, 50, 50, 2]))
fhat = nn.Sequential(nn.Linear(2, 50), nn.Tanh(),
                    nn.Linear(50, 50), nn.Tanh(),
                    nn.Linear(50, 50), nn.Tanh(),
                    nn.Linear(50, 2))

layer_sizes = np.array([2, 100, 1])

V = model.MakePSD(model.ICNN(layer_sizes),2)
# input = torch.randn(1,2, requires_grad=True)
# output = V(torch.randn(1,2))
# # print(torch.autograd.grad(V(input),input))
# output.backward(torch.ones_like(output))

# for name, weight in V.named_parameters():
#     print(name, weight.grad)
f_net = model.rootfind_module(fhat,V)

# f_net = model.dynamics_rootfind(fhat,V)



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

criterion = nn.MSELoss()

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
        optimizer.zero_grad()
        # self.V,self.fhat,target,root,x
        # outputs = f_net(inputs)
        # outputs = rootfind(V, fhat, V(inputs), inputs)
        # print(i)
        outputs_f = f_net(inputs)
        # print(outputs_f[0])
        loss = criterion(outputs_f, labels)
        # print(V(inputs).backward(torch.ones_like(V(inputs))))
        loss.backward()
        # print(list(f_net.parameters())[0].grad)
        optimizer.step()
        running_loss += loss.item()

    writer.add_scalar('Loss', running_loss, epoch)
    for name, weight in f_net.named_parameters():
        writer.add_histogram(name, weight, epoch)
        print(name, weight.grad)

        # print(f'{name}')
        # writer.add_histogram(f'{name}.grad', weight.grad, epoch)
    print(epoch)


print('Finished Training')
writer.close()
